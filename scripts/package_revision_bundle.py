#!/usr/bin/env python3
"""Create and integrity-check the major-revision reproducibility bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "ACI_MAJOR_REVISION_REPRODUCIBILITY_BUNDLE.zip"
SHA_PATH = ROOT / "ACI_MAJOR_REVISION_REPRODUCIBILITY_BUNDLE.zip.sha256"


def _run_capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def _candidate_files() -> list[Path]:
    exact = [
        ROOT / "INTEGRATION_REPORT.md",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "pyproject.toml",
        ROOT / "paper_figures" / "manifest.yaml",
        ROOT / "source_diff.patch",
        ROOT / "changed_files.txt",
    ]
    directories = [
        ROOT / "environment",
        ROOT / "result" / "table",
        ROOT / "result" / "figure",
        ROOT / "result" / "artifact",
        ROOT / "logs",
        ROOT / "state",
        ROOT / "acli" / "revision",
        ROOT / "configs",
        ROOT / "tests",
    ]
    globs = [
        "experiment/06_*.py",
        "experiment/07_*.py",
        "experiment/08_*.py",
        "experiment/09_*.py",
        "experiment/10_*.py",
        "notebook/06_major_revision_paper_figures.ipynb",
        "scripts/*revision*",
    ]
    files = {path for path in exact if path.is_file()}
    for directory in directories:
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file())
    for pattern in globs:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    excluded_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".revision_cache"}
    return sorted(
        path for path in files
        if not excluded_parts.intersection(path.relative_to(ROOT).parts)
        and path != BUNDLE
        and path != SHA_PATH
    )


def _assert_formal_validation(validation: dict) -> None:
    execution = validation.get("execution")
    state = execution.get("state", {}) if isinstance(execution, dict) else {}
    checks = validation.get("checks", [])
    if validation.get("status") != "passed":
        raise RuntimeError("Refusing to package outputs that have not passed validation")
    if not execution or state.get("profile") != "paper":
        raise RuntimeError("Refusing to package a non-formal or smoke-only validation report")
    if state.get("status") != "experiments_completed":
        raise RuntimeError("Refusing to package an incomplete paper-profile experiment state")
    if not checks or any(not bool(item.get("passed")) for item in checks):
        raise RuntimeError("Formal validation report contains a failed or missing check")


def _required_payload(validation: dict) -> list[Path]:
    numerical = validation.get("numerical", {})
    table_paths = [ROOT / relative for relative in numerical.get("tables", {})]
    figure_paths = [ROOT / item["path"] for item in validation.get("figures", [])]
    required = [
        ROOT / "INTEGRATION_REPORT.md",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "pyproject.toml",
        ROOT / "paper_figures" / "manifest.yaml",
        ROOT / "notebook" / "06_major_revision_paper_figures.ipynb",
        ROOT / "source_diff.patch",
        ROOT / "changed_files.txt",
        ROOT / "environment" / "base_commit.txt",
        ROOT / "environment" / "run_commit.txt",
        ROOT / "environment" / "final_commit.txt",
        ROOT / "logs" / "aci_major_revision_integration.log",
        ROOT / "state" / "revision_pipeline_status.json",
        ROOT / "result" / "artifact" / "output_validation.json",
        ROOT / "result" / "artifact" / "figure_manifest.json",
        ROOT / "result" / "artifact" / "environment_manifest.txt",
        ROOT / "result" / "artifact" / "pytest.xml",
        ROOT / "result" / "artifact" / "legacy_regressions.json",
        ROOT / "result" / "artifact" / "06_major_revision_paper_figures.executed.ipynb",
        ROOT / "result" / "artifact" / "06_major_revision_paper_figures.executed.html",
        *table_paths,
        *figure_paths,
    ]
    for pattern in (
        "acli/revision/*.py",
        "experiment/0[6-9]_*.py",
        "experiment/10_*.py",
        "scripts/*revision*",
        "configs/revision_*.yaml",
        "tests/test_revision_*.py",
    ):
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            raise RuntimeError(f"Required source-snapshot pattern has no matches: {pattern}")
        required.extend(matches)
    return required


def _validated_hashes(validation: dict) -> dict[str, str]:
    hashes = {
        relative: record["sha256"]
        for relative, record in validation["numerical"]["tables"].items()
    }
    hashes.update({item["path"]: item["sha256"] for item in validation["figures"]})
    hashes.update(
        {
            record["path"]: record["sha256"]
            for record in validation["execution"]["artifacts"].values()
        }
    )
    return hashes


def main() -> None:
    # Revalidate immediately before selecting files so a stale passed report
    # cannot authorize packaging subsequently changed tables or figures.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_revision_outputs.py")],
        cwd=ROOT,
        check=True,
    )
    validation_path = ROOT / "result" / "artifact" / "output_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    _assert_formal_validation(validation)

    live_state = json.loads(
        (ROOT / "state" / "revision_pipeline_status.json").read_text(encoding="utf-8")
    )
    if live_state.get("status") != "experiments_completed":
        raise RuntimeError("Live experiment state is incomplete")
    if live_state.get("pipeline_status") not in {"running", "completed"}:
        raise RuntimeError("Live end-to-end pipeline state is not packageable")
    if live_state.get("pipeline_phase") not in {"packaging", "complete"}:
        raise RuntimeError("Packaging is only allowed during or after the formal packaging phase")

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError(f"Refusing to package a dirty source tree:\n{dirty}")

    base = (ROOT / "environment" / "base_commit.txt").read_text(encoding="utf-8").strip()
    (ROOT / "changed_files.txt").write_text(
        _run_capture(["git", "diff", "--name-status", base, "HEAD"]), encoding="utf-8"
    )
    (ROOT / "source_diff.patch").write_text(
        _run_capture(["git", "diff", "--binary", base, "HEAD"]), encoding="utf-8"
    )

    required = _required_payload(validation)
    missing = [
        str(path.relative_to(ROOT))
        for path in required
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise RuntimeError(f"Required bundle payload is absent or empty: {missing}")
    candidates = _candidate_files()
    candidate_set = set(candidates)
    omitted = [str(path.relative_to(ROOT)) for path in required if path not in candidate_set]
    if omitted:
        raise RuntimeError(f"Required bundle payload was omitted by source selection: {omitted}")

    validated_hashes = _validated_hashes(validation)
    changed_after_validation = [
        relative
        for relative, expected in validated_hashes.items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected
    ]
    if changed_after_validation:
        raise RuntimeError(
            f"Validated artifacts changed before bundle creation: {changed_after_validation}"
        )

    temporary_directory = ROOT / ".revision_cache"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    zip_descriptor, zip_name = tempfile.mkstemp(
        prefix="ACI_MAJOR_REVISION_", suffix=".zip.tmp", dir=temporary_directory
    )
    os.close(zip_descriptor)
    temporary_bundle = Path(zip_name)
    temporary_sha = temporary_directory / f"{SHA_PATH.name}.{os.getpid()}.tmp"
    try:
        with zipfile.ZipFile(
            temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in candidates:
                archive.write(path, path.relative_to(ROOT).as_posix())

        with zipfile.ZipFile(temporary_bundle, "r") as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP integrity failure at {bad}")
            changed_in_archive = [
                relative
                for relative, expected in validated_hashes.items()
                if hashlib.sha256(archive.read(relative)).hexdigest() != expected
            ]
            if changed_in_archive:
                raise RuntimeError(
                    f"Bundled artifacts differ from formal validation: {changed_in_archive}"
                )

        unzip = shutil.which("unzip")
        if unzip is None:
            raise RuntimeError("The required external ZIP integrity checker 'unzip' is unavailable")
        subprocess.run([unzip, "-t", str(temporary_bundle)], cwd=ROOT, check=True)

        digest = hashlib.sha256(temporary_bundle.read_bytes()).hexdigest()
        temporary_sha.write_text(f"{digest}  {BUNDLE.name}\n", encoding="utf-8")
        os.replace(temporary_bundle, BUNDLE)
        try:
            os.replace(temporary_sha, SHA_PATH)
        except BaseException:
            SHA_PATH.unlink(missing_ok=True)
            raise
    finally:
        temporary_bundle.unlink(missing_ok=True)
        temporary_sha.unlink(missing_ok=True)
    print(f"Created {BUNDLE}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
