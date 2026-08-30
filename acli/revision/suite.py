"""Checkpointed orchestration for the major-revision experiment suite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "state" / "revision_pipeline_status.json"
CHECKPOINT_DIR = REPO_ROOT / "state" / "checkpoints"

STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "finite_length_validation",
        "experiment/05_finite_length_simulation_validation.py",
        ("result/table/05_finite_length_simulation_validation.csv",),
    ),
    (
        "shared_target_accumulation",
        "experiment/06_shared_target_accumulation.py",
        (
            "result/table/06_accumulation_profiles.csv",
            "result/table/06_redundancy_representative_cases.csv",
            "result/table/06_accumulation_stress_summary.csv",
        ),
    ),
    (
        "sharp_frontier_and_tail_metrics",
        "experiment/07_sharp_frontier_and_tail_metrics.py",
        (
            "result/table/07_frontier_comparison.csv",
            "result/table/07_same_J_tail_comparison.csv",
            "result/table/07_constant_gap.csv",
            "result/table/07_tail_metric_summary.csv",
        ),
    ),
    (
        "finite_K_convergence",
        "experiment/08_finite_K_convergence.py",
        (
            "result/table/08_finite_K_convergence.csv",
            "result/table/08_finite_K_summary.csv",
        ),
    ),
    (
        "weak_screening_validity",
        "experiment/09_weak_screening_validity.py",
        (
            "result/table/09_weak_screening_validity.csv",
            "result/table/09_calibration_audit.csv",
            "result/table/09_auc_epsilon_J_mapping.csv",
        ),
    ),
)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_profile(profile: str | Path, root: Path = REPO_ROOT) -> tuple[dict[str, Any], Path]:
    """Load a named profile or an explicit YAML path."""
    candidate = Path(profile)
    if not candidate.suffix:
        candidate = root / "configs" / f"revision_{profile}.yaml"
    elif not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Revision profile not found: {candidate}")
    config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Profile must contain a YAML mapping: {candidate}")
    return config, candidate


def config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(root: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_manifest(paths: Sequence[str], root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Expected non-empty stage output: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    return manifest


def output_write_tokens(
    paths: Sequence[str], root: Path = REPO_ROOT
) -> dict[str, tuple[int, int] | None]:
    """Return inode/mtime tokens used to prove that a stage rewrote every output."""
    tokens: dict[str, tuple[int, int] | None] = {}
    for relative in paths:
        path = root / relative
        if path.is_file():
            stat = path.stat()
            tokens[relative] = (int(stat.st_ino), int(stat.st_mtime_ns))
        else:
            tokens[relative] = None
    return tokens


def checkpoint_is_valid(
    checkpoint: Path,
    expected: Sequence[str],
    digest: str,
    root: Path = REPO_ROOT,
) -> bool:
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("config_sha256") != digest:
            return False
        if payload.get("git_commit") != git_commit(root):
            return False
        current = output_manifest(expected, root)
        recorded = payload.get("outputs", [])
        return current == recorded
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_suite(profile: str, workers: int | None = None, resume: bool = False) -> dict[str, Any]:
    """Run each experiment as an isolated process with atomic stage checkpoints."""
    config, config_path = load_profile(profile)
    profile_name = str(config.get("profile", Path(profile).stem.replace("revision_", "")))
    requested_workers = int(workers if workers is not None else config.get("workers", 1))
    safe_workers = max(1, min(requested_workers, int(config.get("workers", requested_workers))))
    digest = config_digest(config)
    commit = git_commit()
    try:
        profile_path = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        profile_path = str(config_path.resolve())
    status: dict[str, Any] = {
        "profile": profile_name,
        "profile_path": profile_path,
        "config_sha256": digest,
        "git_commit": commit,
        "workers": safe_workers,
        "status": "running",
        "stages": {},
    }
    atomic_write_json(STATE_PATH, status)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "MPLBACKEND": "Agg",
            "ACI_WORKERS": str(safe_workers),
        }
    )

    for stage, script, expected in STAGES:
        checkpoint = CHECKPOINT_DIR / f"{profile_name}.{stage}.json"
        if resume and checkpoint_is_valid(checkpoint, expected, digest):
            status["stages"][stage] = {"status": "skipped_valid_checkpoint"}
            atomic_write_json(STATE_PATH, status)
            print(f"[suite] {stage}: valid checkpoint, skipped", flush=True)
            continue

        status["stages"][stage] = {"status": "running", "script": script}
        atomic_write_json(STATE_PATH, status)
        command = [
            sys.executable,
            str(REPO_ROOT / script),
            "--profile",
            str(config_path),
            "--workers",
            str(safe_workers),
        ]
        print(f"[suite] {stage}: {' '.join(command)}", flush=True)
        try:
            before_tokens = output_write_tokens(expected)
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
            after_tokens = output_write_tokens(expected)
            stale = [path for path in expected if after_tokens[path] == before_tokens[path]]
            if stale:
                raise RuntimeError(
                    f"stage exited successfully without rewriting expected outputs: {stale}"
                )
            outputs = output_manifest(expected)
            payload = {
                "stage": stage,
                "status": "completed",
                "profile": profile_name,
                "config_sha256": digest,
                "git_commit": commit,
                "outputs": outputs,
            }
            atomic_write_json(checkpoint, payload)
            status["stages"][stage] = payload
            atomic_write_json(STATE_PATH, status)
        except BaseException as exc:
            status["status"] = "failed"
            status["stages"][stage] = {
                "status": "failed",
                "script": script,
                "error": f"{type(exc).__name__}: {exc}",
            }
            atomic_write_json(STATE_PATH, status)
            raise

    status["status"] = "experiments_completed"
    atomic_write_json(STATE_PATH, status)
    return status


__all__ = [
    "CHECKPOINT_DIR",
    "REPO_ROOT",
    "STATE_PATH",
    "STAGES",
    "atomic_write_json",
    "checkpoint_is_valid",
    "config_digest",
    "git_commit",
    "load_profile",
    "output_manifest",
    "output_write_tokens",
    "run_suite",
]
