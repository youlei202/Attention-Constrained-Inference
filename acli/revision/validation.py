"""Validation of numerical tables, notebook artifacts, figures, and provenance."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

from .suite import (
    CHECKPOINT_DIR,
    STAGES,
    atomic_write_json,
    checkpoint_is_valid,
    config_digest,
    git_commit,
    load_profile,
)


METADATA_COLUMNS = {"seed", "configuration", "git_commit", "method"}
NOTEBOOK_STEM = "06_major_revision_paper_figures.executed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_tables() -> list[str]:
    outputs: list[str] = []
    for _, _, stage_outputs in STAGES:
        outputs.extend(stage_outputs)
    return outputs


def _check(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def _finite_numeric_frame(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            bad.append(str(column))
    return not bad, bad


def validate_table_set(
    root: Path,
    checks: list[dict[str, Any]],
    require_paper_profile: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for relative in required_tables():
        path = root / relative
        _check(path.is_file() and path.stat().st_size > 0, f"non-empty table: {relative}", checks)
        frame = pd.read_csv(path)
        _check(not frame.empty, f"rows present: {relative}", checks)
        missing_value_columns = frame.columns[frame.isna().any()].tolist()
        _check(
            not missing_value_columns,
            f"no missing values: {relative} (bad={missing_value_columns})",
            checks,
        )
        missing = sorted(METADATA_COLUMNS.difference(frame.columns))
        _check(not missing, f"provenance columns: {relative} (missing={missing})", checks)
        finite, bad = _finite_numeric_frame(frame)
        _check(finite, f"finite numeric values: {relative} (bad={bad})", checks)
        frames[relative] = frame
        summary[relative] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    calibration = pd.read_csv(root / "result/table/09_calibration_audit.csv")
    max_calibration = float(calibration["calibration_error"].abs().max())
    _check(max_calibration <= 1.0e-10, f"calibration error <=1e-10 (actual={max_calibration:.3e})", checks)

    frontier = pd.read_csv(root / "result/table/07_frontier_comparison.csv")
    interior = frontier.loc[~frontier["ceiling_hit"].astype(bool)]
    ceiling = frontier.loc[frontier["ceiling_hit"].astype(bool)]
    max_residual = float(interior["q_star_residual"].abs().max()) if len(interior) else 0.0
    _check(max_residual <= 2.0e-10, f"q-star residual <=2e-10 (actual={max_residual:.3e})", checks)
    max_ceiling_q_error = (
        float((ceiling["q_star"] - ceiling["q_feasible_upper"]).abs().max())
        if len(ceiling)
        else 0.0
    )
    min_ceiling_budget_slack = (
        float((ceiling["J_bits"] - ceiling["ceiling_information_bits"]).min())
        if len(ceiling)
        else 0.0
    )
    _check(
        max_ceiling_q_error <= 2.0e-12,
        f"ceiling-hit q-star equals feasible upper bound (error={max_ceiling_q_error:.3e})",
        checks,
    )
    _check(
        min_ceiling_budget_slack >= -2.0e-10,
        f"ceiling-hit information budget reaches ceiling (slack={min_ceiling_budget_slack:.3e})",
        checks,
    )

    stress = pd.read_csv(root / "result/table/06_accumulation_stress_summary.csv")
    stress_limits = {
        "max_monotonicity_violation": 2.0e-12,
        "max_concavity_violation": 2.0e-12,
        "max_entropy_violation": 2.0e-12,
        "max_transcript_identity_error": 2.0e-10,
    }
    stress_actual: dict[str, float] = {}
    for column, tolerance in stress_limits.items():
        actual = float(stress[column].abs().max())
        stress_actual[column] = actual
        _check(actual <= tolerance, f"{column} <= {tolerance:.1e} (actual={actual:.3e})", checks)

    profiles = frames["result/table/06_accumulation_profiles.csv"]
    max_profile_monotonicity = 0.0
    max_profile_concavity = 0.0
    max_profile_entropy = 0.0
    for _, group in profiles.groupby(["curve_type", "pi", "beta", "dimension"], sort=False):
        ordered = group.sort_values("n")
        phi = ordered["phi_bits"].to_numpy(dtype=float)
        ceiling = ordered["entropy_ceiling_bits"].to_numpy(dtype=float)
        margins = np.diff(phi)
        if margins.size:
            max_profile_monotonicity = max(max_profile_monotonicity, float(np.max(-margins)))
        if margins.size > 1:
            max_profile_concavity = max(max_profile_concavity, float(np.max(np.diff(margins))))
        max_profile_entropy = max(max_profile_entropy, float(np.max(phi - ceiling)))
    _check(
        max_profile_monotonicity <= 2.0e-12,
        f"accumulation profiles monotone (violation={max_profile_monotonicity:.3e})",
        checks,
    )
    _check(
        max_profile_concavity <= 2.0e-12,
        f"accumulation profiles concave (violation={max_profile_concavity:.3e})",
        checks,
    )
    _check(
        max_profile_entropy <= 2.0e-12,
        f"accumulation profiles obey entropy ceilings (violation={max_profile_entropy:.3e})",
        checks,
    )

    finite_summary = pd.read_csv(root / "result/table/08_finite_K_summary.csv")
    k10000 = finite_summary.loc[finite_summary["K"] == 10000]
    _check(not k10000.empty, "finite-K summary includes K=10000", checks)
    metrics = {
        key: float(k10000.iloc[0][key])
        for key in ("median_relative_boost_error", "p90_relative_boost_error", "max_relative_boost_error")
    }
    metrics["median_loglog_slope"] = float(finite_summary["median_loglog_slope"].iloc[-1])

    if require_paper_profile:
        expected_commit = git_commit(root)
        for relative, frame in frames.items():
            commits = sorted(set(frame["git_commit"].astype(str)))
            _check(
                commits == [expected_commit],
                f"table commit matches formal HEAD: {relative} ({commits} == {[expected_commit]})",
                checks,
            )

        finite_length = frames["result/table/05_finite_length_simulation_validation.csv"]
        _check(len(finite_length) == 52, "paper Fig. 6 table has 4 AUC x 13 budget rows", checks)
        _check(
            set(np.round(finite_length["target_auc"], 12)) == {0.55, 0.70, 0.79, 0.90},
            "paper Fig. 6 target-AUC grid is complete",
            checks,
        )
        _check(
            set(finite_length["B"].astype(int))
            == {10, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 1600, 2000},
            "paper Fig. 6 budget grid is complete",
            checks,
        )
        _check(
            set(finite_length["n_trials"].astype(int)) == {6000},
            "paper Fig. 6 uses 6000 Monte Carlo repetitions",
            checks,
        )
        _check(
            float((finite_length["realized_auc"] - finite_length["target_auc"]).abs().max()) <= 1.0e-10,
            "paper Fig. 6 realized AUC matches target",
            checks,
        )

        _check(int(profiles["n"].max()) == 2048, "paper accumulation profiles reach n=2048", checks)
        _check(
            set(profiles.loc[profiles["curve_type"] == "component_target", "dimension"].astype(int))
            == {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048},
            "paper component-target dimension grid is complete",
            checks,
        )
        _check(int(stress["random_channels"].iloc[0]) == 50000, "50,000 channel stress cases ran", checks)
        _check(int(stress["bruteforce_instances"].iloc[0]) == 2000, "2,000 transcript audits ran", checks)
        _check(int(stress["channel_failure_count"].iloc[0]) == 0, "random-channel stress has no failures", checks)

        same_j = frames["result/table/07_same_J_tail_comparison.csv"]
        _check(len(frontier) == 256, "paper sharp-frontier grid has 256 cells", checks)
        _check(len(same_j) == 56, "equal-J table has 7 families x 8 tail fractions", checks)
        _check(
            float(same_j["J_mismatch_bits"].abs().max()) <= 1.0e-10,
            "equal-J channels meet their information target",
            checks,
        )

        convergence = frames["result/table/08_finite_K_convergence.csv"]
        expected_k = {200, 500, 1000, 3000, 10000, 30000, 100000, 300000, 1000000}
        _check(len(convergence) == 810, "finite-K paper sweep has 90 configurations per 9 K values", checks)
        _check(set(convergence["K"].astype(int)) == expected_k, "finite-K paper K grid is complete", checks)
        _check(
            set(convergence.groupby("K").size().astype(int)) == {90},
            "finite-K sweep has 90 configurations at every K",
            checks,
        )
        _check(set(finite_summary["K"].astype(int)) == expected_k, "finite-K summary K grid is complete", checks)
        _check(
            set(finite_summary["n_configurations"].astype(int)) == {90},
            "finite-K summaries cover all 90 configurations",
            checks,
        )
        _check(
            set(convergence["quadrature_order"].astype(int)) == {40},
            "paper finite-K sweep uses configured quadrature order 40",
            checks,
        )
        _check(
            set(finite_summary["mc_audit_reps"].astype(int)) == {12000},
            "paper finite-K Monte Carlo audit uses 12,000 repetitions",
            checks,
        )
        _check(
            finite_summary["mc_audit_within_3se"].astype(bool).all(),
            "finite-K deterministic benchmark agrees with Monte Carlo within 3 SE",
            checks,
        )
        _check(metrics["median_relative_boost_error"] <= 0.01, "finite-K K=10000 median error <=1%", checks)
        _check(metrics["p90_relative_boost_error"] <= 0.03, "finite-K K=10000 p90 error <=3%", checks)
        _check(metrics["max_relative_boost_error"] <= 0.05, "finite-K K=10000 maximum error <=5%", checks)
        _check(
            -1.1 <= metrics["median_loglog_slope"] <= -0.8,
            "finite-K median log-log slope is consistent with first-order convergence",
            checks,
        )

        validity = frames["result/table/09_weak_screening_validity.csv"]
        audit = frames["result/table/09_calibration_audit.csv"]
        mapping = frames["result/table/09_auc_epsilon_J_mapping.csv"]
        _check(len(validity) == 1008, "weak-screening paper grid has 4 x 14 x 18 rows", checks)
        _check(len(audit) == 60 and len(mapping) == 60, "weak-screening calibration tables have 60 rows", checks)
        _check(validity["alpha"].nunique() == 18, "weak-screening alpha grid has 18 points", checks)
        _check(validity["J_fraction_h2"].nunique() == 14, "weak-screening J grid has 14 points", checks)

    return {
        "tables": summary,
        "max_calibration_error": max_calibration,
        "max_q_star_residual": max_residual,
        "max_q_star_ceiling_error": max_ceiling_q_error,
        "min_q_star_ceiling_budget_slack": min_ceiling_budget_slack,
        "stress": stress_actual,
        "accumulation_profiles": {
            "max_monotonicity_violation": max_profile_monotonicity,
            "max_concavity_violation": max_profile_concavity,
            "max_entropy_violation": max_profile_entropy,
        },
        "finite_K_metrics": metrics,
    }


def validate_figures(root: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = root / "paper_figures/manifest.yaml"
    _check(manifest_path.is_file(), "paper figure manifest exists", checks)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    figures = payload.get("figures", []) if isinstance(payload, dict) else []
    _check(len(figures) == 8, "manifest maps exactly eight paper figures", checks)
    records: list[dict[str, Any]] = []
    for item in figures:
        stem = str(item["output"])
        for extension in ("pdf", "png"):
            relative = f"result/figure/paper/{stem}.{extension}"
            path = root / relative
            _check(path.is_file() and path.stat().st_size > 0, f"non-empty paper figure: {relative}", checks)
            records.append(
                {
                    "paper_figure": item["paper_figure"],
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    atomic_write_json(root / "result/artifact/figure_manifest.json", {"figures": records})
    return records


def validate_execution_artifacts(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, Path] = {
        "executed_notebook": root / f"result/artifact/{NOTEBOOK_STEM}.ipynb",
        "notebook_html": root / f"result/artifact/{NOTEBOOK_STEM}.html",
        "pytest_xml": root / "result/artifact/pytest.xml",
        "legacy_regressions": root / "result/artifact/legacy_regressions.json",
    }
    for label, path in artifacts.items():
        _check(path.is_file() and path.stat().st_size > 0, f"non-empty {label}", checks)

    test_root = ET.parse(artifacts["pytest_xml"]).getroot()
    failures = int(test_root.attrib.get("failures", 0))
    errors = int(test_root.attrib.get("errors", 0))
    if test_root.tag == "testsuites":
        failures = sum(int(node.attrib.get("failures", 0)) for node in test_root.findall("testsuite"))
        errors = sum(int(node.attrib.get("errors", 0)) for node in test_root.findall("testsuite"))
    _check(failures == 0 and errors == 0, f"pytest XML clean (failures={failures}, errors={errors})", checks)

    state_path = root / "state/revision_pipeline_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _check(state.get("profile") == "paper", "formal state uses paper profile", checks)
    _check(state.get("status") == "experiments_completed", "all experiment stages completed", checks)

    current = git_commit(root)
    config, _ = load_profile("paper", root)
    expected_config_digest = config_digest(config)
    _check(
        state.get("config_sha256") == expected_config_digest,
        "formal state matches the checked-in paper profile",
        checks,
    )
    recorded_stages = state.get("stages", {})
    _check(
        set(recorded_stages) == {stage for stage, _, _ in STAGES},
        "formal state records every required experiment stage",
        checks,
    )
    for stage, _, expected_outputs in STAGES:
        stage_record = recorded_stages.get(stage, {})
        stage_status = stage_record.get("status")
        _check(
            stage_status in {"completed", "skipped_valid_checkpoint"},
            f"formal stage completed or resumed from a valid checkpoint: {stage}",
            checks,
        )
        if stage_status == "completed":
            _check(
                stage_record.get("git_commit") == current
                and stage_record.get("config_sha256") == expected_config_digest,
                f"formal stage provenance matches commit/config: {stage}",
                checks,
            )
        checkpoint = CHECKPOINT_DIR / f"paper.{stage}.json"
        _check(
            checkpoint_is_valid(checkpoint, expected_outputs, expected_config_digest, root),
            f"formal checkpoint output hashes remain valid: {stage}",
            checks,
        )

    run_commit_path = root / "environment/run_commit.txt"
    recorded = run_commit_path.read_text(encoding="utf-8").strip()
    _check(recorded == current, f"formal run commit matches HEAD ({recorded} == {current})", checks)
    final_recorded = (root / "environment/final_commit.txt").read_text(encoding="utf-8").strip()
    _check(final_recorded == current, "recorded final source commit matches HEAD", checks)

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    _check(
        branch == "major-revision-reproducibility",
        f"formal run uses required branch (actual={branch})",
        checks,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    _check(not dirty.strip(), f"formal source tree is clean ({dirty})", checks)

    regression = json.loads(artifacts["legacy_regressions"].read_text(encoding="utf-8"))
    _check(regression.get("status") == "passed", "legacy experiments 00-05 passed", checks)
    _check(
        regression.get("git_commit") == current,
        "legacy regression provenance matches formal HEAD",
        checks,
    )

    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=root, text=True, capture_output=True, check=False
    )
    _check(diff_check.returncode == 0, f"git diff --check ({diff_check.stdout}{diff_check.stderr})", checks)
    return {
        "state": state,
        "formal_run_commit": recorded,
        "final_commit": final_recorded,
        "branch": branch,
        "legacy_regressions": regression,
        "artifacts": {
            label: {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in artifacts.items()
        },
        "pytest_failures": failures,
        "pytest_errors": errors,
    }


def validate_reference_comparison(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate an optional external-reference comparison rather than trusting its presence."""
    path = root / "result/artifact/reference_comparison.csv"
    if not path.exists():
        return {"status": "not_performed_no_reference_input"}
    _check(path.is_file() and path.stat().st_size > 0, "reference comparison is non-empty", checks)
    frame = pd.read_csv(path)
    _check(not frame.empty, "reference comparison contains rows", checks)
    finite, bad = _finite_numeric_frame(frame)
    _check(finite, f"reference comparison numeric values are finite (bad={bad})", checks)
    required = {"quantity", "absolute_difference", "tolerance", "passed"}
    missing = sorted(required.difference(frame.columns))
    _check(not missing, f"reference comparison schema is complete (missing={missing})", checks)
    normalized = frame["passed"].astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "pass", "passed", "fail", "failed"}
    _check(normalized.isin(allowed).all(), "reference pass/fail values parse strictly", checks)
    parsed = normalized.isin({"true", "1", "pass", "passed"})
    _check(parsed.all(), "all reference comparisons passed", checks)
    _check(
        (
            frame["absolute_difference"].abs().to_numpy(dtype=float)
            <= frame["tolerance"].to_numpy(dtype=float)
        ).all(),
        "all reference differences are within their declared tolerances",
        checks,
    )
    return {"status": "present_and_validated", "rows": len(frame), "columns": list(frame.columns)}


def write_environment_manifest(root: Path) -> Path:
    target = root / "result/artifact/environment_manifest.txt"
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"git_commit={git_commit(root)}",
    ]
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines.append("\n[pip-freeze]\n" + freeze.rstrip())
    except (OSError, subprocess.CalledProcessError) as exc:
        lines.append(f"pip_freeze_error={exc}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def validate_all_outputs(root: Path, require_paper_profile: bool = True) -> dict[str, Any]:
    """Run every formal check and persist a machine-readable report."""
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {"status": "running", "checks": checks}
    output_path = root / "result/artifact/output_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["numerical"] = validate_table_set(root, checks, require_paper_profile=require_paper_profile)
        report["figures"] = validate_figures(root, checks)
        if require_paper_profile:
            report["execution"] = validate_execution_artifacts(root, checks)
        report["environment_manifest"] = str(write_environment_manifest(root).relative_to(root))
        report["reference_comparison"] = validate_reference_comparison(root, checks)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output_path, report)
        raise
    atomic_write_json(output_path, report)
    return report


__all__ = [
    "METADATA_COLUMNS",
    "NOTEBOOK_STEM",
    "required_tables",
    "validate_all_outputs",
    "validate_execution_artifacts",
    "validate_figures",
    "validate_reference_comparison",
    "validate_table_set",
    "write_environment_manifest",
]
