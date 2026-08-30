#!/usr/bin/env python3
"""Audit calibrated weak-screening approximations across tail regimes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.channels import (
    CONTINUOUS_FAMILIES,
    calibrate_epsilon_for_auc,
    calibrate_epsilon_for_information,
    make_continuous_channel,
)
from acli.revision.information import h2
from acli.revision.reproducibility import canonical_configuration, stable_task_seed
from acli.revision.suite import git_commit, load_profile


FAMILIES = ("gaussian", "pareto4", "student_t5", "uniform")
FIG6_GAUSSIAN_AUC_TARGETS = (0.55, 0.70, 0.79, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", help="smoke, paper, or a YAML path")
    parser.add_argument("--workers", type=int, default=1, help="maximum process workers")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Validate and atomically replace one CSV in its destination directory."""

    if frame.empty:
        raise ValueError(f"refusing to write empty table: {path.name}")
    for column in frame.select_dtypes(include=[np.number]).columns:
        if not np.all(np.isfinite(frame[column].to_numpy(dtype=np.float64))):
            raise ValueError(f"non-finite values in {path.name}:{column}")
    if frame.isna().any(axis=None):
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"missing values in {path.name}: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _software_versions() -> str:
    return canonical_configuration(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        }
    )


def _provenance(
    master_seed: int,
    task: str,
    configuration: dict[str, Any],
    method: str,
    commit: str,
    software: str,
) -> dict[str, Any]:
    return {
        "seed": stable_task_seed(master_seed, "09_weak_screening", task, configuration),
        "configuration": canonical_configuration(configuration),
        "git_commit": commit,
        "software": software,
        "method": method,
    }


def _information_task(
    task: tuple[str, float, float, tuple[float, ...]]
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    family, p, j_fraction, alpha_grid = task
    target_j = float(j_fraction * h2(p))
    epsilon = calibrate_epsilon_for_information(family, p, target_j)
    channel = make_continuous_channel(family, p, epsilon)
    actual_j = channel.mutual_information_bits()
    actual_auc = channel.auc()
    weak_epsilon_j = epsilon * epsilon * p * (1.0 - p) / (2.0 * np.log(2.0))
    rows: list[dict[str, float | str]] = []
    for alpha in alpha_grid:
        g_alpha = float(channel.base_quantile(1.0 - alpha))
        tail_mean = channel.base_top_tail_mean(alpha)
        exact_q = channel.top_tail_precision(alpha)
        weak_epsilon_q = p + epsilon * p * (1.0 - p) * tail_mean
        weak_j_q = p + np.sqrt(2.0 * np.log(2.0) * p * (1.0 - p) * actual_j) * tail_mean
        exact_boost = exact_q - p
        if exact_boost <= 0.0:
            raise ArithmeticError("positive screening strength produced no top-tail boost")
        rows.append(
            {
                "family": family,
                "p": p,
                "alpha": float(alpha),
                "J_fraction_h2": j_fraction,
                "target_J_bits": target_j,
                "actual_J_bits": actual_j,
                "J_mismatch_bits": actual_j - target_j,
                "epsilon": epsilon,
                "intercept": channel.intercept,
                "realized_auc": actual_auc,
                "g_alpha": g_alpha,
                "base_tail_mean": tail_mean,
                "tail_locality": epsilon * abs(g_alpha),
                "exact_q_alpha": exact_q,
                "weak_epsilon_q": weak_epsilon_q,
                "weak_J_q": weak_j_q,
                "relative_boost_error_epsilon": abs(weak_epsilon_q - exact_q) / exact_boost,
                "relative_boost_error_J": abs(weak_j_q - exact_q) / exact_boost,
                "weak_epsilon_J_bits": weak_epsilon_j,
                "calibration_error": channel.calibration_error,
            }
        )
    summary: dict[str, float | str] = {
        "source": "J_grid",
        "family": family,
        "p": p,
        "target_metric": "J_fraction_h2",
        "target_value": j_fraction,
        "target_J_bits": target_j,
        "actual_J_bits": actual_j,
        "J_fraction_h2": actual_j / float(h2(p)),
        "target_auc": actual_auc,
        "realized_auc": actual_auc,
        "epsilon": epsilon,
        "intercept": channel.intercept,
        "mean_eta": channel.mean_eta,
        "calibration_error": channel.calibration_error,
    }
    return rows, summary


def _auc_task(target_auc: float, p: float = 0.01) -> dict[str, float | str]:
    epsilon = calibrate_epsilon_for_auc("gaussian", p, target_auc)
    channel = make_continuous_channel("gaussian", p, epsilon)
    actual_j = channel.mutual_information_bits()
    return {
        "source": "fig6_auc_target",
        "family": "gaussian",
        "p": p,
        "target_metric": "auc",
        "target_value": float(target_auc),
        "target_J_bits": actual_j,
        "actual_J_bits": actual_j,
        "J_fraction_h2": actual_j / float(h2(p)),
        "target_auc": float(target_auc),
        "realized_auc": channel.auc(),
        "epsilon": epsilon,
        "intercept": channel.intercept,
        "mean_eta": channel.mean_eta,
        "calibration_error": channel.calibration_error,
    }


def _parallel_map(function: Any, tasks: Iterable[Any], workers: int) -> list[Any]:
    task_list = list(tasks)
    if workers <= 1:
        return [function(task) for task in task_list]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, task_list))


def _attach_validity_provenance(
    rows: list[dict[str, Any]], master_seed: int, commit: str, software: str
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        configuration = {
            "family": row["family"],
            "p": row["p"],
            "alpha": row["alpha"],
            "J_fraction_h2": row["J_fraction_h2"],
            "target_J_bits": row["target_J_bits"],
        }
        enriched.append(
            {
                **row,
                **_provenance(
                    master_seed,
                    "validity_cell",
                    configuration,
                    "exact_calibrated_quadrature_with_first_order_approximations",
                    commit,
                    software,
                ),
            }
        )
    return enriched


def _attach_summary_provenance(
    rows: list[dict[str, Any]], master_seed: int, commit: str, software: str
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        configuration = {
            "source": row["source"],
            "family": row["family"],
            "p": row["p"],
            "target_metric": row["target_metric"],
            "target_value": row["target_value"],
        }
        enriched.append(
            {
                **row,
                **_provenance(
                    master_seed,
                    "calibration_audit",
                    configuration,
                    "exact_intercept_root_and_deterministic_quadrature",
                    commit,
                    software,
                ),
            }
        )
    return enriched


def _mapping_rows(
    summaries: list[dict[str, Any]], master_seed: int, commit: str, software: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        configuration = {
            "mapping_kind": summary["source"],
            "family": summary["family"],
            "p": summary["p"],
            "target_metric": summary["target_metric"],
            "target_value": summary["target_value"],
        }
        rows.append(
            {
                "mapping_kind": summary["source"],
                "family": summary["family"],
                "p": summary["p"],
                "target_auc": summary["target_auc"],
                "realized_auc": summary["realized_auc"],
                "epsilon": summary["epsilon"],
                "intercept": summary["intercept"],
                "J_bits": summary["actual_J_bits"],
                "J_fraction_h2": summary["J_fraction_h2"],
                "mean_eta": summary["mean_eta"],
                "calibration_error": summary["calibration_error"],
                **_provenance(
                    master_seed,
                    "auc_epsilon_J_mapping",
                    configuration,
                    "exact_calibrated_root_and_deterministic_quadrature",
                    commit,
                    software,
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config, _ = load_profile(args.profile)
    profile_name = str(config.get("profile", args.profile))
    alpha_count = int(config["weak_validity_alpha_points"])
    j_count = int(config["weak_validity_j_points"])
    if alpha_count < 2 or j_count < 2:
        raise ValueError("weak-validity grids require at least two points")
    if any(family not in CONTINUOUS_FAMILIES for family in FAMILIES):
        raise AssertionError("experiment family list and channel registry disagree")

    p = 0.01
    alpha_grid = tuple(float(value) for value in np.geomspace(1e-4, 0.3, alpha_count))
    j_fraction_grid = tuple(float(value) for value in np.geomspace(1e-4, 0.3, j_count))
    requested_workers = max(1, int(args.workers))
    profile_workers = max(1, int(config.get("workers", requested_workers)))
    workers = min(requested_workers, profile_workers, len(FAMILIES) * j_count)
    master_seed = int(config["master_seed"])
    commit = git_commit(ROOT)
    software = _software_versions()
    print(
        f"[09] profile={profile_name} workers={workers} "
        f"families={len(FAMILIES)} J={j_count} alpha={alpha_count}",
        flush=True,
    )

    information_tasks = [
        (family, p, j_fraction, alpha_grid)
        for family in FAMILIES
        for j_fraction in j_fraction_grid
    ]
    evaluated = _parallel_map(_information_task, information_tasks, workers)
    raw_validity = [row for result_rows, _ in evaluated for row in result_rows]
    j_summaries = [summary for _, summary in evaluated]

    auc_workers = min(workers, len(FIG6_GAUSSIAN_AUC_TARGETS))
    auc_summaries = _parallel_map(_auc_task, FIG6_GAUSSIAN_AUC_TARGETS, auc_workers)
    all_summaries = [*j_summaries, *auc_summaries]

    validity_rows = _attach_validity_provenance(
        raw_validity, master_seed, commit, software
    )
    audit_rows = _attach_summary_provenance(
        all_summaries, master_seed, commit, software
    )
    mapping_rows = _mapping_rows(all_summaries, master_seed, commit, software)

    outputs = ROOT / "result/table"
    atomic_csv(
        pd.DataFrame(validity_rows).sort_values(
            ["family", "J_fraction_h2", "alpha"], kind="stable"
        ),
        outputs / "09_weak_screening_validity.csv",
    )
    atomic_csv(
        pd.DataFrame(audit_rows).sort_values(
            ["source", "family", "target_value"], kind="stable"
        ),
        outputs / "09_calibration_audit.csv",
    )
    atomic_csv(
        pd.DataFrame(mapping_rows).sort_values(
            ["mapping_kind", "family", "target_auc"], kind="stable"
        ),
        outputs / "09_auc_epsilon_J_mapping.csv",
    )
    max_calibration_error = max(float(row["calibration_error"]) for row in audit_rows)
    print(
        f"[09] wrote {len(validity_rows)} validity rows, {len(audit_rows)} audit rows; "
        f"max calibration error={max_calibration_error:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
