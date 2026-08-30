#!/usr/bin/env python3
"""Deterministic finite-K top-B convergence sweep for paper Figure 7."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.channels import make_continuous_channel_for_information
from acli.revision.finite_k import (
    finite_k_top_b_precision,
    large_k_top_tail_precision,
    monte_carlo_top_b_precision,
    relative_boost_error,
)
from acli.revision.information import h2
from acli.revision.reproducibility import canonical_configuration, stable_task_seed
from acli.revision.suite import config_digest, git_commit, load_profile


FAMILIES = ("gaussian", "pareto4", "student_t5")
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1)
SCREENING_LEVELS = (0.003, 0.01, 0.03)
PREVALENCES = (0.01, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_channel(family: str, p: float, j_fraction: float):
    target = j_fraction * float(h2(p))
    return make_continuous_channel_for_information(family, p, target), target


def realized_selection_fraction(K: int, requested_alpha: float) -> tuple[int, float]:
    """Map a nominal fraction to its integer budget and actual finite-pool fraction."""
    budget = max(1, min(K, int(round(requested_alpha * K))))
    return budget, budget / K


def evaluate_task(
    task: dict,
    channel,
    target_j: float,
    actual_j: float,
    calibration_error: float,
    q_limit: float,
    master_seed: int,
    commit: str,
) -> dict:
    K = int(task["K"])
    requested_alpha = float(task["alpha"])
    B, realized_alpha = realized_selection_fraction(K, requested_alpha)
    q_finite = finite_k_top_b_precision(
        channel,
        K,
        B,
        quadrature_order=int(task["quadrature_order"]),
    )
    error = relative_boost_error(q_finite, q_limit, task["p"])
    configuration = {
        **task,
        "B": B,
        "realized_alpha": realized_alpha,
        "q_alpha_fraction": realized_alpha,
        "quadrature_order": int(task["quadrature_order"]),
        "target_J_bits": target_j,
    }
    return {
        "sweep": "paper_full" if task["full_sweep"] else "paper_figure",
        "paper_figure_configuration": bool(
            abs(task["p"] - 0.01) < 1e-15 and abs(task["j_fraction"] - 0.03) < 1e-15
        ),
        "p": task["p"],
        "family": task["family"],
        "screening_level": task["j_fraction"],
        "J_fraction_h2": task["j_fraction"],
        "target_J_bits": target_j,
        "actual_J_bits": actual_j,
        "J_mismatch": actual_j - target_j,
        "epsilon": channel.epsilon,
        "intercept": channel.intercept,
        "calibration_error": calibration_error,
        "alpha": requested_alpha,
        "K": K,
        "B": B,
        "realized_alpha": realized_alpha,
        "q_alpha_fraction": realized_alpha,
        "quadrature_order": int(task["quadrature_order"]),
        "q_K_B": q_finite,
        "q_alpha": q_limit,
        "relative_boost_error": error,
        "seed": stable_task_seed(master_seed, "08_finite_K", "deterministic_precision", configuration),
        "configuration": canonical_configuration(configuration),
        "git_commit": commit,
        "method": "deterministic_order_statistic_quadrature",
    }


def main() -> None:
    args = parse_args()
    profile, _ = load_profile(args.profile)
    master_seed = int(profile["master_seed"])
    K_values = tuple(int(value) for value in profile["finite_k_values"])
    full_sweep = bool(profile["finite_k_full_sweep"])
    profile_sha256 = config_digest(profile)
    quadrature_order = int(profile["quadrature_order"])
    finite_k_mc_reps = int(profile["finite_k_mc_reps"])
    prevalences = PREVALENCES if full_sweep else (0.01,)
    levels = SCREENING_LEVELS if full_sweep else (0.03,)
    tasks = [
        {
            "p": p,
            "family": family,
            "j_fraction": level,
            "alpha": alpha,
            "K": K,
            "full_sweep": full_sweep,
            "profile_sha256": profile_sha256,
            "quadrature_order": quadrature_order,
        }
        for K in K_values
        for p in prevalences
        for family in FAMILIES
        for level in levels
        for alpha in ALPHAS
    ]
    workers = max(1, min(int(args.workers), int(profile.get("large_k_workers", 8)), 8))
    prepared = {}
    for p in prevalences:
        for family in FAMILIES:
            for level in levels:
                channel, target_j = build_channel(family, p, level)
                prepared[(family, p, level)] = (
                    channel,
                    target_j,
                    channel.mutual_information_bits(),
                    channel.calibration_error,
                )
    realized_fractions = {
        realized_selection_fraction(int(task["K"]), float(task["alpha"]))[1]
        for task in tasks
    }
    tail_limits = {
        (family, p, level, realized_alpha): large_k_top_tail_precision(
            prepared[(family, p, level)][0], realized_alpha
        )
        for p in prevalences
        for family in FAMILIES
        for level in levels
        for realized_alpha in realized_fractions
    }
    commit = git_commit(ROOT)
    print(f"[08] evaluating {len(tasks)} deterministic configurations with {workers} workers", flush=True)
    rows = Parallel(n_jobs=workers, prefer="processes", verbose=5)(
        delayed(evaluate_task)(
            task,
            *prepared[(task["family"], task["p"], task["j_fraction"])],
            tail_limits[
                (
                    task["family"],
                    task["p"],
                    task["j_fraction"],
                    realized_selection_fraction(int(task["K"]), float(task["alpha"]))[1],
                )
            ],
            master_seed,
            commit,
        )
        for task in tasks
    )
    frame = pd.DataFrame(rows).sort_values(
        ["K", "p", "family", "screening_level", "alpha"], ignore_index=True
    )

    slopes: list[float] = []
    group_columns = ["p", "family", "screening_level", "alpha"]
    for _, group in frame.groupby(group_columns, sort=False):
        positive = group.loc[group["relative_boost_error"] > 0].sort_values("K")
        if len(positive) >= 2:
            slope = float(
                np.polyfit(
                    np.log(positive["K"].to_numpy(dtype=float)),
                    np.log(positive["relative_boost_error"].to_numpy(dtype=float)),
                    1,
                )[0]
            )
            slopes.append(slope)
    median_slope = float(np.median(slopes)) if slopes else 0.0

    audit_p = 0.01
    audit_family = "gaussian"
    audit_level = 0.03
    audit_alpha = 0.01
    audit_K = min(K_values, key=lambda value: abs(value - 1000))
    audit_B, audit_fraction = realized_selection_fraction(audit_K, audit_alpha)
    audit_channel = prepared[(audit_family, audit_p, audit_level)][0]
    audit_deterministic = float(
        frame.loc[
            (frame["p"] == audit_p)
            & (frame["family"] == audit_family)
            & (frame["screening_level"] == audit_level)
            & (frame["alpha"] == audit_alpha)
            & (frame["K"] == audit_K),
            "q_K_B",
        ].iloc[0]
    )
    audit_configuration = {
        "p": audit_p,
        "family": audit_family,
        "screening_level": audit_level,
        "requested_alpha": audit_alpha,
        "realized_alpha": audit_fraction,
        "K": audit_K,
        "B": audit_B,
        "n_trials": finite_k_mc_reps,
        "profile_sha256": profile_sha256,
    }
    audit_seed = stable_task_seed(
        master_seed, "08_finite_K", "monte_carlo_audit", audit_configuration
    )
    audit_estimate, audit_se = monte_carlo_top_b_precision(
        audit_channel,
        audit_K,
        audit_B,
        n_trials=finite_k_mc_reps,
        seed=audit_seed,
    )
    audit_abs_difference = abs(audit_estimate - audit_deterministic)
    audit_within_3se = audit_abs_difference <= 3.0 * audit_se + 2.0e-10
    print(
        f"[08] MC audit reps={finite_k_mc_reps} difference={audit_abs_difference:.3e} "
        f"SE={audit_se:.3e} within_3SE={audit_within_3se}",
        flush=True,
    )

    reference = {
        "median_relative_boost_error": 0.00125,
        "p90_relative_boost_error": 0.00994,
        "max_relative_boost_error": 0.0201,
        "median_loglog_slope": -0.995,
    }
    summary_rows: list[dict] = []
    for K, group in frame.groupby("K", sort=True):
        values = group["relative_boost_error"].to_numpy(dtype=float)
        metrics = {
            "median_relative_boost_error": float(np.median(values)),
            "p90_relative_boost_error": float(np.quantile(values, 0.90)),
            "max_relative_boost_error": float(np.max(values)),
        }
        configuration = {
            "profile": profile["profile"],
            "profile_sha256": profile_sha256,
            "K": int(K),
            "K_values": list(K_values),
            "prevalences": list(prevalences),
            "families": list(FAMILIES),
            "screening_levels": list(levels),
            "alphas": list(ALPHAS),
            "full_sweep": full_sweep,
            "budget_rule": "B=max(1,round(alpha*K))",
            "asymptotic_tail_fraction": "realized_alpha=B/K",
            "quadrature_order": quadrature_order,
            "finite_k_mc_reps": finite_k_mc_reps,
            "n_configurations": int(len(group)),
        }
        at_reference = full_sweep and int(K) == 10000 and len(group) == 90
        not_applicable: float | str = "not_applicable"
        summary_rows.append(
            {
                "K": int(K),
                "n_configurations": len(group),
                **metrics,
                "median_loglog_slope": median_slope,
                "mc_audit_configuration": canonical_configuration(audit_configuration),
                "mc_audit_seed": audit_seed,
                "mc_audit_reps": finite_k_mc_reps,
                "mc_audit_deterministic_precision": audit_deterministic,
                "mc_audit_estimate_precision": audit_estimate,
                "mc_audit_se": audit_se,
                "mc_audit_abs_difference": audit_abs_difference,
                "mc_audit_within_3se": audit_within_3se,
                "reference_available": at_reference,
                "reference_scope": "paper_full_K10000_90_configurations",
                "reference_median_relative_boost_error": (
                    reference["median_relative_boost_error"] if at_reference else not_applicable
                ),
                "reference_p90_relative_boost_error": (
                    reference["p90_relative_boost_error"] if at_reference else not_applicable
                ),
                "reference_max_relative_boost_error": (
                    reference["max_relative_boost_error"] if at_reference else not_applicable
                ),
                "reference_median_loglog_slope": (
                    reference["median_loglog_slope"] if at_reference else not_applicable
                ),
                "median_difference_from_reference": (
                    metrics["median_relative_boost_error"]
                    - reference["median_relative_boost_error"]
                    if at_reference
                    else not_applicable
                ),
                "p90_difference_from_reference": (
                    metrics["p90_relative_boost_error"]
                    - reference["p90_relative_boost_error"]
                    if at_reference
                    else not_applicable
                ),
                "max_difference_from_reference": (
                    metrics["max_relative_boost_error"]
                    - reference["max_relative_boost_error"]
                    if at_reference
                    else not_applicable
                ),
                "slope_difference_from_reference": (
                    median_slope - reference["median_loglog_slope"]
                    if at_reference
                    else not_applicable
                ),
                "seed": stable_task_seed(master_seed, "08_finite_K", "summary", configuration),
                "configuration": canonical_configuration(configuration),
                "git_commit": commit,
                "method": "deterministic_summary_with_Monte_Carlo_audit",
            }
        )

    output = ROOT / "result/table"
    atomic_csv(frame, output / "08_finite_K_convergence.csv")
    atomic_csv(pd.DataFrame(summary_rows), output / "08_finite_K_summary.csv")
    print(f"[08] median log-log slope={median_slope:.6f}", flush=True)


if __name__ == "__main__":
    main()
