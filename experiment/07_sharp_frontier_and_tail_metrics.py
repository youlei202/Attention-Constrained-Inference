#!/usr/bin/env python3
"""Sharp screening frontiers and equal-information upper-tail comparisons."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.channels import (  # noqa: E402
    CONTINUOUS_FAMILIES,
    make_continuous_channel_for_information,
    make_score_channel,
)
from acli.revision.frontier import (  # noqa: E402
    binary_selection_information,
    feasible_q_bounds,
    q_one_branch,
    q_one_branch_raw,
    q_pinsker_clipped,
    q_pinsker_raw,
    q_star,
)
from acli.revision.information import h2  # noqa: E402
from acli.revision.reproducibility import (  # noqa: E402
    canonical_configuration,
    stable_task_seed,
)
from acli.revision.suite import git_commit, load_profile  # noqa: E402


PAPER_P_GRID = (0.001, 0.01, 0.05, 0.1)
PAPER_ALPHA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
PAPER_J_FRACTION_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)

SMOKE_P_GRID = (0.01, 0.1)
SMOKE_ALPHA_GRID = (1e-3, 1e-2, 1e-1, 3e-1)
SMOKE_J_FRACTION_GRID = (1e-3, 1e-2, 1e-1, 3e-1)

MATCHED_P = 0.01
MATCHED_J_FRACTION = 0.01
MATCHED_FAMILIES = (
    "gaussian",
    "student_t5",
    "pareto4",
    "pareto5",
    "rare_spike",
    "three_point",
    "uniform",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Validate and atomically replace one result CSV."""

    if frame.empty:
        raise ValueError(f"refusing to write an empty result table: {path.name}")
    missing_metadata = {"seed", "configuration", "git_commit", "method"} - set(
        frame.columns
    )
    if missing_metadata:
        raise ValueError(f"{path.name} lacks metadata columns: {sorted(missing_metadata)}")
    if frame.isna().any(axis=None):
        bad = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{path.name} contains missing values in {bad}")
    for column in frame.select_dtypes(include=[np.number]).columns:
        if not np.all(np.isfinite(frame[column].to_numpy(dtype=np.float64))):
            raise ValueError(f"{path.name} contains non-finite values in {column}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def provenance(
    master_seed: int,
    task: str,
    configuration: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    return {
        "seed": stable_task_seed(master_seed, "07_frontier_tail", task, configuration),
        "configuration": canonical_configuration(configuration),
        "git_commit": git_commit(ROOT),
        "method": method,
    }


def small_j_coefficients(p: float, alpha: float) -> dict[str, float]:
    """Coefficients in ``q-p = c sqrt(J/alpha) + o(sqrt(J))``."""

    sharp = float(np.sqrt(2.0 * np.log(2.0) * p * (1.0 - p) * (1.0 - alpha)))
    one_branch = float(np.sqrt(np.log(2.0) / 2.0))
    two_branch = float(np.sqrt(np.log(2.0) * (1.0 - alpha) / 2.0))
    return {
        "small_j_sharp_coefficient": sharp,
        "small_j_one_branch_pinsker_coefficient": one_branch,
        "small_j_two_branch_pinsker_coefficient": two_branch,
        "small_j_one_branch_to_sharp_ratio": one_branch / sharp,
        "small_j_two_branch_to_sharp_ratio": two_branch / sharp,
    }


def selected_frontier_grid(config: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    if bool(config.get("frontier_full_grid", False)):
        return PAPER_P_GRID, PAPER_ALPHA_GRID, PAPER_J_FRACTION_GRID
    return SMOKE_P_GRID, SMOKE_ALPHA_GRID, SMOKE_J_FRACTION_GRID


def frontier_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    master_seed = int(config["master_seed"])
    p_grid, alpha_grid, j_fraction_grid = selected_frontier_grid(config)
    rows: list[dict[str, Any]] = []
    for p in p_grid:
        entropy = float(h2(p))
        for alpha in alpha_grid:
            lower, upper = feasible_q_bounds(p, alpha)
            ceiling_information = binary_selection_information(upper, p, alpha)
            coefficients = small_j_coefficients(p, alpha)
            for j_fraction in j_fraction_grid:
                information = float(j_fraction * entropy)
                sharp = q_star(p, alpha, information)
                sharp_information = binary_selection_information(sharp, p, alpha)
                ceiling_hit = information >= ceiling_information - 2e-14
                one_raw = q_one_branch_raw(p, alpha, information)
                one_clipped = q_one_branch(p, alpha, information)
                two_raw = q_pinsker_raw(p, alpha, information)
                two_clipped = q_pinsker_clipped(p, alpha, information)
                normalization = float(np.sqrt(information / alpha))
                task_configuration = {
                    "p": p,
                    "alpha": alpha,
                    "J_fraction": j_fraction,
                    "frontier_full_grid": bool(config.get("frontier_full_grid", False)),
                }
                rows.append(
                    {
                        "p": p,
                        "alpha": alpha,
                        "H2_p_bits": entropy,
                        "J_fraction": j_fraction,
                        "J_bits": information,
                        "q_feasible_lower": lower,
                        "q_feasible_upper": upper,
                        "q_star": sharp,
                        "q_star_information_bits": sharp_information,
                        "q_star_residual": sharp_information - information,
                        "ceiling_information_bits": ceiling_information,
                        "ceiling_hit": bool(ceiling_hit),
                        "q_one_branch_raw": one_raw,
                        "q_one_branch_clipped": one_clipped,
                        "q_two_branch_pinsker_raw": two_raw,
                        "q_two_branch_pinsker_clipped": two_clipped,
                        "q_star_boost": sharp - p,
                        "one_branch_raw_gap": one_raw - sharp,
                        "one_branch_clipped_gap": one_clipped - sharp,
                        "two_branch_raw_gap": two_raw - sharp,
                        "two_branch_clipped_gap": two_clipped - sharp,
                        "finite_j_sharp_coefficient": (sharp - p) / normalization,
                        "finite_j_one_branch_coefficient": (one_raw - p) / normalization,
                        "finite_j_two_branch_coefficient": (two_raw - p) / normalization,
                        "finite_to_small_j_sharp_ratio": (
                            (sharp - p) / normalization
                        )
                        / coefficients["small_j_sharp_coefficient"],
                        **coefficients,
                        **provenance(
                            master_seed,
                            "frontier_grid_point",
                            task_configuration,
                            "exact_binary_KL_inversion_and_analytic_Pinsker_bounds",
                        ),
                    }
                )
    return rows


def _discrete_channel_for_information(family: str, p: float, target_j: float):
    """Calibrate discrete-channel epsilon to the requested actual MI."""

    def residual(epsilon: float) -> float:
        return float(make_score_channel(family, p, epsilon).J_bits - target_j)

    upper = 0.25
    for _ in range(40):
        if residual(upper) >= 0.0:
            break
        upper *= 2.0
    else:
        raise ValueError(f"{family} cannot reach target J={target_j:.12g} bits")
    epsilon = float(
        brentq(
            residual,
            0.0,
            upper,
            xtol=5e-14,
            rtol=4.0 * np.finfo(float).eps,
        )
    )
    return make_score_channel(family, p, epsilon)


def matched_channel(family: str, p: float, target_j: float):
    if family in CONTINUOUS_FAMILIES:
        return make_continuous_channel_for_information(family, p, target_j)
    return _discrete_channel_for_information(family, p, target_j)


def same_j_tail_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    master_seed = int(config["master_seed"])
    target_j = MATCHED_J_FRACTION * float(h2(MATCHED_P))
    rows: list[dict[str, Any]] = []
    for family in MATCHED_FAMILIES:
        channel = matched_channel(family, MATCHED_P, target_j)
        actual_j = float(channel.J_bits)
        auc = float(channel.auc())
        epsilon = float(channel.epsilon)
        intercept = float(channel.intercept)
        for alpha in PAPER_ALPHA_GRID:
            q_alpha = float(channel.top_tail_precision(alpha))
            tail_information = binary_selection_information(q_alpha, MATCHED_P, alpha)
            if tail_information > actual_j + 2e-10:
                raise ArithmeticError(
                    f"tail information exceeded total channel information for {family}"
                )
            # Data processing gives J_tail <= J.  At an exactly matched
            # discrete cutoff the two independently evaluated expressions can
            # differ by a few ulps, so enforce the analytic boundary here.
            tail_information = float(np.clip(tail_information, 0.0, actual_j))
            sharp = q_star(MATCHED_P, alpha, actual_j)
            one_raw = q_one_branch_raw(MATCHED_P, alpha, actual_j)
            one_clipped = q_one_branch(MATCHED_P, alpha, actual_j)
            two_raw = q_pinsker_raw(MATCHED_P, alpha, actual_j)
            two_clipped = q_pinsker_clipped(MATCHED_P, alpha, actual_j)
            coefficients = small_j_coefficients(MATCHED_P, alpha)
            normalization = float(np.sqrt(actual_j / alpha))
            channel_coefficient = (q_alpha - MATCHED_P) / normalization
            sharp_coefficient = (sharp - MATCHED_P) / normalization
            boost_fraction = (q_alpha - MATCHED_P) / (sharp - MATCHED_P)
            task_configuration = {
                "family": family,
                "p": MATCHED_P,
                "target_J_fraction": MATCHED_J_FRACTION,
                "target_J_bits": target_j,
                "alpha": alpha,
                "epsilon": epsilon,
            }
            rows.append(
                {
                    "family": family,
                    "p": MATCHED_P,
                    "alpha": alpha,
                    "H2_p_bits": float(h2(MATCHED_P)),
                    "target_J_fraction": MATCHED_J_FRACTION,
                    "target_J_bits": target_j,
                    "actual_J_bits": actual_j,
                    "J_mismatch_bits": actual_j - target_j,
                    "J_mismatch": actual_j - target_j,
                    "J_abs_mismatch_bits": abs(actual_j - target_j),
                    "epsilon": epsilon,
                    "intercept": intercept,
                    "calibration_error": float(channel.calibration_error),
                    "AUC": auc,
                    "auc": auc,
                    "q_alpha": q_alpha,
                    "J_tail_bits": tail_information,
                    "J_tail_over_J": tail_information / actual_j,
                    "q_star": sharp,
                    "q_one_branch_raw": one_raw,
                    "q_one_branch_clipped": one_clipped,
                    "one_branch_pinsker": one_clipped,
                    "q_two_branch_pinsker_raw": two_raw,
                    "q_two_branch_pinsker_clipped": two_clipped,
                    "two_branch_pinsker": two_clipped,
                    "q_alpha_over_q_star": q_alpha / sharp,
                    "tail_boost_fraction_of_sharp": boost_fraction,
                    "finite_j_channel_coefficient": channel_coefficient,
                    "small_J_coefficient": channel_coefficient,
                    "finite_j_sharp_coefficient": sharp_coefficient,
                    "finite_j_upper_lower_coefficient_ratio": (
                        sharp_coefficient / channel_coefficient
                    ),
                    "small_j_sharp_to_channel_coefficient_ratio": (
                        coefficients["small_j_sharp_coefficient"]
                        / channel_coefficient
                    ),
                    # Compatibility names used by the figure notebook.  The
                    # lower coefficient is the tight local binary-KL frontier;
                    # the upper coefficient is its two-branch Pinsker envelope.
                    "coefficient_lower": coefficients["small_j_sharp_coefficient"],
                    "coefficient_upper": coefficients[
                        "small_j_two_branch_pinsker_coefficient"
                    ],
                    "coefficient_ratio_to_lower": (
                        channel_coefficient
                        / coefficients["small_j_sharp_coefficient"]
                    ),
                    "coefficient_ratio_to_upper": (
                        channel_coefficient
                        / coefficients["small_j_two_branch_pinsker_coefficient"]
                    ),
                    **coefficients,
                    **provenance(
                        master_seed,
                        "same_J_tail_point",
                        task_configuration,
                        "exact_calibrated_channel_quadrature_or_discrete_sum",
                    ),
                }
            )
    return rows


def constant_gap_rows(
    config: dict[str, Any],
    frontier: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract finite-J and local coefficient gaps from every frontier point."""

    master_seed = int(config["master_seed"])
    rows: list[dict[str, Any]] = []
    for source in frontier:
        p = float(source["p"])
        alpha = float(source["alpha"])
        j_fraction = float(source["J_fraction"])
        sharp_boost = float(source["q_star_boost"])
        one_boost = float(source["q_one_branch_clipped"] - p)
        two_boost = float(source["q_two_branch_pinsker_clipped"] - p)
        task_configuration = {
            "p": p,
            "alpha": alpha,
            "J_fraction": j_fraction,
        }
        rows.append(
            {
                "p": p,
                "alpha": alpha,
                "J_fraction": j_fraction,
                "J_bits": float(source["J_bits"]),
                "ceiling_hit": bool(source["ceiling_hit"]),
                "q_star": float(source["q_star"]),
                "q_one_branch_clipped": float(source["q_one_branch_clipped"]),
                "q_two_branch_pinsker_clipped": float(
                    source["q_two_branch_pinsker_clipped"]
                ),
                "sharp_boost": sharp_boost,
                "one_branch_boost": one_boost,
                "two_branch_boost": two_boost,
                "one_branch_additive_gap": one_boost - sharp_boost,
                "two_branch_additive_gap": two_boost - sharp_boost,
                "one_branch_boost_ratio": one_boost / sharp_boost,
                "two_branch_boost_ratio": two_boost / sharp_boost,
                "finite_j_sharp_coefficient": float(
                    source["finite_j_sharp_coefficient"]
                ),
                "small_j_sharp_coefficient": float(
                    source["small_j_sharp_coefficient"]
                ),
                "small_j_one_branch_pinsker_coefficient": float(
                    source["small_j_one_branch_pinsker_coefficient"]
                ),
                "small_j_two_branch_pinsker_coefficient": float(
                    source["small_j_two_branch_pinsker_coefficient"]
                ),
                "small_j_one_branch_to_sharp_ratio": float(
                    source["small_j_one_branch_to_sharp_ratio"]
                ),
                "small_j_two_branch_to_sharp_ratio": float(
                    source["small_j_two_branch_to_sharp_ratio"]
                ),
                "finite_to_small_j_sharp_ratio": float(
                    source["finite_to_small_j_sharp_ratio"]
                ),
                **provenance(
                    master_seed,
                    "constant_gap_point",
                    task_configuration,
                    "analytic_small_J_coefficients_and_exact_finite_J_frontier",
                ),
            }
        )
    return rows


def tail_metric_summary_rows(
    config: dict[str, Any],
    same_j: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    master_seed = int(config["master_seed"])
    grouped = {
        family: [row for row in same_j if row["family"] == family]
        for family in MATCHED_FAMILIES
    }
    mean_coverages = {
        family: float(np.mean([row["J_tail_over_J"] for row in rows]))
        for family, rows in grouped.items()
    }
    ranks = {
        family: rank + 1
        for rank, family in enumerate(
            sorted(MATCHED_FAMILIES, key=lambda item: mean_coverages[item], reverse=True)
        )
    }

    result: list[dict[str, Any]] = []
    for family, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        coverages = np.asarray([row["J_tail_over_J"] for row in ordered], dtype=float)
        boost_fractions = np.asarray(
            [row["tail_boost_fraction_of_sharp"] for row in ordered], dtype=float
        )
        q_values = np.asarray([row["q_alpha"] for row in ordered], dtype=float)
        max_index = int(np.argmax(coverages))
        min_index = int(np.argmin(coverages))
        task_configuration = {
            "family": family,
            "p": MATCHED_P,
            "target_J_fraction": MATCHED_J_FRACTION,
            "alpha_grid": list(PAPER_ALPHA_GRID),
        }
        result.append(
            {
                "family": family,
                "p": MATCHED_P,
                "target_J_fraction": MATCHED_J_FRACTION,
                "target_J_bits": float(ordered[0]["target_J_bits"]),
                "actual_J_bits": float(ordered[0]["actual_J_bits"]),
                "J_abs_mismatch_bits": float(ordered[0]["J_abs_mismatch_bits"]),
                "AUC": float(ordered[0]["AUC"]),
                "alpha_count": len(ordered),
                "mean_J_tail_over_J": float(np.mean(coverages)),
                "median_J_tail_over_J": float(np.median(coverages)),
                "min_J_tail_over_J": float(coverages[min_index]),
                "max_J_tail_over_J": float(coverages[max_index]),
                "alpha_at_min_J_tail_over_J": float(ordered[min_index]["alpha"]),
                "alpha_at_max_J_tail_over_J": float(ordered[max_index]["alpha"]),
                "mean_tail_boost_fraction_of_sharp": float(np.mean(boost_fractions)),
                "min_tail_boost_fraction_of_sharp": float(np.min(boost_fractions)),
                "max_tail_boost_fraction_of_sharp": float(np.max(boost_fractions)),
                "q_alpha_at_smallest_alpha": float(q_values[0]),
                "q_alpha_at_largest_alpha": float(q_values[-1]),
                "max_q_alpha": float(np.max(q_values)),
                "tail_coverage_rank": ranks[family],
                **provenance(
                    master_seed,
                    "tail_metric_family_summary",
                    task_configuration,
                    "deterministic_summary_of_equal_J_tail_metrics",
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    config, config_path = load_profile(args.profile)
    configured_workers = max(1, int(config.get("workers", 1)))
    workers = max(1, min(int(args.workers), configured_workers))
    output_dir = ROOT / "result/table"
    print(
        f"[07] profile={config['profile']} config={config_path.name} workers={workers} "
        f"full_grid={bool(config.get('frontier_full_grid', False))}",
        flush=True,
    )

    frontier = frontier_rows(config)
    same_j = same_j_tail_rows(config)
    constant_gap = constant_gap_rows(config, frontier)
    summary = tail_metric_summary_rows(config, same_j)

    atomic_csv(pd.DataFrame(frontier), output_dir / "07_frontier_comparison.csv")
    atomic_csv(pd.DataFrame(same_j), output_dir / "07_same_J_tail_comparison.csv")
    atomic_csv(pd.DataFrame(constant_gap), output_dir / "07_constant_gap.csv")
    atomic_csv(pd.DataFrame(summary), output_dir / "07_tail_metric_summary.csv")
    print(
        f"[07] complete: frontier={len(frontier)} same_J={len(same_j)} "
        f"constant_gap={len(constant_gap)} summaries={len(summary)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
