#!/usr/bin/env python3
"""Calibrated finite-length validation data for paper Figure 6."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.channels import make_continuous_channel_for_auc
from acli.revision.finite_k import finite_k_top_b_precision, large_k_top_tail_precision
from acli.revision.frontier import q_one_branch, q_one_branch_raw, q_pinsker_clipped, q_star
from acli.revision.information import h2
from acli.revision.reproducibility import canonical_configuration, make_rng, stable_task_seed
from acli.revision.suite import git_commit, load_profile


TARGET_AUCS = (0.55, 0.70, 0.79, 0.90)
B_GRID = (10, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 1600, 2000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", help="smoke, paper, or profile YAML")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def gaussian_tail_mean(alpha: float) -> float:
    threshold = float(norm.isf(alpha))
    return float(norm.pdf(threshold) / alpha)


def finite_pool_oracle_hits(K: int, p: float, B: int) -> float:
    """Exact E[min(B,N)] for N~Binomial(K,p), evaluated by survival sums."""
    return float(np.sum(binom.sf(np.arange(B), K, p)))


def simulate_gain_grid(
    channel,
    K: int,
    budgets: tuple[int, ...],
    repetitions: int,
    delta: float,
    seed: int,
) -> dict[int, tuple[float, float, float, float]]:
    """Jointly simulate all budgets while selecting with argpartition."""
    rng = make_rng(seed)
    max_budget = max(budgets)
    gains = {budget: np.empty(repetitions, dtype=np.float64) for budget in budgets}
    hits = {budget: np.empty(repetitions, dtype=np.float64) for budget in budgets}
    gain_correct = 1.0 + math.log2(1.0 - delta)
    gain_flip = 1.0 + math.log2(delta)
    chunk_size = min(64, repetitions)
    offset = 0
    while offset < repetitions:
        count = min(chunk_size, repetitions - offset)
        eta = channel.sample_eta(count * K, rng=rng).reshape(count, K)
        candidate_indices = np.argpartition(eta, K - max_budget, axis=1)[:, -max_budget:]
        candidates = np.take_along_axis(eta, candidate_indices, axis=1)
        descending = np.take_along_axis(
            candidates,
            np.argsort(candidates, axis=1)[:, ::-1],
            axis=1,
        )
        informative = rng.binomial(1, descending).astype(np.float64)
        flipped = rng.random(descending.shape) < delta
        realized = informative * np.where(flipped, gain_flip, gain_correct)
        cumulative_gain = np.cumsum(realized, axis=1)
        cumulative_hits = np.cumsum(informative, axis=1)
        for budget in budgets:
            gains[budget][offset : offset + count] = cumulative_gain[:, budget - 1]
            hits[budget][offset : offset + count] = cumulative_hits[:, budget - 1]
        offset += count

    output: dict[int, tuple[float, float, float, float]] = {}
    for budget in budgets:
        output[budget] = (
            float(gains[budget].mean()),
            float(gains[budget].std(ddof=1) / math.sqrt(repetitions)),
            float(hits[budget].mean()),
            float(hits[budget].std(ddof=1) / math.sqrt(repetitions)),
        )
    return output


def main() -> None:
    args = parse_args()
    profile, profile_path = load_profile(args.profile)
    master_seed = int(profile["master_seed"])
    repetitions = int(profile["reconstructed_fig6_reps"])
    p = 0.01
    K = 10_000
    delta = 0.10
    i_ver = 1.0 - float(h2(delta))
    rows: list[dict] = []

    for target_auc in TARGET_AUCS:
        channel = make_continuous_channel_for_auc("gaussian", p, target_auc)
        J_bits = channel.mutual_information_bits()
        realized_auc = channel.auc()
        joint_configuration = {
            "family": "gaussian",
            "p": p,
            "target_auc": target_auc,
            "epsilon": channel.epsilon,
            "intercept": channel.intercept,
            "K": K,
            "B_grid": B_GRID,
            "delta": delta,
            "repetitions": repetitions,
            "profile": profile["profile"],
        }
        task_seed = stable_task_seed(master_seed, "05_finite_length", "scenario_MC", joint_configuration)
        monte_carlo = simulate_gain_grid(channel, K, B_GRID, repetitions, delta, task_seed)
        print(
            f"[05] AUC={target_auc:.2f} epsilon={channel.epsilon:.6g} "
            f"J={J_bits:.6g} reps={repetitions}",
            flush=True,
        )

        for B in B_GRID:
            alpha = B / K
            q_finite = finite_k_top_b_precision(channel, K, B)
            q_tail = large_k_top_tail_precision(channel, alpha)
            m_g = gaussian_tail_mean(alpha)
            weak_epsilon_q = float(
                np.clip(p + channel.epsilon * p * (1.0 - p) * m_g, p, min(1.0, p / alpha))
            )
            weak_j_q = float(
                np.clip(
                    p + m_g * math.sqrt(2.0 * math.log(2.0) * p * (1.0 - p) * J_bits),
                    p,
                    min(1.0, p / alpha),
                )
            )
            sharp_q = q_star(p, alpha, J_bits)
            pinsker_two_q = q_pinsker_clipped(p, alpha, J_bits)
            pinsker_original_raw_q = q_one_branch_raw(p, alpha, J_bits)
            pinsker_original_clipped_q = q_one_branch(p, alpha, J_bits)
            oracle_hits = finite_pool_oracle_hits(K, p, B)
            pinsker_original_pool_clipped_hits = min(
                B, oracle_hits, B * pinsker_original_raw_q
            )
            mc_gain, mc_gain_se, mc_hits, mc_hits_se = monte_carlo[B]
            row_configuration = {**joint_configuration, "B": B, "alpha": alpha}
            rows.append(
                {
                    "scenario": f"auc_{target_auc:.2f}",
                    "family": "gaussian",
                    "target_auc": target_auc,
                    "realized_auc": realized_auc,
                    "auc_hat": realized_auc,
                    "epsilon": channel.epsilon,
                    "eps": channel.epsilon,
                    "intercept": channel.intercept,
                    "calibration_error": channel.calibration_error,
                    "p": p,
                    "p0": p,
                    "p_hat": p,
                    "J_bits": J_bits,
                    "J": J_bits,
                    "K": K,
                    "B": B,
                    "alpha": alpha,
                    "oversampling_ratio": K / B,
                    "delta": delta,
                    "I_ver_bits": i_ver,
                    "mG": m_g,
                    "n_trials": repetitions,
                    "n_for_J": 0,
                    "n_for_auc": 0,
                    "mc_gain_bits": mc_gain,
                    "mc_gain_se_bits": mc_gain_se,
                    "mc_hits": mc_hits,
                    "mc_hits_se": mc_hits_se,
                    "exact_finite_K_precision": q_finite,
                    "exact_finite_K_gain_bits": i_ver * B * q_finite,
                    "large_K_top_tail_precision": q_tail,
                    "large_K_top_tail_gain_bits": i_ver * B * q_tail,
                    "weak_epsilon_precision": weak_epsilon_q,
                    "weak_epsilon_gain_bits": i_ver * B * weak_epsilon_q,
                    "weak_J_precision": weak_j_q,
                    "weak_J_gain_bits": i_ver * B * weak_j_q,
                    "sharp_binary_KL_precision": sharp_q,
                    "sharp_binary_KL_gain_bits": i_ver * B * sharp_q,
                    "two_branch_pinsker_precision": pinsker_two_q,
                    "two_branch_pinsker_gain_bits": i_ver * B * pinsker_two_q,
                    "original_pinsker_jakob_raw_precision": pinsker_original_raw_q,
                    "original_pinsker_jakob_raw_gain_bits": i_ver * B * pinsker_original_raw_q,
                    "original_pinsker_jakob_clipped_precision": pinsker_original_clipped_q,
                    "original_pinsker_jakob_clipped_gain_bits": i_ver
                    * B
                    * pinsker_original_clipped_q,
                    "original_pinsker_jakob_pool_clipped_gain_bits": i_ver
                    * pinsker_original_pool_clipped_hits,
                    # Unqualified compatibility alias: the analytic raw bound,
                    # never the separately labeled finite-pool composite.
                    "original_pinsker_jakob_gain_bits": i_ver * B * pinsker_original_raw_q,
                    "finite_pool_oracle_hits": oracle_hits,
                    "finite_pool_oracle_gain_bits": i_ver * oracle_hits,
                    "baseline_random_gain_bits": i_ver * B * p,
                    "seed": task_seed,
                    "configuration": canonical_configuration(row_configuration),
                    "git_commit": git_commit(ROOT),
                    "method": "Monte_Carlo_gain_and_deterministic_order_statistic_quadrature",
                    # Backward-compatible columns consumed by notebook/05.
                    "sim_gain_bits": mc_gain,
                    "sim_gain_bits_se": mc_gain_se,
                    "sim_gain_over_Iver": mc_gain / i_ver,
                    "sim_gain_over_Iver_se": mc_gain_se / i_ver,
                    "sim_hits": mc_hits,
                    "sim_hits_se": mc_hits_se,
                    "theory_gain_bits": i_ver * B * weak_j_q,
                    "theory_gain_over_Iver": B * weak_j_q,
                    "theorem6_upper_gain_bits": i_ver * B * pinsker_original_raw_q,
                    "theorem6_upper_gain_over_Iver": B * pinsker_original_raw_q,
                    "upper_thm6_pool_gain_bits": i_ver * pinsker_original_pool_clipped_hits,
                    "upper_thm6_pool_gain_over_Iver": pinsker_original_pool_clipped_hits,
                    "oracle_unlimited_gain_bits": i_ver * B,
                    "oracle_gain_bits": i_ver * B,
                    "oracle_pool_hits": oracle_hits,
                    "oracle_pool_gain_bits": i_ver * oracle_hits,
                    "oracle_pool_gain_over_Iver": oracle_hits,
                    "benchmark_gain_bits": i_ver * B * q_finite,
                    "benchmark_gain_over_Iver": B * q_finite,
                }
            )

    output = ROOT / "result/table/05_finite_length_simulation_validation.csv"
    atomic_csv(pd.DataFrame(rows), output)
    print(f"[05] saved {output} from {profile_path}", flush=True)


if __name__ == "__main__":
    main()
