"""Calibrated weak-screening achievability experiment.

The historical filename is retained for compatibility.  The implemented
posterior is exactly calibrated at every finite epsilon:

    eta(g) = sigmoid(a_epsilon + epsilon*g),  E[eta(G)] = p.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.channels import make_continuous_channel
from acli.revision.reproducibility import canonical_configuration, make_rng, stable_task_seed
from acli.revision.suite import git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="use a small regression grid")
    return parser.parse_args()


def mG_gaussian(alpha: float) -> float:
    quantile = float(norm.isf(alpha))
    return float(norm.pdf(quantile) / alpha)


def simulate_top_b(channel, K: int, B: int, reps: int, seed: int) -> tuple[float, float]:
    rng = make_rng(seed)
    values = np.empty(reps, dtype=float)
    for trial in range(reps):
        eta = channel.sample_eta(K, rng=rng)
        values[trial] = float(np.sum(eta[np.argpartition(eta, -B)[-B:]]))
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(reps))


def main() -> None:
    args = parse_args()
    master_seed = 2468
    p = 0.01
    alpha = 0.05
    epsilons = (0.02, 0.05, 0.10)
    K_values = (500, 2000) if args.smoke else (2000, 5000, 10000, 20000, 40000)
    reps = 40 if args.smoke else 700
    m_g = mG_gaussian(alpha)
    converse_constant = math.sqrt(math.log(2.0) / 2.0)
    rows: list[dict] = []

    for epsilon in epsilons:
        channel = make_continuous_channel("gaussian", p, epsilon)
        J_bits = channel.mutual_information_bits()
        prediction_constant = math.sqrt(alpha) * m_g * math.sqrt(
            2.0 * math.log(2.0) * p * (1.0 - p)
        )
        for K in K_values:
            B = max(1, int(alpha * K))
            configuration = {
                "p": p,
                "epsilon": epsilon,
                "alpha": alpha,
                "K": K,
                "B": B,
                "reps": reps,
            }
            seed = stable_task_seed(master_seed, "legacy_03", "top_b", configuration)
            simulated, se = simulate_top_b(channel, K, B, reps, seed)
            baseline = B * p
            scale = math.sqrt(max(J_bits * B * K, np.finfo(float).tiny))
            weak_screening_prediction = baseline + prediction_constant * scale
            one_branch_pinsker_bound = baseline + converse_constant * scale
            rows.append(
                {
                    "eps": epsilon,
                    "epsilon": epsilon,
                    "intercept": channel.intercept,
                    "calibration_error": channel.calibration_error,
                    "K": K,
                    "B": B,
                    "alpha": alpha,
                    "p0": p,
                    "p_hat": p,
                    "J": J_bits,
                    "J_bits": J_bits,
                    "realized_auc": channel.auc(),
                    "n_for_J": 0,
                    "n_trials": reps,
                    "sim_E_hits": simulated,
                    "sim_SE_hits": se,
                    "baseline_Bp": baseline,
                    "sim_bonus_over_Bp": simulated - baseline,
                    "norm_bonus": (simulated - baseline) / scale,
                    "norm_SE": se / scale,
                    "gaussian_tail_mean": m_g,
                    "weak_screening_prediction_coefficient": prediction_constant,
                    "one_branch_pinsker_coefficient": converse_constant,
                    "calibrated_weak_screening_prediction": weak_screening_prediction,
                    "one_branch_pinsker_upper_bound": one_branch_pinsker_bound,
                    # Historical numbered aliases retained for downstream compatibility.
                    "theorem10_mG": m_g,
                    "theorem10_c_pred": prediction_constant,
                    "corollary11_converse_const": converse_constant,
                    "theorem10_inner_prediction": weak_screening_prediction,
                    "corollary11_outer_upper_bound": one_branch_pinsker_bound,
                    "seed": seed,
                    "configuration": canonical_configuration(configuration),
                    "git_commit": git_commit(ROOT),
                    "method": "exact_calibration_plus_Monte_Carlo_top_B",
                }
            )

    # Historical output filename retained for notebook and automation compatibility.
    output = ROOT / "result/table/03_theorem10_achievability_weak_screening.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved calibrated weak-screening results: {output}")


if __name__ == "__main__":
    main()
