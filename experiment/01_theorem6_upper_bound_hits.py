"""Experiment 01: Theorem 6 (converse) — simulate hits and compare to the upper envelope.

We simulate the expected number of informative verified records ("hits") under a top-B policy:
  hits := sum_{i in verified} 1{T_i=1}

Theorem 6 implies (ignoring I_ver scaling, focusing on hit-count part):
  E[hits] <= Bp + sqrt( (ln 2)/2 * J * B * K )

Output:
  result/table/01_theorem6_upper_bound_hits.csv
"""

import sys
from pathlib import Path
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import set_seed, RunMeta, get_device
from acli.screening import GaussianMixtureScreening
from acli.benchmark import estimate_J_monte_carlo, simulate_hits_topB
from acli.revision.frontier import (
    binary_selection_information,
    feasible_q_bounds,
    q_one_branch_raw,
    q_pinsker_clipped,
    q_star as exact_q_star,
)
from acli.revision.reproducibility import canonical_configuration, stable_task_seed
from acli.revision.suite import git_commit


def main():
    # ---------- defaults ----------
    seed = 456
    set_seed(seed)

    device = get_device(prefer_cuda=True)

    p = 0.01
    mu = 0.5            # weaker screening -> more "haystack"
    B = 20              # fixed verification budget
    K_list = [200, 400, 800, 1500, 3000, 6000, 12000]
    n_trials = 1200
    n_for_J = 400_000

    out_csv = ROOT / "result" / "table" / "01_theorem6_upper_bound_hits.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianMixtureScreening(p=p, mu=mu, sigma=1.0)
    j_configuration = {"p": p, "mu": mu, "sigma": 1.0, "n_for_J": n_for_J}
    j_seed = stable_task_seed(seed, "legacy_01", "estimate_J", j_configuration)
    J = estimate_J_monte_carlo(model, n=n_for_J, seed=j_seed)

    rows = []
    for K in K_list:
        alpha = B / K
        task_configuration = {
            "p": p,
            "mu": mu,
            "K": K,
            "B": B,
            "n_trials": n_trials,
            "n_for_J": n_for_J,
            "J_seed": j_seed,
            "J_bits": J,
        }
        task_seed = stable_task_seed(seed, "legacy_01", "simulate_hits", task_configuration)
        sim, se = simulate_hits_topB(
            model,
            K=K,
            B=B,
            n_trials=n_trials,
            seed=task_seed,
            device=device,
        )
        ub = B * p + math.sqrt((math.log(2) / 2.0) * J * B * K)
        sharp_precision = exact_q_star(p, alpha, J)
        two_branch_precision = q_pinsker_clipped(p, alpha, J)
        one_branch_raw_precision = q_one_branch_raw(p, alpha, J)
        oracle_precision = feasible_q_bounds(p, alpha)[1]
        ceiling_hit = sharp_precision == oracle_precision and J >= binary_selection_information(
            oracle_precision, p, alpha
        ) - 2e-12
        residual = 0.0 if ceiling_hit else abs(
            binary_selection_information(sharp_precision, p, alpha) - J
        )
        rows.append(
            dict(
                K=K,
                B=B,
                p=p,
                mu=mu,
                device=device,
                J=J,
                n_trials=n_trials,
                n_for_J=n_for_J,
                J_seed=j_seed,
                sim_E_hits=sim,
                sim_SE=se,
                theorem6_upper_bound_hits=ub,
                baseline_Bp=B * p,
                alpha=alpha,
                empirical=sim / B,
                q_star=sharp_precision,
                two_branch_pinsker=two_branch_precision,
                one_branch_pinsker=one_branch_raw_precision,
                oracle=oracle_precision,
                empirical_hits=sim,
                q_star_hits=B * sharp_precision,
                two_branch_pinsker_hits=B * two_branch_precision,
                one_branch_pinsker_hits=B * one_branch_raw_precision,
                oracle_hits=B * oracle_precision,
                q_star_residual=residual,
                q_star_ceiling_hit=ceiling_hit,
                seed=task_seed,
                configuration=canonical_configuration(task_configuration),
                git_commit=git_commit(ROOT),
                method="Monte_Carlo_empirical_with_exact_two_branch_frontier",
            )
        )
        print(f"[{device}] K={K:6d} | sim={sim:.6f} ± {2*se:.6f} | UB={ub:.6f} | Bp={B*p:.6f}")

    meta = RunMeta.now(seed=seed, device=device).__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    df["master_seed"] = seed

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
