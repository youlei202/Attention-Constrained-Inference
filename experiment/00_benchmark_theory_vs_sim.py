"""Experiment 00: Appendix A benchmark — theory line vs simulation.

Goal:
  Compare (via CSV) the asymptotic theory line
  E[sum_{top-B} eta] ≈ K * E[eta * 1{top-α}]
with Monte Carlo simulation under top-B selection.

Output:
  result/table/00_benchmark_theory_vs_sim.csv
"""

import sys
from pathlib import Path

import pandas as pd

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import set_seed, RunMeta, get_device
from acli.screening import GaussianMixtureScreening
from acli.benchmark import (
    benchmark_asymptotic_top_mass_sum_gaussian_mixture,
    benchmark_simulate_topB_sum_eta,
)
from acli.revision.reproducibility import canonical_configuration, stable_task_seed
from acli.revision.suite import git_commit


def main():
    # ---------- defaults (no CLI args) ----------
    seed = 123
    set_seed(seed)

    # If torch+CUDA exists, we will automatically use it.
    device = get_device(prefer_cuda=True)

    p = 0.01
    mu = 1.0
    alpha = 0.02  # B/K
    K_list = [500, 1000, 2000, 5000, 10000, 20000]
    n_trials = 700

    out_csv = ROOT / "result" / "table" / "00_benchmark_theory_vs_sim.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianMixtureScreening(p=p, mu=mu, sigma=1.0)

    rows = []
    for K in K_list:
        B = int(alpha * K)
        configuration = {
            "K": K,
            "B": B,
            "alpha": alpha,
            "p": p,
            "mu": mu,
            "n_trials": n_trials,
        }
        task_seed = stable_task_seed(seed, "legacy_00", "top_B_simulation", configuration)
        theory = benchmark_asymptotic_top_mass_sum_gaussian_mixture(model, K=K, alpha=alpha)
        sim, se = benchmark_simulate_topB_sum_eta(
            model, K=K, B=B, n_trials=n_trials, seed=task_seed, device=device
        )
        rows.append(
            dict(
                K=K,
                B=B,
                alpha=alpha,
                p=p,
                mu=mu,
                n_trials=n_trials,
                device=device,
                theory_E_sumTopB_eta=theory,
                sim_E_sumTopB_eta=sim,
                sim_SE=se,
                seed=task_seed,
                configuration=canonical_configuration(configuration),
                git_commit=git_commit(ROOT),
                method="Gaussian_mixture_exact_tail_formula_and_Monte_Carlo",
            )
        )
        print(f"[{device}] K={K:6d}, B={B:5d} | theory={theory:.6f}, sim={sim:.6f} ± {2*se:.6f}")

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
