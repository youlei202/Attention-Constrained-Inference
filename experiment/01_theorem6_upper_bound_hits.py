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
    J = estimate_J_monte_carlo(model, n=n_for_J, seed=seed)

    rows = []
    for K in K_list:
        sim, se = simulate_hits_topB(model, K=K, B=B, n_trials=n_trials, seed=seed + K, device=device)
        ub = B * p + math.sqrt((math.log(2) / 2.0) * J * B * K)
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
                sim_E_hits=sim,
                sim_SE=se,
                theorem6_upper_bound_hits=ub,
                baseline_Bp=B * p,
            )
        )
        print(f"[{device}] K={K:6d} | sim={sim:.6f} ± {2*se:.6f} | UB={ub:.6f} | Bp={B*p:.6f}")

    meta = RunMeta.now(seed=seed, device=device).__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
