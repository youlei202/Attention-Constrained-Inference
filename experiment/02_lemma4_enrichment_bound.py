"""Experiment 02: Lemma 4 selection enrichment bound.

We construct a selection rule S by verifying the top-α fraction of records by η(Z),
and compare empirical enrichment P(T=1|S=1) to the bound:

  P(T=1|S=1) <= p + sqrt( (ln2)/(2α) * J )

Output:
  result/table/02_lemma4_enrichment_bound.csv
"""

import os
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import set_seed, RunMeta
from acli.screening import GaussianMixtureScreening
from acli.benchmark import estimate_J_monte_carlo


def main():
    # ---------- defaults ----------
    seed = 789
    set_seed(seed)

    p = 0.01
    mu = 0.5
    K = 50_000           # large to estimate conditional probability tightly
    alpha_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    n_trials = 40
    n_for_J = 600_000

    out_csv = ROOT / "result" / "table" / "02_lemma4_enrichment_bound.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianMixtureScreening(p=p, mu=mu, sigma=1.0)
    J = estimate_J_monte_carlo(model, n=n_for_J, seed=seed)

    rng = np.random.default_rng(seed)

    rows = []
    for alpha in alpha_list:
        hits_selected = []
        for t in range(n_trials):
            T, Z = model.sample(K, device="cpu", seed=int(rng.integers(0, 2**31-1)))
            eta = model.score(Z)

            B = int(alpha * K)
            # select top-B by eta (equiv to top-B by Z here)
            idx = np.argpartition(eta, -B)[-B:]
            # empirical enrichment
            hits_selected.append(float(np.mean(T[idx])))

        emp = float(np.mean(hits_selected))
        se = float(np.std(hits_selected, ddof=1) / math.sqrt(n_trials))
        bound = p + math.sqrt((math.log(2) / (2.0 * alpha)) * J)

        rows.append(
            dict(
                K=K,
                alpha=alpha,
                B=int(alpha * K),
                p=p,
                mu=mu,
                J=J,
                n_trials=n_trials,
                n_for_J=n_for_J,
                emp_P_T1_given_selected=emp,
                emp_SE=se,
                lemma4_bound=bound,
            )
        )
        print(f"alpha={alpha:>5.3f} | emp={emp:.6f} ± {2*se:.6f} | bound={bound:.6f}")

    meta = RunMeta.now(seed=seed, device="cpu").__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
