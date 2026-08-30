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
from acli.revision.reproducibility import canonical_configuration, make_rng, stable_task_seed
from acli.revision.suite import git_commit


def main():
    # ---------- defaults ----------
    seed = 789
    set_seed(seed)

    p = 0.01
    mu = 0.5
    K = 50_000           # large to estimate conditional probability tightly
    alpha_list = np.arange(0.5, 0.001, -0.001)  # from 5% to 0.5% by 0.5%
    n_trials = 40
    n_for_J = 600_000

    out_csv = ROOT / "result" / "table" / "02_lemma4_enrichment_bound.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianMixtureScreening(p=p, mu=mu, sigma=1.0)
    j_configuration = {"p": p, "mu": mu, "sigma": 1.0, "n_for_J": n_for_J}
    j_seed = stable_task_seed(seed, "legacy_02", "estimate_J", j_configuration)
    J = estimate_J_monte_carlo(model, n=n_for_J, seed=j_seed)

    # All alpha values can share one ranked pool per trial.  This preserves the
    # marginal Monte Carlo experiment while avoiding nearly one billion
    # redundant score draws and selections in the historical nested loop.
    pool_configuration = {
        "K": K,
        "alpha_list": [float(value) for value in alpha_list],
        "p": p,
        "mu": mu,
        "n_trials": n_trials,
    }
    pool_seed = stable_task_seed(seed, "legacy_02", "shared_ranked_pool", pool_configuration)
    rng = make_rng(pool_seed)
    enrichments = np.empty((n_trials, len(alpha_list)), dtype=np.float64)
    budgets = np.asarray([int(alpha * K) for alpha in alpha_list], dtype=np.int64)
    for trial in range(n_trials):
        T, Z = model.sample(K, device="cpu", seed=int(rng.integers(0, 2**31 - 1)))
        order = np.argsort(Z)[::-1]
        cumulative_hits = np.cumsum(T[order], dtype=np.float64)
        enrichments[trial] = cumulative_hits[budgets - 1] / budgets

    rows = []
    for column, alpha in enumerate(alpha_list):
        B = int(budgets[column])
        emp = float(np.mean(enrichments[:, column]))
        se = float(np.std(enrichments[:, column], ddof=1) / math.sqrt(n_trials))
        bound = p + math.sqrt((math.log(2) / (2.0 * alpha)) * J)

        configuration = {
            "K": K,
            "alpha": float(alpha),
            "B": B,
            "p": p,
            "mu": mu,
            "n_trials": n_trials,
            "n_for_J": n_for_J,
            "J_seed": j_seed,
            "pool_seed": pool_seed,
        }

        rows.append(
            dict(
                K=K,
                alpha=alpha,
                B=B,
                p=p,
                mu=mu,
                J=J,
                n_trials=n_trials,
                n_for_J=n_for_J,
                J_seed=j_seed,
                seed=pool_seed,
                emp_P_T1_given_selected=emp,
                emp_SE=se,
                lemma4_bound=bound,
                configuration=canonical_configuration(configuration),
                git_commit=git_commit(ROOT),
                method="shared_ranked_pool_Monte_Carlo",
            )
        )
        print(f"alpha={alpha:>5.3f} | emp={emp:.6f} ± {2*se:.6f} | bound={bound:.6f}")

    meta = RunMeta.now(seed=seed, device="cpu").__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    df["master_seed"] = seed

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
