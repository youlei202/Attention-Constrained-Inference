"""Experiment 03: Theorem 10 / Corollary 11 — weak-screening achievability (√JBK bonus).

We instantiate Assumption 8 (weak screening via a log-odds score) with a 1-D score:
    logit(η) = logit(p0) + ε G,     G ~ N(0,1),
    η = P(T=1 | Z)  (the Bayes posterior score).

We consider a score-based verification policy:
  - inspect K records, compute η_i
  - verify B = ⌊αK⌋ records with the largest η_i (equivalently largest G_i)
  - "hits" := number of informative verified records = sum_{i in verified} 1{T_i=1}

Since E[T_i | η_i] = η_i, we have
    E[hits] = E[ sum_{top-B} η_i ].

Theorem 10 implies that, as ε → 0 and K → ∞ with α fixed,
    E[hits] = B p + c_G(p,α) √(J B K) + o(√(J B K)),
where
    J = I(T;Z),
and (from the proof) the constant can be written as
    c_G(p,α) = √α · m_G(α) · √(2 ln 2 · p(1-p)),
with
    m_G(α) := E[G | G ≥ q_α],   P(G ≥ q_α) = α.

Corollary 11 / Theorem 6 yields a converse constant
    c_upper = √(ln 2 / 2),
so that (in terms of hits)
    E[hits] ≤ B p + c_upper √(J B K).

This experiment:
  - Monte Carlo estimates p := E[η] and J from η (no need to sample T)
  - Monte Carlo simulates E[hits] via sum of the top-B η's
  - reports the normalized bonus (E[hits]-Bp)/√(J B K)

Output:
  result/table/03_theorem10_achievability_weak_screening.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import set_seed, RunMeta


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    return float(math.log(p / (1.0 - p)))


def estimate_p_and_J_from_eta(p0: float, eps: float, n: int, seed: int) -> tuple[float, float]:
    """Estimate p := E[η] and J := I(T;Z) from η samples.

    For binary T, the mutual information can be written as
        J = E[ KL(Bern(η) || Bern(p)) ]   (bits),
    where p = P(T=1) = E[η].
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(n).astype(np.float64)
    eta = _logistic(_logit(p0) + eps * G)

    p = float(np.mean(eta))

    # J = E[ η log2(η/p) + (1-η) log2((1-η)/(1-p)) ]
    eta_clip = np.clip(eta, 1e-12, 1.0 - 1e-12)
    term = eta_clip * np.log2(eta_clip / p) + (1.0 - eta_clip) * np.log2((1.0 - eta_clip) / (1.0 - p))
    J = float(np.mean(term))
    return p, J


def simulate_E_hits_topB(p0: float, eps: float, K: int, B: int, n_trials: int, seed: int) -> tuple[float, float]:
    """Simulate E[hits] under top-B selection via E[sum_{top-B} η]."""
    if B <= 0:
        return 0.0, 0.0
    if B > K:
        raise ValueError("Need B <= K")

    rng = np.random.default_rng(seed)
    vals = np.empty(n_trials, dtype=np.float64)

    s = _logit(p0)
    for t in range(n_trials):
        G = rng.standard_normal(K).astype(np.float64)
        eta = _logistic(s + eps * G)
        # select top-B η
        idx = np.argpartition(eta, -B)[-B:]
        vals[t] = float(np.sum(eta[idx]))

    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n_trials))
    return mean, se


def mG_gaussian(alpha: float) -> float:
    """Exact m_G(α) for G~N(0,1): E[G | G ≥ q_α], with P(G ≥ q_α)=α."""
    q = float(norm.isf(alpha))  # upper (1-α)-quantile
    return float(norm.pdf(q) / alpha)


def main():
    # ---------- defaults (no CLI args) ----------
    seed = 2468
    set_seed(seed)

    # Baseline prevalence parameter used in the logit intercept.
    # NOTE: Under Assumption 8, p should equal E[η]. For symmetric G and eps>0, this is only
    # exact at p=0.5; for other p it differs by O(eps^2). We therefore estimate p := E[η]
    # and use it consistently in Bp, J, and constants.
    p0 = 0.01

    alpha = 0.05  # B/K
    eps_list = [0.02, 0.05, 0.10]  # weak-screening regime: keep eps small

    K_list = [2000, 5000, 10000, 20000, 40000]
    n_trials = 700
    n_for_J = 500_000

    out_csv = ROOT / "result" / "table" / "03_theorem10_achievability_weak_screening.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Theorem constants that depend only on (p,alpha,G-dist)
    mG = mG_gaussian(alpha)
    c_upper = math.sqrt(math.log(2.0) / 2.0)  # converse constant √(ln2/2)

    rows = []
    for eps in eps_list:
        # Estimate p and J for this eps
        p_hat, J = estimate_p_and_J_from_eta(p0=p0, eps=eps, n=n_for_J, seed=seed + int(1e6 * eps))

        # Achievability constant (from Thm 10 proof, Gaussian G)
        c_pred = math.sqrt(alpha) * mG * math.sqrt(2.0 * math.log(2.0) * p_hat * (1.0 - p_hat))

        for K in K_list:
            B = int(alpha * K)
            sim, se = simulate_E_hits_topB(p0=p0, eps=eps, K=K, B=B, n_trials=n_trials, seed=seed + K + int(1e6 * eps))

            baseline = B * p_hat
            denom = math.sqrt(max(J * B * K, 1e-30))
            bonus = sim - baseline
            norm_bonus = bonus / denom
            norm_se = se / denom

            inner_pred = baseline + c_pred * denom
            outer_ub = baseline + c_upper * denom

            rows.append(
                dict(
                    eps=eps,
                    K=K,
                    B=B,
                    alpha=alpha,
                    p0=p0,
                    p_hat=p_hat,
                    J=J,
                    n_for_J=n_for_J,
                    n_trials=n_trials,
                    sim_E_hits=sim,
                    sim_SE_hits=se,
                    baseline_Bp=baseline,
                    sim_bonus_over_Bp=bonus,
                    norm_bonus=norm_bonus,
                    norm_SE=norm_se,
                    theorem10_mG=mG,
                    theorem10_c_pred=c_pred,
                    corollary11_converse_const=c_upper,
                    theorem10_inner_prediction=inner_pred,
                    corollary11_outer_upper_bound=outer_ub,
                )
            )

            print(
                f"eps={eps:>4.2f} | K={K:6d}, B={B:5d} | "
                f"p≈{p_hat:.5f}, J≈{J:.3e} | "
                f"E[hits]≈{sim:.4f} (±{2*se:.4f}) | "
                f"(bonus/√JBK)≈{norm_bonus:.4f} (±{2*norm_se:.4f})"
            )

    meta = RunMeta.now(seed=seed, device="cpu").__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
