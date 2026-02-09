"""Experiment 04: Proposition 12 / 13 — tail leverage and implied information gain.

This experiment targets Section IV-C ("The haystack regime and tail leverage"):

  - Proposition 12: Gaussian scores yield logarithmic tail leverage.
  - Proposition 13: Pareto right tail yields polynomial tail leverage.

Core tail functional (Assumption 8):

    m_G(α) := (1/α) E[ G · 1{G ≥ q_α} ] = E[G | G ≥ q_α],
    where P(G ≥ q_α) = α.

We estimate m_G(α) via Monte Carlo for small α and compare to theory:
  - Gaussian exact identity (truncated normal): m_G(α) = φ(q_α) / α, q_α = Φ^{-1}(1-α).
  - Gaussian asymptotic: √(2 log(1/α)).
  - Pareto asymptotic: C(ν) α^{-1/ν}.

Optional (more intuitive y-axis): implied *information gain*.

In the haystack regime α = B/K → 0, Eq. (8) in the paper rewrites the leading
square-root term in the achievability bound as (normalized by I_ver):

    Gain / I_ver  ≈  Bp  +  √(2 ln 2 · p(1-p)) · m_G(α) · B · √J,

where:
  - p is prevalence P(T=1)
  - J is screening quality I(T;Z) measured in bits
  - B is verification budget (deep attention)
  - I_ver is per-informative-verification information gain

So we also export Gain/I_ver ("gain_over_Iver_*" columns) computed from m_G(α)
under a user-chosen (p, J, B, I_ver). This makes the tail-leverage plots easier
to interpret as "how much information gain you get" when you vary K/B = 1/α.

Output:
  result/table/04_tail_leverage_gaussian_vs_pareto.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import RunMeta, set_seed


def _empirical_tail_mean(G: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """Estimate m_G(α) by averaging the top-⌊αn⌋ samples.

    Returns:
        m_hat: empirical tail mean
        q_hat: empirical threshold (≈ (1-α)-quantile)
        k:     number of samples in the tail
    """
    n = len(G)
    k = max(1, int(round(alpha * n)))

    # threshold for top-k
    thresh = np.partition(G, n - k)[n - k]
    top = G[G >= thresh]
    return float(np.mean(top)), float(thresh), int(k)


def _sample_pareto_tail(n: int, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Sample X with tail P(X>=x) = x^{-nu}, x>=1 (Pareto Type I)."""
    u = rng.random(n)
    u = np.clip(u, 1e-12, 1.0)
    x = u ** (-1.0 / nu)
    return x.astype(np.float64)


def _gain_from_mG(
    mG: float,
    *,
    p: float,
    J_bits: float,
    B: int,
    I_ver: float,
    clip_to_oracle: bool = True,
) -> dict:
    """Map m_G(α) -> information gain using Eq. (8) (haystack rewrite).

    Returns a dict with:
      - gain_over_Iver_raw: Bp + C m_G B √J
      - gain_over_Iver: clipped to ≤ B (oracle) if clip_to_oracle=True
      - gain_bits: I_ver * gain_over_Iver
      - bonus_over_Iver: gain_over_Iver_raw - Bp
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    if J_bits < 0.0:
        raise ValueError("J_bits must be nonnegative")
    if B <= 0:
        raise ValueError("B must be positive")
    if I_ver < 0.0:
        raise ValueError("I_ver must be nonnegative")

    C = math.sqrt(2.0 * math.log(2.0) * p * (1.0 - p))
    bonus = C * float(mG) * float(B) * math.sqrt(float(J_bits))

    baseline = float(B) * float(p)
    raw = baseline + bonus

    clipped = min(float(B), raw) if clip_to_oracle else raw
    return {
        "gain_const_C": C,
        "baseline_over_Iver": baseline,
        "oracle_over_Iver": float(B),
        "bonus_over_Iver": bonus,
        "gain_over_Iver_raw": raw,
        "gain_over_Iver": clipped,
        "gain_bits": float(I_ver) * clipped,
    }


def main() -> None:
    seed = 1357
    set_seed(seed)

    # ------------------ Config for the Gain mapping ------------------
    # These do NOT affect m_G(α). They only affect the derived gain columns.
    p = 0.01
    J_bits = 0.01
    B = 50
    I_ver = 1.0  # set to your verifier's I(Θ;V|T=1) if you want gain in bits

    # ------------------ Sampling config ------------------
    # Choose enough samples so that even α=1e-4 has a few hundred points in the tail.
    n_samples = 2_000_000
    alpha_list = [
        5e-1,
        2e-1,
        1e-1,
        5e-2,
        2e-2,
        1e-2,
        5e-3,
        2e-3,
        1e-3,
        5e-4,
        2e-4,
        1e-4,
    ]

    # Pareto exponent (must be >3 per Prop. 13 assumptions)
    nu = 4.0
    mu = nu / (nu - 1.0)
    var = nu / ((nu - 1.0) ** 2 * (nu - 2.0))
    sigma = math.sqrt(var)
    C_nu = (nu / (nu - 1.0)) / sigma

    out_csv = ROOT / "result" / "table" / "04_tail_leverage_gaussian_vs_pareto.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    # ---------- Gaussian ----------
    G_gauss = rng.standard_normal(n_samples).astype(np.float64)

    for alpha in alpha_list:
        m_hat, q_hat, k = _empirical_tail_mean(G_gauss, alpha)

        q = float(norm.isf(alpha))
        m_exact = float(norm.pdf(q) / alpha)
        m_asym = float(math.sqrt(2.0 * math.log(1.0 / alpha)))

        gain_emp = _gain_from_mG(m_hat, p=p, J_bits=J_bits, B=B, I_ver=I_ver)
        gain_exact = _gain_from_mG(m_exact, p=p, J_bits=J_bits, B=B, I_ver=I_ver)
        gain_asym = _gain_from_mG(m_asym, p=p, J_bits=J_bits, B=B, I_ver=I_ver)

        rows.append(
            {
                "dist": "gaussian",
                "alpha": alpha,
                "oversampling_ratio": 1.0 / alpha,  # K/B
                "n_samples": n_samples,
                "k_tail": k,
                "q_emp": q_hat,
                "mG_emp": m_hat,
                "mG_exact": m_exact,
                "mG_asympt": m_asym,
                "ratio_emp_to_exact": m_hat / m_exact,
                "ratio_emp_to_asympt": m_hat / m_asym,
                # gain mapping parameters
                "p": p,
                "J": J_bits,
                "B": B,
                "I_ver": I_ver,
                # gain columns (emp/exact/asym)
                "gain_over_Iver_emp": gain_emp["gain_over_Iver"],
                "gain_over_Iver_exact": gain_exact["gain_over_Iver"],
                "gain_over_Iver_asympt": gain_asym["gain_over_Iver"],
                "gain_over_Iver_raw_emp": gain_emp["gain_over_Iver_raw"],
                "gain_over_Iver_raw_exact": gain_exact["gain_over_Iver_raw"],
                "gain_over_Iver_raw_asympt": gain_asym["gain_over_Iver_raw"],
                "bonus_over_Iver_emp": gain_emp["bonus_over_Iver"],
                "bonus_over_Iver_exact": gain_exact["bonus_over_Iver"],
                "bonus_over_Iver_asympt": gain_asym["bonus_over_Iver"],
                "baseline_over_Iver": gain_emp["baseline_over_Iver"],
                "oracle_over_Iver": gain_emp["oracle_over_Iver"],
                "gain_bits_emp": gain_emp["gain_bits"],
                "gain_bits_exact": gain_exact["gain_bits"],
                "gain_bits_asympt": gain_asym["gain_bits"],
                "gain_const_C": gain_emp["gain_const_C"],
            }
        )
        print(
            f"[Gaussian] α={alpha:>7.1e} | m_emp={m_hat:.4f} | m_exact={m_exact:.4f} "
            f"| gain/Iver(emp)={gain_emp['gain_over_Iver']:.4f}"
        )

    # ---------- Pareto ----------
    X = _sample_pareto_tail(n_samples, nu=nu, rng=rng)
    G_pareto = (X - mu) / sigma

    for alpha in alpha_list:
        m_hat, q_hat, k = _empirical_tail_mean(G_pareto, alpha)
        m_asym = float(C_nu * (alpha ** (-1.0 / nu)))

        gain_emp = _gain_from_mG(m_hat, p=p, J_bits=J_bits, B=B, I_ver=I_ver)
        gain_asym = _gain_from_mG(m_asym, p=p, J_bits=J_bits, B=B, I_ver=I_ver)

        rows.append(
            {
                "dist": f"pareto_nu_{nu:g}",
                "alpha": alpha,
                "oversampling_ratio": 1.0 / alpha,  # K/B
                "n_samples": n_samples,
                "k_tail": k,
                "q_emp": q_hat,
                "mG_emp": m_hat,
                "mG_exact": np.nan,  # no simple closed form for standardized G
                "mG_asympt": m_asym,
                "ratio_emp_to_exact": np.nan,
                "ratio_emp_to_asympt": m_hat / m_asym,
                # pareto params
                "nu": nu,
                "pareto_mu": mu,
                "pareto_sigma": sigma,
                "pareto_C_nu": C_nu,
                # gain mapping parameters
                "p": p,
                "J": J_bits,
                "B": B,
                "I_ver": I_ver,
                # gain columns (emp/asym)
                "gain_over_Iver_emp": gain_emp["gain_over_Iver"],
                "gain_over_Iver_exact": np.nan,
                "gain_over_Iver_asympt": gain_asym["gain_over_Iver"],
                "gain_over_Iver_raw_emp": gain_emp["gain_over_Iver_raw"],
                "gain_over_Iver_raw_exact": np.nan,
                "gain_over_Iver_raw_asympt": gain_asym["gain_over_Iver_raw"],
                "bonus_over_Iver_emp": gain_emp["bonus_over_Iver"],
                "bonus_over_Iver_exact": np.nan,
                "bonus_over_Iver_asympt": gain_asym["bonus_over_Iver"],
                "baseline_over_Iver": gain_emp["baseline_over_Iver"],
                "oracle_over_Iver": gain_emp["oracle_over_Iver"],
                "gain_bits_emp": gain_emp["gain_bits"],
                "gain_bits_exact": np.nan,
                "gain_bits_asympt": gain_asym["gain_bits"],
                "gain_const_C": gain_emp["gain_const_C"],
            }
        )
        print(
            f"[Pareto ν={nu:g}] α={alpha:>7.1e} | m_emp={m_hat:.4f} | C α^(-1/ν)={m_asym:.4f} "
            f"| gain/Iver(emp)={gain_emp['gain_over_Iver']:.4f}"
        )

    meta = RunMeta.now(seed=seed, device="cpu").__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
