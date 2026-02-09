"""Experiment 05: Finite-length simulation validation (Figure 4).

We validate that finite-length Monte Carlo points lie close to the JBK scaling-law
prediction for *information gain under log-loss* when using the Top-B policy.

Model (decoupled claims; Assumption 9)
-------------------------------------
We use the decoupled-claim specialization where each inspected record i has:
- a latent type T_i ∈ {0,1} indicating whether verification is informative,
- an inspection statistic/feature G_i (a cheap screening signal),
- an independent latent claim Θ_i (only revealed via verification when T_i=1).

Screening model ("Logistic Regression")
--------------------------------------
We generate (T,G) via a 1-D logistic regression model:

    G ~ N(0,1),
    η(G) := P(T=1 | G) = sigmoid(logit(p0) + ε·G).

Top-B policy
------------
Inspect K records (observe G_1..G_K), compute η_i=η(G_i), then verify the B
records with the largest η_i (equivalently largest G_i).

Verification channel and log-loss gain
--------------------------------------
For each verified record:
- if T_i=0: verification is uninformative about Θ_i,
- if T_i=1: verification sends V_i through a BSC(δ) from Θ_i (Θ_i ~ Bern(1/2)).

Under log-loss, the *information gain* (in bits) equals

    IG := H(Θ) - D(K,B),

and in the decoupled model it concentrates around

    E[IG] = I_ver · E[hits],

where hits is the number of informative verified records (sum of T_i over
verified indices) and I_ver = 1 - h2(δ) for the BSC(δ) channel.

Theory curve (JBK scaling)
--------------------------
Theorem 10 (Eq. 6/7/8 in the paper) predicts in the weak-screening regime:

    E[IG]/I_ver ≈ min{B, Bp + c_G(p,α) · sqrt(J B K)}

with α=B/K, J=I(T;G) (bits), and for Gaussian G:

    c_G(p,α) = sqrt(2 ln 2 · p(1-p) · α) · m_G(α),
    m_G(α) = E[G | G ≥ q_α],   P(G ≥ q_α)=α.

We plot the above prediction against finite-length Monte Carlo points.

Output
------
Writes:
  result/table/05_finite_length_simulation_validation.csv

The companion notebook renders:
  result/figure/05_finite_length_simulation_validation.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.utils import RunMeta, get_device, h2, set_seed


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    return float(math.log(p / (1.0 - p)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mG_gaussian(alpha: float) -> float:
    """Exact upper-tail mean m_G(α) for G~N(0,1)."""
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("Need alpha in (0,1)")
    q = float(norm.isf(alpha))  # upper (1-α)-quantile
    return float(norm.pdf(q) / alpha)



def expected_min_binomial_normal_approx(K: int, p: float, B: int) -> float:
    """Approximate E[min(B, N)] for N~Binomial(K,p) using a normal approximation."""
    p = float(min(max(p, 0.0), 1.0))
    B = int(B)
    if B <= 0:
        return 0.0
    mu = K * p
    # If B is far above the mean, min(B,N)=N w.h.p., so expectation is mu.
    if B >= K:
        return float(min(B, mu))
    var = K * p * (1.0 - p)
    if var <= 1e-12:
        return float(min(B, mu))
    sigma = var ** 0.5
    z = (B - mu) / sigma
    # E[(X-B)_+] for X~N(mu,sigma^2) is sigma*phi(z) + (mu-B)*(1-Phi(z))
    tail = float(1.0 - norm.cdf(z))
    exc = float(sigma * norm.pdf(z) + (mu - B) * tail)
    return float(mu - exc)


def benchmark_hits_singleletter_logistic_gaussian(p0: float, eps: float, K: int, B: int) -> float:
    """Appendix A (Cor. 18) single-letter benchmark for E[hits] under Top-B.

    With G~N(0,1), score η(G)=sigmoid(logit(p0)+eps*G), and α=B/K,
        E[hits] ≈ K * E[ η(G) · 1{G ≥ q_α} ],  where P(G≥q_α)=α.

    This avoids the weak-screening linearization and remains accurate for larger eps.
    """
    if B <= 0:
        return 0.0
    if B > K:
        raise ValueError('Need B <= K')
    alpha = B / K
    q = float(norm.isf(alpha))
    s = _logit(p0)

    if abs(eps) < 1e-12:
        # η is constant = p0; tail mass is alpha.
        return float(K * (p0 * alpha))

    def integrand(g: float) -> float:
        # sigmoid(s + eps*g) * phi(g)
        # Use expit-like stable computation.
        x = s + eps * g
        if x >= 0:
            z = math.exp(-x)
            eta = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            eta = z / (1.0 + z)
        return float(eta * norm.pdf(g))

    val, _ = quad(integrand, q, float('inf'), limit=200)
    return float(K * val)

def estimate_p_and_J(p0: float, eps: float, n: int, seed: int) -> tuple[float, float]:
    """Estimate p := P(T=1) and J := I(T;G) (bits) via Monte Carlo.

    For binary T with posterior η(G)=P(T=1|G),
        J = E[ KL(Bern(η) || Bern(p)) ],   where p = E[η].
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n).astype(np.float64)
    eta = _sigmoid(_logit(p0) + eps * g)

    p = float(np.mean(eta))

    eta_clip = np.clip(eta, 1e-12, 1.0 - 1e-12)
    p_clip = float(min(max(p, 1e-12), 1.0 - 1e-12))
    kl = eta_clip * np.log2(eta_clip / p_clip) + (1.0 - eta_clip) * np.log2((1.0 - eta_clip) / (1.0 - p_clip))
    J = float(np.mean(kl))
    return p, J


def estimate_auc(p0: float, eps: float, n: int, seed: int) -> float:
    """Estimate AUC for predicting T from the score η(G).

    Uses the rank-statistic (Mann–Whitney) formula. Ties are negligible
    since η is continuous almost surely.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n).astype(np.float64)
    eta = _sigmoid(_logit(p0) + eps * g)
    t = (rng.random(n) < eta).astype(np.int8)

    n_pos = int(t.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(eta)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)

    rank_sum_pos = float(ranks[t == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def simulate_gain_topB(
    p0: float,
    eps: float,
    delta: float,
    K: int,
    B: int,
    n_trials: int,
    seed: int,
    device: str = "cpu",
) -> tuple[float, float, float, float]:
    """Monte Carlo simulation for information gain under Top-B.

    Returns:
        mean_gain_bits, se_gain_bits, mean_hits, se_hits

    Implementation notes:
    - We simulate the *realized* log-loss reduction (bits) using the BSC(δ)
      verification channel. The gain per informative verification depends only
      on whether the channel flipped (no need to explicitly sample Θ).
    - If torch+CUDA is available and device=="cuda", we use a batched GPU path.
    """
    if B <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if B > K:
        raise ValueError("Need B <= K")

    delta = float(delta)
    if not (0.0 < delta < 0.5):
        raise ValueError("Need delta in (0, 0.5) for a meaningful BSC")

    gain_correct = 1.0 + math.log2(1.0 - delta)
    gain_flip = 1.0 + math.log2(delta)

    # ---------- GPU batched path ----------
    if device == "cuda" and _TORCH_OK and torch is not None and torch.cuda.is_available():
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))

        g = torch.randn((n_trials, K), device="cuda", generator=gen)
        logits = _logit(p0) + float(eps) * g
        eta = torch.sigmoid(logits)

        T = torch.bernoulli(eta, generator=gen)  # (n_trials, K)
        top_idx = torch.topk(eta, k=B, dim=1, largest=True, sorted=False).indices
        Tver = T.gather(1, top_idx)  # (n_trials, B)
        hits = Tver.sum(dim=1)

        # Channel flips for the verified indices (we only count flips when T=1)
        flip = torch.bernoulli(torch.full((n_trials, B), delta, device="cuda"), generator=gen)

        gc = torch.tensor(gain_correct, device="cuda")
        gf = torch.tensor(gain_flip, device="cuda")
        per = torch.where(flip < 0.5, gc, gf)  # (n_trials, B)

        gains = (per * Tver).sum(dim=1)  # (n_trials,)

        mean_gain = float(gains.mean().item())
        se_gain = float(gains.std(unbiased=True).item() / math.sqrt(n_trials))

        mean_hits = float(hits.mean().item())
        se_hits = float(hits.std(unbiased=True).item() / math.sqrt(n_trials))

        return mean_gain, se_gain, mean_hits, se_hits

    # ---------- CPU fallback ----------
    rng = np.random.default_rng(seed)
    s = _logit(p0)

    gains = np.empty(n_trials, dtype=np.float64)
    hits_arr = np.empty(n_trials, dtype=np.float64)

    for t in range(n_trials):
        g = rng.standard_normal(K).astype(np.float64)
        eta = _sigmoid(s + eps * g)

        T = (rng.random(K) < eta).astype(np.int8)
        idx = np.argpartition(eta, -B)[-B:]
        Tver = T[idx]  # 0/1

        # flips over the verified indices
        flips = (rng.random(B) < delta).astype(np.int8)
        per = np.where(flips == 0, gain_correct, gain_flip)

        gains[t] = float(np.sum(per * Tver))
        hits_arr[t] = float(np.sum(Tver))

    mean_gain = float(gains.mean())
    se_gain = float(gains.std(ddof=1) / math.sqrt(n_trials))

    mean_hits = float(hits_arr.mean())
    se_hits = float(hits_arr.std(ddof=1) / math.sqrt(n_trials))

    return mean_gain, se_gain, mean_hits, se_hits


def main():
    # ---------- defaults ----------
    seed = 505
    set_seed(seed)

    device = get_device(prefer_cuda=True)

    # Inspection budget (attention) and verification budgets (vary B)
    # NOTE: K=1e4 is CPU-friendly; set K=1e5 for a heavier run.
    K = 10_000
    B_list = [10, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 1600, 2000]
    B_list = [B for B in B_list if B <= K]

    # Logistic-regression screening parameters
    p0 = 0.01  # base prevalence parameter inside the logit intercept

    # Robustness sweep: weak vs stronger screening (target AUC roughly ~0.55 / 0.7 / 0.8 / 0.9)
    scenarios = [
        dict(name="weak", eps=0.20),
        dict(name="auc~0.7", eps=0.75),
        dict(name="auc~0.8", eps=1.20),
        # Stronger screening for a 4-panel figure (AUC typically ≳0.88–0.92 depending on p0)
        dict(name="auc~0.9", eps=2.00),
    ]

    # Verification channel (BSC)
    delta = 0.10
    I_ver = 1.0 - h2(delta)

    # Monte Carlo budgets
    n_trials = 400
    n_for_J = 500_000
    n_for_auc = 200_000

    out_csv = ROOT / "result" / "table" / "05_finite_length_simulation_validation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for sc in scenarios:
        eps = float(sc["eps"])
        name = str(sc["name"])

        # Estimate p and J once per scenario
        p_hat, J = estimate_p_and_J(p0=p0, eps=eps, n=n_for_J, seed=seed + int(1e6 * eps))
        auc_hat = estimate_auc(p0=p0, eps=eps, n=n_for_auc, seed=seed + int(2e6 * eps))

        print(f"\n[{name}] eps={eps:.3f} | estimated p≈{p_hat:.5f}, J≈{J:.3e} bits, AUC≈{auc_hat:.3f}")

        for B in B_list:
            alpha = B / K
            mG = mG_gaussian(alpha)

            # Eq. (7) constant (Gaussian scores)
            cG = math.sqrt(2.0 * math.log(2.0) * p_hat * (1.0 - p_hat) * alpha) * mG

            denom = math.sqrt(max(J * B * K, 1e-30))

            # Finite-pool oracle: even with perfect knowledge of T, we cannot exceed
            # the number of informative items available among the K inspected records.
            oracle_pool_hits = expected_min_binomial_normal_approx(K=K, p=p_hat, B=B)

            # Weak-screening theory (Eq. 6/8), clipped by the finite pool.
            theory_hits = min(B, oracle_pool_hits, B * p_hat + cG * denom)
            theory_gain_bits = I_ver * theory_hits

            # Appendix A single-letter benchmark (Cor. 18): works beyond weak-screening.
            benchmark_hits = benchmark_hits_singleletter_logistic_gaussian(p0=p0, eps=eps, K=K, B=B)
            benchmark_hits = min(B, oracle_pool_hits, benchmark_hits)
            benchmark_gain_bits = I_ver * benchmark_hits

            # Theorem 6 converse (optional reference), also clipped by the finite pool.
            upper_hits = min(B, oracle_pool_hits, B * p_hat + math.sqrt((math.log(2.0) / 2.0) * J * B * K))
            upper_gain_bits = I_ver * upper_hits

            # Monte Carlo simulation of realized log-loss reduction
            sim_gain_bits, sim_gain_se, sim_hits, sim_hits_se = simulate_gain_topB(
                p0=p0,
                eps=eps,
                delta=delta,
                K=K,
                B=B,
                n_trials=n_trials,
                seed=seed + 10_000 * int(1000 * eps) + B,
                device=device,
            )

            rows.append(
                dict(
                    scenario=name,
                    eps=eps,
                    p0=p0,
                    p_hat=p_hat,
                    J=J,
                    auc_hat=auc_hat,
                    K=K,
                    B=B,
                    alpha=alpha,
                    oversampling_ratio=K / B,
                    delta=delta,
                    I_ver_bits=I_ver,
                    mG=mG,
                    cG=cG,
                    n_trials=n_trials,
                    n_for_J=n_for_J,
                    n_for_auc=n_for_auc,
                    sim_gain_bits=sim_gain_bits,
                    sim_gain_bits_se=sim_gain_se,
                    sim_gain_over_Iver=sim_gain_bits / I_ver,
                    sim_gain_over_Iver_se=sim_gain_se / I_ver,
                    sim_hits=sim_hits,
                    sim_hits_se=sim_hits_se,
                    theory_gain_bits=theory_gain_bits,
                    theory_gain_over_Iver=theory_hits,
                    theorem6_upper_gain_bits=upper_gain_bits,
                    theorem6_upper_gain_over_Iver=upper_hits,
                    # Combined upper envelope: cannot exceed finite-pool oracle
                    upper_thm6_pool_gain_bits=min(upper_gain_bits, I_ver * oracle_pool_hits, I_ver * B),
                    upper_thm6_pool_gain_over_Iver=min(upper_hits, oracle_pool_hits, B),
                    baseline_random_gain_bits=I_ver * (B * p_hat),

                    # Two oracles: (i) unlimited-supply (B·I_ver) and (ii) finite-pool (≈I_ver·E[min(B,N_inf)])
                    oracle_unlimited_gain_bits=I_ver * B,
                    oracle_gain_bits=I_ver * B,  # backward-compatible alias
                    oracle_pool_hits=oracle_pool_hits,
                    oracle_pool_gain_bits=I_ver * oracle_pool_hits,
                    oracle_pool_gain_over_Iver=oracle_pool_hits,

                    # Theory curves
                    benchmark_gain_bits=benchmark_gain_bits,
                    benchmark_gain_over_Iver=benchmark_hits,

                )
            )

            print(
                f"  B={B:4d} (K/B={K/B:6.1f}) | "
                f"sim IG={sim_gain_bits:8.3f} ± {2*sim_gain_se:7.3f} bits | "
                f"theory={theory_gain_bits:8.3f} bits | "
                f"UB={upper_gain_bits:8.3f} bits"
            )

    meta = RunMeta.now(seed=seed, device=device).__dict__
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v

    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
