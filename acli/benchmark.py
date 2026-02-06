"""Benchmark + simulation helpers.

This module provides:
- Monte Carlo estimation of J=I(T;Z) (screening quality)
- Monte Carlo simulation for top-B selection
- A closed-form asymptotic *theory line* for Gaussian-mixture screening
  for the Appendix A benchmark (Corollary 18 style).

Notes:
- Theorem 6 in the paper upper bounds information gain. In the haystack
  specialization, a core intermediate quantity is the expected number
  of informative verified records ("hits"). We simulate this too.
"""

from __future__ import annotations

from typing import Tuple

import math
import numpy as np

try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

from .screening import GaussianMixtureScreening
from .policy import TopBPolicy


def estimate_J_monte_carlo(model: GaussianMixtureScreening, n: int = 400_000, seed: int = 0) -> float:
    """Convenience wrapper."""
    return model.estimate_J(n=n, seed=seed)


def benchmark_asymptotic_top_mass_sum_gaussian_mixture(
    model: GaussianMixtureScreening,
    K: int,
    alpha: float,
) -> float:
    """Theory line for Appendix A: E[sum_{top-B} η] ≈ K * E[η 1{top-α}]."""
    per = model.benchmark_asymptotic_top_mass_per_record(alpha=alpha)
    return float(K * per)


def benchmark_simulate_topB_sum_eta(
    model: GaussianMixtureScreening,
    K: int,
    B: int,
    n_trials: int = 2000,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Monte Carlo for E[sum_{top-B} η] under top-B policy.

    Returns:
        mean, standard_error

    Implementation notes:
    - If torch+CUDA is available and device=="cuda", this function uses a batched GPU path:
      it draws (n_trials, K) samples at once and uses torch.topk for selection.
    - Otherwise it falls back to a CPU loop (numpy).
    """
    if B <= 0:
        return 0.0, 0.0
    if B > K:
        raise ValueError("Need B <= K")

    # ---------- GPU batched path ----------
    if device == "cuda" and _TORCH_OK and torch.cuda.is_available():
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))

        p = float(model.p)
        mu = float(model.mu)
        sigma = float(model.sigma)

        # sample T and Z
        T = torch.bernoulli(torch.full((n_trials, K), p, device="cuda"), generator=gen)
        Z = torch.randn((n_trials, K), device="cuda", generator=gen) * sigma + mu * T

        eta = model.score_torch(Z).to(torch.float32)

        top_vals = torch.topk(eta, k=B, dim=1, largest=True, sorted=False).values
        sums = top_vals.sum(dim=1)  # (n_trials,)

        mean = float(sums.mean().item())
        se = float(sums.std(unbiased=True).item() / math.sqrt(n_trials))
        return mean, se

    # ---------- CPU fallback ----------
    rng = np.random.default_rng(seed)
    policy = TopBPolicy()

    vals = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        T, Z = model.sample(K, device="cpu", seed=int(rng.integers(0, 2**31 - 1)))
        eta = model.score(Z)
        idx = policy.select(eta, B, rng=rng)
        vals[t] = float(np.sum(eta[idx]))
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n_trials))
    return mean, se


def simulate_hits_topB(
    model: GaussianMixtureScreening,
    K: int,
    B: int,
    n_trials: int = 2000,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Monte Carlo for E[hits] where hits = sum_{i in verified} 1{T_i=1} under top-B selection.

    Returns:
        mean, standard_error

    Same GPU/CPU behavior as benchmark_simulate_topB_sum_eta.
    """
    if B <= 0:
        return 0.0, 0.0
    if B > K:
        raise ValueError("Need B <= K")

    # ---------- GPU batched path ----------
    if device == "cuda" and _TORCH_OK and torch.cuda.is_available():
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))

        p = float(model.p)
        mu = float(model.mu)
        sigma = float(model.sigma)

        T = torch.bernoulli(torch.full((n_trials, K), p, device="cuda"), generator=gen)
        Z = torch.randn((n_trials, K), device="cuda", generator=gen) * sigma + mu * T
        eta = model.score_torch(Z).to(torch.float32)

        top_idx = torch.topk(eta, k=B, dim=1, largest=True, sorted=False).indices  # (n_trials, B)
        hits = T.gather(1, top_idx).sum(dim=1)  # (n_trials,)

        mean = float(hits.mean().item())
        se = float(hits.std(unbiased=True).item() / math.sqrt(n_trials))
        return mean, se

    # ---------- CPU fallback ----------
    rng = np.random.default_rng(seed)
    policy = TopBPolicy()

    vals = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        T, Z = model.sample(K, device="cpu", seed=int(rng.integers(0, 2**31 - 1)))
        eta = model.score(Z)
        idx = policy.select(eta, B, rng=rng)
        vals[t] = float(np.sum(T[idx]))
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(n_trials))
    return mean, se
