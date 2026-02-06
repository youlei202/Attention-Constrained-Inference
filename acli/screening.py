"""Screening models.

In the paper's haystack specialization, inspection reveals a cheap statistic Z
that helps identify whether a record is informative (T=1) vs uninformative (T=0),
but does not directly reveal Θ.

We provide a minimal base class and a concrete Gaussian-mixture screening model
often used for synthetic experiments.

Gaussian mixture:
    Z | T=0 ~ N(0, 1)
    Z | T=1 ~ N(mu, 1)
    P(T=1)=p

For this model, the posterior score η(Z)=P(T=1|Z) is a logistic function
of Z, and is monotone in Z when mu>0, which is convenient for benchmark theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

from scipy.stats import norm


class ScreeningModel:
    """Abstract screening model."""

    def sample(self, K: int, device: str = "cpu", seed: Optional[int] = None):
        """Sample (T,Z) pairs for K inspected records."""
        raise NotImplementedError

    def score(self, Z):
        """Compute η(Z)=P(T=1|Z)."""
        raise NotImplementedError

    def estimate_J(self, n: int = 300_000, seed: int = 0) -> float:
        """Monte-Carlo estimate of J=I(T;Z) in bits."""
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianMixtureScreening(ScreeningModel):
    p: float = 0.01
    mu: float = 1.0
    sigma: float = 1.0

    def sample(self, K: int, device: str = "cpu", seed: Optional[int] = None):
        """Sample K i.i.d. (T,Z) pairs.

        Returns:
            T: {0,1} array/tensor of shape (K,)
            Z: real array/tensor of shape (K,)
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        if device == "cuda" and _TORCH_OK and torch.cuda.is_available():
            # Use torch on GPU for speed (best-effort reproducible).
            # We still use numpy RNG to generate seeds for torch to keep behavior stable.
            gen = torch.Generator(device="cuda")
            if seed is not None:
                gen.manual_seed(int(seed))

            T = torch.bernoulli(torch.full((K,), self.p, device="cuda"), generator=gen).to(torch.int64)
            Z = torch.randn((K,), device="cuda", generator=gen) * self.sigma + self.mu * T.to(torch.float32)
            return T, Z

        # CPU / numpy
        T = rng.binomial(1, self.p, size=K).astype(np.int64)
        Z = rng.normal(loc=self.mu * T, scale=self.sigma, size=K).astype(np.float64)
        return T, Z

    def score(self, Z):
        """Posterior η(Z)=P(T=1|Z) for the Gaussian mixture model."""
        # log likelihood ratio f1/f0 for equal variances:
        # log_lr = (mu/sigma^2)*Z - mu^2/(2sigma^2)
        mu = self.mu
        s2 = self.sigma * self.sigma
        log_lr = (mu / s2) * Z - 0.5 * (mu * mu) / s2

        # posterior log-odds = log(p/(1-p)) + log_lr
        log_odds = np.log(self.p / (1.0 - self.p)) + log_lr
        # logistic
        return 1.0 / (1.0 + np.exp(-log_odds))

    def score_torch(self, Z):
        """Torch version of score (expects torch tensor Z)."""
        if not _TORCH_OK:
            raise RuntimeError("torch not available")
        mu = float(self.mu)
        s2 = float(self.sigma * self.sigma)
        log_lr = (mu / s2) * Z - 0.5 * (mu * mu) / s2
        log_odds = np.log(self.p / (1.0 - self.p)) + log_lr
        return 1.0 / (1.0 + torch.exp(-log_odds))

    def estimate_J(self, n: int = 300_000, seed: int = 0) -> float:
        """Estimate J=I(T;Z) in bits via MC: E[log2 P(T|Z)/P(T)]."""
        rng = np.random.default_rng(seed)
        T = rng.binomial(1, self.p, size=n).astype(np.int64)
        Z = rng.normal(loc=self.mu * T, scale=self.sigma, size=n).astype(np.float64)
        eta = self.score(Z)
        eta = np.clip(eta, 1e-12, 1 - 1e-12)
        term = np.where(T == 1, np.log2(eta / self.p), np.log2((1.0 - eta) / (1.0 - self.p)))
        return float(np.mean(term))

    def mixture_tail_prob(self, tau: float) -> float:
        """P(Z >= tau) under the marginal mixture distribution."""
        # Z|T=0 ~ N(0,sigma^2); Z|T=1 ~ N(mu,sigma^2)
        s = self.sigma
        p = self.p
        return float(p * norm.sf((tau - self.mu) / s) + (1.0 - p) * norm.sf(tau / s))

    def solve_tau_for_alpha(self, alpha: float, tol: float = 1e-10) -> float:
        """Solve tau such that P(Z >= tau)=alpha in the mixture."""
        # monotone decreasing in tau -> bisection
        lo, hi = -20.0 * self.sigma, 20.0 * self.sigma + abs(self.mu)
        # ensure bracket
        def f(t):
            return self.mixture_tail_prob(t) - alpha

        # Expand bracket if necessary
        while f(lo) < 0:
            hi = lo
            lo -= 10.0 * self.sigma
            if lo < -1e6:
                raise RuntimeError("Failed to bracket root on the left")
        while f(hi) > 0:
            lo = hi
            hi += 10.0 * self.sigma
            if hi > 1e6:
                raise RuntimeError("Failed to bracket root on the right")

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                break
        return 0.5 * (lo + hi)

    def benchmark_asymptotic_top_mass_per_record(self, alpha: float) -> float:
        """Asymptotic per-record top-α mass E[ η(Z) 1{in top-α} ].

        For this model (mu>0), η is strictly increasing in Z, so selecting top-α by η
        equals thresholding Z at tau where P(Z>=tau)=α. Then:

            E[ η(Z) 1{Z>=tau} ] = p * P(Z>=tau | T=1)
                              = p * sf((tau-mu)/sigma)

        This is a convenient closed form to draw a *theory line* for the Appendix A
        benchmark (Corollary 18).
        """
        tau = self.solve_tau_for_alpha(alpha)
        return float(self.p * norm.sf((tau - self.mu) / self.sigma))
