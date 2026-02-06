"""ACLI: Attention-Constrained Leaderless Inference.

Core code for experiments accompanying the paper:
"Attention-Constrained Leaderless Inference: Fundamental Limits Under Log-Loss"

This package is intentionally small and modular:
- screening models (produce cheap scores)
- verification models (produce expensive signals)
- policies (how to spend verification budget)
- benchmark/theory utilities (closed-form/asymptotic lines)
- reproducible simulation helpers

"""

from .utils import set_seed, get_device
from .screening import ScreeningModel, GaussianMixtureScreening
from .policy import Policy, TopBPolicy, RandomPolicy
from .benchmark import (
    benchmark_asymptotic_top_mass_sum_gaussian_mixture,
    benchmark_simulate_topB_sum_eta,
    simulate_hits_topB,
    estimate_J_monte_carlo,
)

__all__ = [
    "set_seed",
    "get_device",
    "ScreeningModel",
    "GaussianMixtureScreening",
    "Policy",
    "TopBPolicy",
    "RandomPolicy",
    "benchmark_asymptotic_top_mass_sum_gaussian_mixture",
    "benchmark_simulate_topB_sum_eta",
    "simulate_hits_topB",
    "estimate_J_monte_carlo",
]
__version__ = "0.1.0"
