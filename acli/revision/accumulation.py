r"""Information-accumulation profiles for shared and component targets.

The central object in this module is

.. math::

    \Phi(n) = I(\Theta; W_1, \ldots, W_n),

where the verification observations are conditionally iid given the target.
All information quantities are measured in bits and every returned profile is
indexed from zero (so ``profile[0] == 0``).

The implementation deliberately keeps the shared-target profile separate from
the decoupled linear benchmark.  The former is bounded by target entropy and
has diminishing marginal returns; the latter represents fresh information
about a new target at every hit and therefore has no common finite ceiling.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy


PROFILE_TOLERANCE = 2e-12
_LOG_2 = float(np.log(2.0))

ArrayLike = Union[Sequence[float], np.ndarray]


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number in [0, 1]")
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def _entropy_bernoulli(probability: float) -> float:
    """Binary entropy in bits, including its exact boundary values."""

    p = _unit_interval(probability, "probability")
    return float(-(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)) / _LOG_2)


def _profile_array(phi: ArrayLike, *, minimum_length: int = 1) -> np.ndarray:
    profile = np.asarray(phi, dtype=np.float64)
    if profile.ndim != 1 or profile.size < minimum_length:
        raise ValueError(
            f"phi must be a one-dimensional profile of length at least {minimum_length}"
        )
    if not np.all(np.isfinite(profile)):
        raise ValueError("phi must contain only finite values")
    return profile


def profile_margins(phi: ArrayLike) -> np.ndarray:
    """Return the discrete margins ``Phi(n) - Phi(n - 1)``.

    The returned array has length ``len(phi) - 1``.  No projection or clipping
    is performed, which makes this function useful for numerical audits.
    """

    return np.diff(_profile_array(phi)).astype(np.float64, copy=False)


def profile_violations(
    phi: ArrayLike,
    entropy_ceiling: Optional[float] = None,
) -> dict[str, float]:
    """Summarize monotonicity, concavity, and entropy-ceiling violations.

    Values named ``*_violation`` are nonnegative and equal zero when the
    corresponding property holds exactly.  This function intentionally does
    not apply a tolerance; callers can compare the values with the tolerance
    appropriate for their numerical experiment.
    """

    profile = _profile_array(phi)
    margins = np.diff(profile)
    margin_increases = np.diff(margins)

    minimum_margin = float(np.min(margins)) if margins.size else 0.0
    maximum_margin_increase = (
        float(np.max(margin_increases)) if margin_increases.size else 0.0
    )

    if entropy_ceiling is None:
        maximum_entropy_excess = 0.0
    else:
        ceiling = float(entropy_ceiling)
        if not np.isfinite(ceiling) or ceiling < 0.0:
            raise ValueError("entropy_ceiling must be finite and nonnegative")
        maximum_entropy_excess = max(0.0, float(np.max(profile) - ceiling))

    return {
        "minimum_value": float(np.min(profile)),
        "origin_abs_error": abs(float(profile[0])),
        "minimum_margin": minimum_margin,
        "monotonicity_violation": max(0.0, -minimum_margin),
        "maximum_margin_increase": maximum_margin_increase,
        "concavity_violation": max(0.0, maximum_margin_increase),
        "entropy_ceiling_violation": maximum_entropy_excess,
    }


def assert_valid_profile(
    phi: ArrayLike,
    entropy_ceiling: Optional[float] = None,
    *,
    atol: float = PROFILE_TOLERANCE,
    require_zero_origin: bool = True,
) -> None:
    """Raise ``ValueError`` unless ``phi`` is a valid accumulation profile.

    A valid profile is nonnegative, nondecreasing, and discretely concave.  If
    supplied, ``entropy_ceiling`` is also enforced.  Small floating-point
    excursions up to ``atol`` are accepted; the revision numerical contract
    uses ``2e-12`` by default.
    """

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    diagnostics = profile_violations(phi, entropy_ceiling)
    failures: list[str] = []
    if diagnostics["minimum_value"] < -atol:
        failures.append(f"minimum value {diagnostics['minimum_value']:.3e}")
    if require_zero_origin and diagnostics["origin_abs_error"] > atol:
        failures.append(f"origin error {diagnostics['origin_abs_error']:.3e}")
    if diagnostics["monotonicity_violation"] > atol:
        failures.append(
            f"monotonicity violation {diagnostics['monotonicity_violation']:.3e}"
        )
    if diagnostics["concavity_violation"] > atol:
        failures.append(f"concavity violation {diagnostics['concavity_violation']:.3e}")
    if diagnostics["entropy_ceiling_violation"] > atol:
        failures.append(
            f"entropy-ceiling violation {diagnostics['entropy_ceiling_violation']:.3e}"
        )
    if failures:
        raise ValueError("invalid accumulation profile: " + "; ".join(failures))


def phi_bsc(n: int, pi: float = 0.5, beta: float = 0.1) -> float:
    """Return ``I(Theta; W_1, ..., W_n)`` for repeated BSC observations.

    ``Theta`` is Bernoulli(``pi``), and conditional on ``Theta`` each ``W_i``
    is obtained through an independent binary symmetric channel with crossover
    probability ``beta``.  The computation aggregates sequences by their
    number of ones and evaluates the mixture likelihood in log space, which is
    stable for the paper grid up to at least ``n=2048``.
    """

    n = _nonnegative_integer(n, "n")
    pi = _unit_interval(pi, "pi")
    beta = _unit_interval(beta, "beta")

    target_entropy = _entropy_bernoulli(pi)
    if n == 0 or pi == 0.0 or pi == 1.0 or beta == 0.5:
        return 0.0

    # Relabeling the BSC output turns beta into 1-beta without changing MI.
    beta = min(beta, 1.0 - beta)
    if beta == 0.0:
        return target_entropy

    k = np.arange(n + 1, dtype=np.float64)
    log_choose = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    log_beta = float(np.log(beta))
    log_one_minus_beta = float(np.log1p(-beta))

    # Distribution of K=sum_i W_i conditional on Theta=0 or Theta=1.
    log_p0 = log_choose + k * log_beta + (n - k) * log_one_minus_beta
    log_p1 = log_choose + k * log_one_minus_beta + (n - k) * log_beta
    # ``gammaln``-formed binomial coefficients accumulate a few ulps across a
    # long support.  Renormalizing in log space prevents that harmless PMF drift
    # from appearing as a 1e-12 oscillation once mutual information saturates.
    log_p0 -= logsumexp(log_p0)
    log_p1 -= logsumexp(log_p1)
    log_mixture = logsumexp(
        np.stack((np.log1p(-pi) + log_p0, np.log(pi) + log_p1), axis=0),
        axis=0,
    )

    p0 = np.exp(log_p0)
    p1 = np.exp(log_p1)
    information_nats = (1.0 - pi) * np.sum(p0 * (log_p0 - log_mixture))
    information_nats += pi * np.sum(p1 * (log_p1 - log_mixture))
    information = float(information_nats / _LOG_2)

    # KL is nonnegative and data processing gives the entropy ceiling.  Clip
    # only round-off outside those exact analytic bounds.
    return float(np.clip(information, 0.0, target_entropy))


def phi_bsc_profile(
    n_max: int,
    pi: float = 0.5,
    beta: float = 0.1,
) -> np.ndarray:
    """Return the shared binary-target BSC profile for ``n=0..n_max``."""

    n_max = _nonnegative_integer(n_max, "n_max")
    pi = _unit_interval(pi, "pi")
    beta = _unit_interval(beta, "beta")
    profile = np.fromiter(
        (phi_bsc(n, pi, beta) for n in range(n_max + 1)),
        dtype=np.float64,
        count=n_max + 1,
    )
    assert_valid_profile(profile, _entropy_bernoulli(pi))
    return profile


def component_target_profile(
    n_max: int,
    d: int,
    beta: float = 0.1,
) -> np.ndarray:
    """Return the profile for a ``d``-component balanced binary target.

    Each verification hit names a uniformly sampled component and observes that
    component through a BSC(``beta``).  Component labels are observed.  Thus a
    fixed component receives ``M ~ Binomial(n, 1/d)`` of the first ``n`` hits,
    and symmetry gives

    ``Phi_d(n) = d * E[phi_bsc(M, 1/2, beta)]``.

    This model interpolates between a shared scalar target (``d=1``) and the
    decoupled linear profile as the chance of revisiting a component vanishes.
    Its entropy ceiling is ``d`` bits.
    """

    n_max = _nonnegative_integer(n_max, "n_max")
    if isinstance(d, bool) or not isinstance(d, Integral):
        raise TypeError("d must be a positive integer")
    d = int(d)
    if d <= 0:
        raise ValueError("d must be positive")
    beta = _unit_interval(beta, "beta")

    scalar_profile = phi_bsc_profile(n_max, 0.5, beta)
    if d == 1:
        return scalar_profile

    q = 1.0 / d
    occupancy_pmf = np.array([1.0], dtype=np.float64)
    profile = np.empty(n_max + 1, dtype=np.float64)
    profile[0] = 0.0
    for n in range(1, n_max + 1):
        updated = np.zeros(n + 1, dtype=np.float64)
        updated[:-1] += (1.0 - q) * occupancy_pmf
        updated[1:] += q * occupancy_pmf
        occupancy_pmf = updated
        profile[n] = d * float(np.dot(occupancy_pmf, scalar_profile[: n + 1]))

    # The two analytic ceilings are useful against round-off near saturation.
    per_hit_information = phi_bsc(1, 0.5, beta)
    ceiling = np.minimum(d, np.arange(n_max + 1) * per_hit_information)
    profile = np.minimum(np.maximum(profile, 0.0), ceiling)
    assert_valid_profile(profile, float(d))
    return profile


def decoupled_profile(n_max: int, i_ver: float) -> np.ndarray:
    """Return the linear profile ``Phi(n) = n * i_ver``.

    This is a fresh-target benchmark, not a shared finite-entropy target.  It is
    therefore checked for monotonicity and concavity but not against an entropy
    ceiling.
    """

    n_max = _nonnegative_integer(n_max, "n_max")
    if isinstance(i_ver, bool) or not isinstance(i_ver, Real):
        raise TypeError("i_ver must be a finite nonnegative real number")
    i_ver = float(i_ver)
    if not np.isfinite(i_ver) or i_ver < 0.0:
        raise ValueError("i_ver must be finite and nonnegative")
    profile = np.arange(n_max + 1, dtype=np.float64) * i_ver
    assert_valid_profile(profile)
    return profile


def concave_interpolation(phi: ArrayLike, x: Union[float, ArrayLike]):
    """Piecewise-linearly interpolate a discrete concave profile.

    ``x`` may be a scalar or an array and must lie in ``[0, len(phi)-1]``.
    A scalar input returns ``float``; an array input returns an array with the
    same shape.  Out-of-range extrapolation is rejected because it can silently
    violate the target-entropy ceiling.
    """

    profile = _profile_array(phi)
    assert_valid_profile(profile, require_zero_origin=False)
    points = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain only finite values")
    if np.any(points < 0.0) or np.any(points > profile.size - 1):
        raise ValueError(f"x must lie in [0, {profile.size - 1}]")

    lower = np.floor(points).astype(np.int64)
    upper = np.minimum(lower + 1, profile.size - 1)
    fraction = points - lower
    interpolated = profile[lower] + fraction * (profile[upper] - profile[lower])
    if points.ndim == 0:
        return float(interpolated)
    return np.asarray(interpolated, dtype=np.float64)


def poisson_binomial_pmf(probabilities: ArrayLike) -> np.ndarray:
    """Return the PMF of a sum of independent, non-identical Bernoulli draws.

    The dynamic program is exact up to floating-point arithmetic and handles
    probabilities equal to zero or one without special approximations.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")

    pmf = np.array([1.0], dtype=np.float64)
    for probability in probs:
        updated = np.zeros(pmf.size + 1, dtype=np.float64)
        updated[:-1] += (1.0 - probability) * pmf
        updated[1:] += probability * pmf
        pmf = updated

    # Dynamic convolution is nonnegative analytically.  Normalize the tiny sum
    # drift accumulated over a large budget while retaining exact endpoint mass.
    pmf = np.maximum(pmf, 0.0)
    total = float(np.sum(pmf))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("failed to compute a valid Poisson-binomial PMF")
    pmf /= total
    return pmf


def expected_profile_poisson_binomial(
    phi: ArrayLike,
    probabilities: ArrayLike,
) -> float:
    """Compute ``E[Phi(N)]`` for a Poisson-binomial hit count ``N``.

    This is the deterministic/exact gain conditional on selected posterior hit
    probabilities.  It does not replace the probabilities by their average and
    therefore does not make a binomial approximation.
    """

    profile = _profile_array(phi)
    assert_valid_profile(profile)
    pmf = poisson_binomial_pmf(probabilities)
    if profile.size < pmf.size:
        raise ValueError(
            "phi must contain entries through the number of Bernoulli probabilities"
        )
    return float(np.dot(profile[: pmf.size], pmf))


__all__ = [
    "PROFILE_TOLERANCE",
    "assert_valid_profile",
    "component_target_profile",
    "concave_interpolation",
    "decoupled_profile",
    "expected_profile_poisson_binomial",
    "phi_bsc",
    "phi_bsc_profile",
    "poisson_binomial_pmf",
    "profile_margins",
    "profile_violations",
]
