"""Sharp information--selection frontiers for binary relevance labels.

Let ``T`` be Bernoulli with prior ``p`` and let a selector choose a fraction
``alpha`` of records.  If the selected posterior prevalence is ``q``, mass
conservation fixes the prevalence in the unselected branch to

``q0 = (p - alpha*q) / (1 - alpha)``.

Consequently the least binary selection information compatible with ``q`` is

``Psi(q) = alpha D_2(q || p) + (1-alpha) D_2(q0 || p)``.

All information quantities in this module are in bits.  The routines are
scalar on purpose: they are used as reliable numerical building blocks in
experiment grids, where explicit validation is preferable to silent NumPy
broadcasting.
"""

from __future__ import annotations

import math
from typing import Tuple

from .information import binary_kl_bits


_PROBABILITY_TOLERANCE = 2e-14


def _validate_prior_and_fraction(p: float, alpha: float) -> Tuple[float, float]:
    prior = float(p)
    fraction = float(alpha)
    if not math.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("p must be finite and strictly between zero and one")
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    return prior, fraction


def _validate_information(J: float) -> float:
    information = float(J)
    if not math.isfinite(information) or information < 0.0:
        raise ValueError("J must be a finite, nonnegative number of bits")
    return information


def feasible_q_bounds(p: float, alpha: float) -> Tuple[float, float]:
    """Return the exact feasible interval for selected prevalence ``q``.

    The lower endpoint restricts attention to enrichment (rather than
    depletion), while the upper endpoint enforces both ``q <= 1`` and a
    nonnegative posterior prevalence in the unselected branch.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    return prior, min(1.0, prior / fraction)


def q0_from_q(q: float, p: float, alpha: float) -> float:
    """Return the unselected-branch prevalence implied by mass conservation.

    ``q`` must lie on the enriched feasible branch returned by
    :func:`feasible_q_bounds`.  Values within a few ulps of an endpoint are
    accepted and clipped to eliminate harmless roundoff.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    value = float(q)
    if not math.isfinite(value):
        raise ValueError("q must be finite")
    lower, upper = feasible_q_bounds(prior, fraction)
    if value < lower - _PROBABILITY_TOLERANCE or value > upper + _PROBABILITY_TOLERANCE:
        raise ValueError(f"q must lie in the feasible interval [{lower}, {upper}]")
    value = min(upper, max(lower, value))
    complement = (prior - fraction * value) / (1.0 - fraction)
    if complement < -_PROBABILITY_TOLERANCE or complement > 1.0 + _PROBABILITY_TOLERANCE:
        raise ArithmeticError("mass conservation produced an invalid q0")
    return min(1.0, max(0.0, complement))


def binary_selection_information(q: float, p: float, alpha: float) -> float:
    """Evaluate ``Psi_{p,alpha}(q)`` in bits.

    This is exactly ``I(T; S)`` for a binary selection indicator ``S`` with
    ``P(S=1)=alpha`` and ``P(T=1 | S=1)=q``.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    value = float(q)
    complement = q0_from_q(value, prior, fraction)
    information = fraction * float(binary_kl_bits(value, prior))
    information += (1.0 - fraction) * float(binary_kl_bits(complement, prior))
    # KL is nonnegative; direct floating-point evaluation can undershoot zero
    # by a few ulps at q == p.
    if information < 0.0 and information >= -2e-15:
        return 0.0
    return float(information)


def q_star(p: float, alpha: float, J: float) -> float:
    """Invert the exact two-branch information frontier.

    The returned value is the largest feasible enrichment whose binary
    selection information does not exceed ``J``.  If ``J`` reaches the
    information needed for the feasibility ceiling, that ceiling is returned
    exactly.  Otherwise monotone bisection drives the information residual
    far below the repository requirement of ``2e-10`` bits.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    information = _validate_information(J)
    lower, upper = feasible_q_bounds(prior, fraction)
    if information == 0.0:
        return lower

    ceiling_information = binary_selection_information(upper, prior, fraction)
    if information >= ceiling_information:
        return upper

    lo = lower
    hi = upper
    # Psi is continuous and strictly increasing on the enriched branch.  Two
    # hundred iterations are inexpensive and reduce interval error well below
    # double-precision resolution for every valid parameter setting.
    for _ in range(200):
        mid = lo + 0.5 * (hi - lo)
        if mid == lo or mid == hi:
            break
        value = binary_selection_information(mid, prior, fraction)
        if value <= information:
            lo = mid
        else:
            hi = mid

    # Select the endpoint with the smaller inversion residual.  This matters
    # only at the last bit, but makes result tables stable across platforms.
    lo_residual = abs(binary_selection_information(lo, prior, fraction) - information)
    hi_residual = abs(binary_selection_information(hi, prior, fraction) - information)
    return lo if lo_residual <= hi_residual else hi


def q_one_branch(p: float, alpha: float, J: float, *, clip: bool = True) -> float:
    """Return the legacy one-branch Pinsker enrichment bound.

    The legacy argument retains only the selected branch and gives

    ``p + sqrt(log(2) * J / (2*alpha))``.

    By default the result is clipped to the exact feasibility ceiling.  Pass
    ``clip=False`` (or call :func:`q_one_branch_raw`) when reproducing a table
    that reports the historical, potentially vacuous raw bound.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    information = _validate_information(J)
    value = prior + math.sqrt(math.log(2.0) * information / (2.0 * fraction))
    if clip:
        value = min(value, feasible_q_bounds(prior, fraction)[1])
    return float(value)


def q_one_branch_raw(p: float, alpha: float, J: float) -> float:
    """Return the un-clipped legacy one-branch Pinsker bound."""

    return q_one_branch(p, alpha, J, clip=False)


def q_pinsker_raw(p: float, alpha: float, J: float) -> float:
    """Return the raw two-branch Pinsker bound, before feasibility clipping.

    Applying Pinsker to both branches of :func:`binary_selection_information`
    yields

    ``q <= p + sqrt(log(2) * J * (1-alpha) / (2*alpha))``.
    """

    prior, fraction = _validate_prior_and_fraction(p, alpha)
    information = _validate_information(J)
    return float(
        prior
        + math.sqrt(
            math.log(2.0) * information * (1.0 - fraction) / (2.0 * fraction)
        )
    )


def q_pinsker_clipped(p: float, alpha: float, J: float) -> float:
    """Return the two-branch Pinsker bound clipped to exact feasibility."""

    raw = q_pinsker_raw(p, alpha, J)
    return min(raw, feasible_q_bounds(p, alpha)[1])


__all__ = [
    "feasible_q_bounds",
    "q0_from_q",
    "binary_selection_information",
    "q_star",
    "q_one_branch",
    "q_one_branch_raw",
    "q_pinsker_raw",
    "q_pinsker_clipped",
]
