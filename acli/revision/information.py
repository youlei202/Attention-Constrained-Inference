"""Information-theoretic primitives used by the major-revision experiments.

All public quantities in this module are measured in **bits**.  The
implementations deliberately handle boundary probabilities exactly: in
particular ``0 * log(0)`` is zero, while a positive mass relative to a zero
reference mass has infinite KL divergence.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


ArrayLike = Union[float, np.ndarray]


def _scalar_if_scalar(value: np.ndarray, *inputs: object) -> ArrayLike:
    """Return a Python float when all inputs were scalar-like."""

    if all(np.ndim(item) == 0 for item in inputs):
        return float(np.asarray(value))
    return value


def _probability_array(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def h2(p: object) -> ArrayLike:
    """Binary entropy ``H(Bernoulli(p))`` in bits.

    Unlike implementations which clip unconditionally, this function returns
    exactly zero at ``p=0`` and ``p=1``.
    """

    probability = _probability_array(p, name="p")
    result = np.zeros_like(probability)
    interior = (probability > 0.0) & (probability < 1.0)
    x = probability[interior]
    result[interior] = -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))
    return _scalar_if_scalar(result, p)


def binary_kl_bits(q: object, p: object) -> ArrayLike:
    """Return ``D(Bernoulli(q) || Bernoulli(p))`` in bits.

    ``q`` and ``p`` follow NumPy broadcasting rules.  Boundary cases use the
    standard extended-real conventions, e.g. ``D(Ber(q)||Ber(0))`` is
    infinite for every ``q > 0``.
    """

    q_array = _probability_array(q, name="q")
    p_array = _probability_array(p, name="p")
    q_broadcast, p_broadcast = np.broadcast_arrays(q_array, p_array)
    result = np.zeros_like(q_broadcast, dtype=np.float64)

    positive_q = q_broadcast > 0.0
    impossible_one = positive_q & (p_broadcast == 0.0)
    result[impossible_one] = np.inf
    regular_one = positive_q & (p_broadcast > 0.0)
    result[regular_one] += q_broadcast[regular_one] * np.log2(
        q_broadcast[regular_one] / p_broadcast[regular_one]
    )

    positive_complement = q_broadcast < 1.0
    impossible_zero = positive_complement & (p_broadcast == 1.0)
    result[impossible_zero] = np.inf
    regular_zero = positive_complement & (p_broadcast < 1.0)
    finite_regular_zero = regular_zero & ~np.isinf(result)
    result[finite_regular_zero] += (1.0 - q_broadcast[finite_regular_zero]) * np.log2(
        (1.0 - q_broadcast[finite_regular_zero])
        / (1.0 - p_broadcast[finite_regular_zero])
    )

    # Avoid displaying harmless negative zeros in tables and CSV files.
    result[result == 0.0] = 0.0
    return _scalar_if_scalar(result, q, p)


def entropy_bits(
    probabilities: object,
    axis: Optional[int] = None,
    *,
    normalization_tolerance: float = 1e-12,
) -> ArrayLike:
    """Shannon entropy of a probability distribution in bits.

    Parameters
    ----------
    probabilities:
        Nonnegative probability masses.  With ``axis=None`` (the default),
        the full array is one distribution.  Otherwise each slice along
        ``axis`` is treated as a distribution.
    axis:
        Axis containing the outcomes, or ``None`` to flatten the input.
    normalization_tolerance:
        Maximum accepted absolute error in the total mass.  Accepted inputs
        are renormalized by their (near-one) sum before entropy is evaluated.
    """

    masses = np.asarray(probabilities, dtype=np.float64)
    if masses.size == 0:
        raise ValueError("probabilities must not be empty")
    if not np.all(np.isfinite(masses)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(masses < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if normalization_tolerance < 0.0:
        raise ValueError("normalization_tolerance must be nonnegative")

    totals = np.sum(masses, axis=axis, keepdims=True, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise ValueError("probabilities must have positive total mass")
    if np.any(np.abs(totals - 1.0) > normalization_tolerance):
        raise ValueError("probabilities must sum to one")
    normalized = masses / totals
    terms = np.zeros_like(normalized)
    positive = normalized > 0.0
    terms[positive] = -normalized[positive] * np.log2(normalized[positive])
    result = np.sum(terms, axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return result


def mutual_information_from_joint(
    joint: object,
    *,
    normalization_tolerance: float = 1e-12,
) -> float:
    """Mutual information of a two-dimensional joint PMF, in bits."""

    probabilities = np.asarray(joint, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("joint must be a two-dimensional probability table")
    if probabilities.size == 0 or 0 in probabilities.shape:
        raise ValueError("joint must not be empty")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("joint must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("joint probabilities must be nonnegative")

    total = float(np.sum(probabilities, dtype=np.float64))
    if total <= 0.0 or abs(total - 1.0) > normalization_tolerance:
        raise ValueError("joint probabilities must sum to one")
    probabilities = probabilities / total
    row_marginal = probabilities.sum(axis=1, keepdims=True)
    column_marginal = probabilities.sum(axis=0, keepdims=True)
    independent = row_marginal * column_marginal

    positive = probabilities > 0.0
    value = float(
        np.sum(
            probabilities[positive]
            * np.log2(probabilities[positive] / independent[positive]),
            dtype=np.float64,
        )
    )
    # Roundoff can produce values around -1e-16 for independent tables.
    if value < 0.0 and value >= -5e-14:
        return 0.0
    return value


def posterior_mutual_information_bits(
    posterior: object,
    p: float,
    weights: Optional[object] = None,
) -> float:
    """Compute ``E[D(Ber(posterior) || Ber(p))]`` in bits.

    This identity is convenient for calibrated posterior-score channels.  It
    is public so deterministic experiments can share the same boundary-safe
    KL implementation as the channel classes.
    """

    values = _probability_array(posterior, name="posterior")
    prior = float(_probability_array(p, name="p"))
    divergence = np.asarray(binary_kl_bits(values, prior), dtype=np.float64)
    if weights is None:
        return float(np.mean(divergence))

    masses = np.asarray(weights, dtype=np.float64)
    if masses.shape != values.shape:
        raise ValueError("weights must have the same shape as posterior")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(masses.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return float(np.sum((masses / total) * divergence, dtype=np.float64))


__all__ = [
    "h2",
    "binary_kl_bits",
    "entropy_bits",
    "mutual_information_from_joint",
    "posterior_mutual_information_bits",
]
