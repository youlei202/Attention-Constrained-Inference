"""Deterministic and Monte Carlo top-``B`` posterior precision.

For iid calibrated posterior scores ``eta_1, ..., eta_K``, top-``B``
verification has expected precision

``q_{K,B} = E[sum of the B largest eta_i] / B``.

The deterministic implementation below integrates the exact order-statistic
weights for continuous channels and uses an exact binomial layer-cake formula
for discrete channels (including score ties).  Sampling helpers use
``numpy.argpartition`` so their cost remains linear in ``K`` up to constants.

Channel objects are intentionally duck typed.  The major-revision channel
classes expose ``posterior_quantile`` and ``sample_eta``; the compatibility
paths also recognize ``quantile``, ``sample_posterior``, and ``sample``.
"""

from __future__ import annotations

import inspect
import math
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy import integrate, special, stats


@runtime_checkable
class PosteriorChannel(Protocol):
    """Structural protocol used by the deterministic and MC routines."""

    p: float

    def posterior_quantile(self, u: Any) -> Any:
        """Quantile function of the marginal calibrated posterior score."""

    def sample_eta(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Draw iid posterior scores."""


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _validate_k_b(K: int, B: int) -> Tuple[int, int]:
    population = _positive_integer(K, name="K")
    budget = _positive_integer(B, name="B")
    if budget > population:
        raise ValueError("B must not exceed K")
    return population, budget


def _validate_alpha(alpha: float) -> float:
    fraction = float(alpha)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("alpha must be finite and lie in (0, 1]")
    return fraction


def _member(channel: object, name: str) -> Any:
    value = getattr(channel, name)
    return value() if callable(value) and name in {"posterior_values", "probabilities"} else value


def _discrete_atoms(channel: object) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not hasattr(channel, "posterior_values") or not hasattr(channel, "probabilities"):
        return None
    values = np.asarray(_member(channel, "posterior_values"), dtype=np.float64).reshape(-1)
    probabilities = np.asarray(_member(channel, "probabilities"), dtype=np.float64).reshape(-1)
    if values.size == 0 or values.shape != probabilities.shape:
        raise ValueError("discrete channel atoms and probabilities must have equal nonzero length")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("posterior_values must be finite and lie in [0, 1]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probabilities.sum(dtype=np.float64))
    if total <= 0.0 or abs(total - 1.0) > 1e-10:
        raise ValueError("discrete channel probabilities must sum to one")
    probabilities = probabilities / total

    # Merge duplicate posterior values.  This both handles score ties exactly
    # and avoids zero-width layers in the layer-cake calculation.
    order = np.argsort(values, kind="stable")
    values = values[order]
    probabilities = probabilities[order]
    unique, inverse = np.unique(values, return_inverse=True)
    masses = np.zeros(unique.size, dtype=np.float64)
    np.add.at(masses, inverse, probabilities)
    positive = masses > 0.0
    return unique[positive], masses[positive]


def _quantile_callable(channel: object) -> Optional[Callable[[Any], Any]]:
    for name in ("posterior_quantile", "quantile", "score_quantile", "ppf"):
        candidate = getattr(channel, name, None)
        if callable(candidate):
            return candidate
    return None


def _evaluate_quantile(quantile: Callable[[Any], Any], u: Any) -> np.ndarray:
    probabilities = np.asarray(u, dtype=np.float64)
    try:
        raw = quantile(probabilities)
    except (TypeError, ValueError):
        # A few lightweight user channels implement scalar-only quantiles.
        raw = np.vectorize(quantile, otypes=[np.float64])(probabilities)
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != probabilities.shape:
        try:
            values = np.broadcast_to(values, probabilities.shape)
        except ValueError as error:
            raise ValueError("channel quantile returned an incompatible shape") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("channel quantile must be finite on the open unit interval")
    if np.any((values < -2e-12) | (values > 1.0 + 2e-12)):
        raise ValueError("channel quantile must return posterior probabilities in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def _expected_capped_binomial(K: int, probability: float, cap: int) -> float:
    """Return ``E[min(Binomial(K, probability), cap)]`` stably."""

    if cap >= K:
        return float(K * probability)
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return float(cap)

    # E[X 1{X <= cap-1}] = K p P(Bin(K-1,p) <= cap-2).
    truncated_first_moment = 0.0
    if cap >= 2:
        truncated_first_moment = K * probability * float(
            stats.binom.cdf(cap - 2, K - 1, probability)
        )
    capped_tail = cap * float(stats.binom.sf(cap - 1, K, probability))
    result = truncated_first_moment + capped_tail
    return min(float(cap), max(0.0, float(result)))


def _finite_discrete_precision(
    values: np.ndarray,
    probabilities: np.ndarray,
    K: int,
    B: int,
) -> float:
    # For nonnegative observations x_(K-B+1),...,x_(K), the sum of the top B
    # values is B*v_min + sum_j (v_j-v_{j-1}) min(B,N_{>=v_j}).
    expected_sum = B * float(values[0])
    upper_tail_mass = 1.0
    for index in range(1, values.size):
        upper_tail_mass -= float(probabilities[index - 1])
        layer_width = float(values[index] - values[index - 1])
        expected_sum += layer_width * _expected_capped_binomial(K, upper_tail_mass, B)
    result = expected_sum / B
    return min(1.0, max(0.0, float(result)))


@lru_cache(maxsize=8)
def _unit_legendre_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return an immutable Gauss--Legendre rule on the unit interval."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = np.asarray(0.5 * (nodes + 1.0), dtype=np.float64)
    weights = np.asarray(0.5 * weights, dtype=np.float64)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _regularized_rank_expectation(
    quantile: Callable[[Any], Any],
    a: int,
    b: int,
    order: int,
) -> float:
    """Integrate one beta order statistic after regularizing both endpoints.

    Directly integrating ``Q(BetaPPF(t))`` can leave an infinite derivative at
    ``t=1`` for heavy-tailed score families.  On each half interval we use a
    fourth-power map.  Its Jacobian makes the transformed integrand smooth and
    bounded even when the underlying score quantile has a singular endpoint.
    """

    x, weights = _unit_legendre_rule(order)
    x_fourth = x**4
    lower_t = 0.5 * x_fourth
    upper_t = 1.0 - 0.5 * x_fourth
    beta_probabilities = special.betaincinv(
        a,
        b,
        np.concatenate((lower_t, upper_t)),
    )
    values = _evaluate_quantile(quantile, beta_probabilities)
    jacobian = 2.0 * x**3
    half = x.size
    return float(np.dot(weights * jacobian, values[:half] + values[half:]))


def _rank_integral_precision(
    quantile: Callable[[Any], Any],
    K: int,
    B: int,
    *,
    epsabs: float,
    epsrel: float,
    quadrature_order: int,
) -> float:
    """Integrate each of the top ``B`` beta order-statistic laws.

    This path is particularly stable when ``B`` is small compared with a very
    large ``K``: transforming through the beta CDF prevents adaptive
    quadrature from overlooking a narrow boundary layer near one.
    """

    total = 0.0
    per_rank_abs = max(5e-15, epsabs / B)
    for offset in range(B):
        a = K - offset
        b = offset + 1

        previous = _regularized_rank_expectation(quantile, a, b, quadrature_order)
        expectation = previous
        for order in (2 * quadrature_order, 4 * quadrature_order, 8 * quadrature_order):
            expectation = _regularized_rank_expectation(quantile, a, b, order)
            tolerance = max(per_rank_abs, epsrel * abs(expectation))
            # Comparing two independent quadrature orders validates the
            # returned value rather than relying on QUADPACK's extrapolation
            # error estimate, which is precisely what becomes unreliable at
            # heavy-tail endpoints.
            if abs(expectation - previous) <= tolerance:
                break
            previous = expectation
        else:
            # A non-smooth custom quantile may converge more slowly than the
            # calibrated channels.  Retry with adaptive quadrature on the same
            # endpoint-regularized integrand, then require agreement with the
            # high-order deterministic rule before accepting it.
            def endpoint_regularized_integrand(x: float) -> float:
                if x == 0.0:
                    return 0.0
                x_fourth = x**4
                probabilities = special.betaincinv(
                    a,
                    b,
                    np.array((0.5 * x_fourth, 1.0 - 0.5 * x_fourth)),
                )
                values = _evaluate_quantile(quantile, probabilities)
                return float(2.0 * x**3 * np.sum(values))

            adaptive, error = integrate.quad(
                endpoint_regularized_integrand,
                0.0,
                1.0,
                epsabs=per_rank_abs,
                epsrel=epsrel,
                limit=200,
            )
            tolerance = max(per_rank_abs, epsrel * abs(adaptive))
            if error > tolerance or abs(adaptive - expectation) > 4.0 * tolerance:
                raise ArithmeticError(
                    "top-B rank quadrature failed independent convergence validation"
                )
            expectation = float(adaptive)
        total += expectation
    return float(total / B)


def _selection_weight_integral_precision(
    quantile: Callable[[Any], Any],
    K: int,
    B: int,
    *,
    epsabs: float,
    epsrel: float,
) -> float:
    """Integrate the exact normalized top-``B`` selection weight."""

    scale = K / B

    def integrand(u: float) -> float:
        selected_probability = float(special.betainc(K - B, B, u))
        return scale * selected_probability * float(_evaluate_quantile(quantile, u))

    fraction = B / K
    transition = 1.0 - fraction
    transition_sd = math.sqrt(max(fraction * (1.0 - fraction) / K, 0.0))
    split_points = {0.0, 1.0, transition}
    for multiplier in (2.0, 5.0, 10.0, 18.0):
        split_points.add(min(1.0, max(0.0, transition - multiplier * transition_sd)))
        split_points.add(min(1.0, max(0.0, transition + multiplier * transition_sd)))
    points = sorted(split_points)

    value = 0.0
    interval_abs = max(5e-15, epsabs / max(1, len(points) - 1))
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        part, _ = integrate.quad(
            integrand,
            left,
            right,
            epsabs=interval_abs,
            epsrel=epsrel,
            limit=180,
        )
        value += part
    return float(value)


def finite_k_top_b_precision(
    channel: PosteriorChannel,
    K: int,
    B: int,
    *,
    epsabs: float = 2e-11,
    epsrel: float = 2e-10,
    quadrature_order: int = 32,
) -> float:
    """Compute deterministic expected top-``B`` precision for ``K`` records.

    Discrete posterior channels are evaluated exactly up to floating-point
    binomial CDF error, including ties at the selection boundary.  Continuous
    channels are evaluated by deterministic quadrature of exact order-
    statistic weights; no asymptotic approximation is used.
    """

    population, budget = _validate_k_b(K, B)
    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("quadrature tolerances must be positive")
    base_order = _positive_integer(quadrature_order, name="quadrature_order")
    if base_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    atoms = _discrete_atoms(channel)
    if atoms is not None:
        return _finite_discrete_precision(*atoms, population, budget)

    quantile = _quantile_callable(channel)
    if quantile is None:
        raise TypeError(
            "channel must expose posterior_quantile/quantile or discrete posterior atoms"
        )

    # Selecting every record has precision E[eta]=p for a calibrated channel.
    # Returning the declared prior avoids unnecessary endpoint quadrature.
    if budget == population and hasattr(channel, "p"):
        prior = float(getattr(channel, "p"))
        if math.isfinite(prior) and 0.0 <= prior <= 1.0:
            return prior

    if budget == population:
        value, _ = integrate.quad(
            lambda u: float(_evaluate_quantile(quantile, u)),
            0.0,
            1.0,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=180,
        )
    elif budget <= 48:
        value = _rank_integral_precision(
            quantile,
            population,
            budget,
            epsabs=epsabs,
            epsrel=epsrel,
            quadrature_order=base_order,
        )
    else:
        value = _selection_weight_integral_precision(
            quantile,
            population,
            budget,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    if value < -5e-10 or value > 1.0 + 5e-10:
        raise ArithmeticError("deterministic top-B precision left the probability range")
    return min(1.0, max(0.0, float(value)))


def _discrete_top_tail_precision(
    values: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
) -> float:
    remaining = alpha
    selected_mass = 0.0
    for value, mass in zip(values[::-1], probabilities[::-1]):
        take = min(remaining, float(mass))
        selected_mass += take * float(value)
        remaining -= take
        if remaining <= 2e-15:
            break
    if remaining > 2e-12:
        raise ArithmeticError("discrete tail integration did not collect alpha mass")
    return float(selected_mass / alpha)


def large_k_top_tail_precision(
    channel: PosteriorChannel,
    alpha: float,
    *,
    epsabs: float = 2e-11,
    epsrel: float = 2e-10,
) -> float:
    """Evaluate the large-``K`` top-tail limit by its quantile integral.

    ``q_alpha = alpha**(-1) integral_{1-alpha}^1 Q_eta(u) du``.

    Discrete atoms are integrated exactly, including a fractional boundary
    atom.  If a channel has no exposed quantile but provides its own
    ``top_tail_precision(alpha)`` method, that method is used as a protocol
    fallback.
    """

    fraction = _validate_alpha(alpha)
    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("quadrature tolerances must be positive")

    atoms = _discrete_atoms(channel)
    if atoms is not None:
        return _discrete_top_tail_precision(*atoms, fraction)

    quantile = _quantile_callable(channel)
    if quantile is None:
        method = getattr(channel, "top_tail_precision", None)
        if not callable(method):
            raise TypeError("channel must expose a posterior quantile or top_tail_precision")
        value = float(method(fraction))
    elif fraction == 1.0 and hasattr(channel, "p"):
        value = float(getattr(channel, "p"))
    else:
        integral, _ = integrate.quad(
            lambda u: float(_evaluate_quantile(quantile, u)),
            1.0 - fraction,
            1.0,
            epsabs=max(5e-15, epsabs * fraction),
            epsrel=epsrel,
            limit=220,
        )
        value = integral / fraction

    if not math.isfinite(value) or value < -5e-10 or value > 1.0 + 5e-10:
        raise ArithmeticError("large-K tail precision is not a probability")
    return min(1.0, max(0.0, float(value)))


def relative_boost_error(q_k_b: float, q_alpha: float, p: float) -> float:
    """Return ``|q_{K,B}-q_alpha| / (q_alpha-p)``.

    A channel with no asymptotic boost has a zero denominator.  In that
    degenerate case the function returns zero if the finite value agrees and
    infinity otherwise.
    """

    finite_precision = float(q_k_b)
    limit_precision = float(q_alpha)
    prior = float(p)
    if not all(math.isfinite(value) for value in (finite_precision, limit_precision, prior)):
        raise ValueError("precisions and p must be finite")
    if not 0.0 <= prior <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if not 0.0 <= finite_precision <= 1.0 or not 0.0 <= limit_precision <= 1.0:
        raise ValueError("precisions must lie in [0, 1]")
    boost = limit_precision - prior
    if boost < -2e-12:
        raise ValueError("q_alpha must not be below the prior p")
    discrepancy = abs(finite_precision - limit_precision)
    if boost <= 2e-15:
        return 0.0 if discrepancy <= 2e-15 else math.inf
    return float(discrepancy / boost)


def top_b_precision_from_samples(samples: Any, B: int, *, axis: int = -1) -> Any:
    """Compute top-``B`` posterior precision using ``numpy.argpartition``."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("samples must have at least one dimension")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("posterior samples must be finite and lie in [0, 1]")
    integer_axis = int(axis)
    if integer_axis != axis or not -values.ndim <= integer_axis < values.ndim:
        raise ValueError("axis is out of bounds for the sample array")
    normalized_axis = integer_axis % values.ndim
    population = values.shape[normalized_axis]
    _, budget = _validate_k_b(population, B)
    if budget == population:
        result = np.mean(values, axis=normalized_axis, dtype=np.float64)
    else:
        partition = np.argpartition(values, kth=population - budget, axis=normalized_axis)
        top_indices = np.take(
            partition,
            indices=np.arange(population - budget, population),
            axis=normalized_axis,
        )
        top_values = np.take_along_axis(values, top_indices, axis=normalized_axis)
        result = np.mean(top_values, axis=normalized_axis, dtype=np.float64)
    return float(result) if np.ndim(result) == 0 else result


def _call_sampler(channel: object, n: int, rng: np.random.Generator) -> np.ndarray:
    sampler: Optional[Callable[..., Any]] = None
    for name in ("sample_eta", "sample_posterior", "sample"):
        candidate = getattr(channel, name, None)
        if callable(candidate):
            sampler = candidate
            break
    if sampler is None:
        raise TypeError("channel must expose sample_eta, sample_posterior, or sample")

    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(sampler).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if "rng" in parameters or accepts_kwargs:
        kwargs["rng"] = rng
    elif "seed" in parameters:
        kwargs["seed"] = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
    raw = sampler(n, **kwargs)

    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError("tuple-valued channel samples must have length two")
        scorer = getattr(channel, "score", None)
        if callable(scorer):
            raw = scorer(raw[1])
        else:
            # Compatibility for channels returning (latent, posterior).
            raw = raw[1]
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != n:
        raise ValueError(f"channel sampler returned {values.size} values; expected {n}")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("channel sampler must return posterior probabilities in [0, 1]")
    return values


def monte_carlo_top_b_precision(
    channel: PosteriorChannel,
    K: int,
    B: int,
    *,
    n_trials: int = 2_000,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> Tuple[float, float]:
    """Estimate finite-``K`` top-``B`` precision and its standard error."""

    population, budget = _validate_k_b(K, B)
    trials = _positive_integer(n_trials, name="n_trials")
    if batch_size is None:
        batch = min(trials, max(1, 1_000_000 // population))
    else:
        batch = min(trials, _positive_integer(batch_size, name="batch_size"))

    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    estimates = np.empty(trials, dtype=np.float64)
    completed = 0
    while completed < trials:
        current = min(batch, trials - completed)
        samples = _call_sampler(channel, current * population, rng).reshape(current, population)
        estimates[completed : completed + current] = top_b_precision_from_samples(
            samples, budget, axis=1
        )
        completed += current

    mean = float(estimates.mean(dtype=np.float64))
    standard_error = (
        float(estimates.std(ddof=1, dtype=np.float64) / math.sqrt(trials))
        if trials > 1
        else 0.0
    )
    return mean, standard_error


def finite_k_top_b_precision_monte_carlo(
    channel: PosteriorChannel,
    K: int,
    B: int,
    **kwargs: Any,
) -> Tuple[float, float]:
    """Compatibility alias for :func:`monte_carlo_top_b_precision`."""

    return monte_carlo_top_b_precision(channel, K, B, **kwargs)


# A concise alias used by experiment tables that label the large-K quantity
# as a limit rather than a precision.
large_k_top_tail_limit = large_k_top_tail_precision


__all__ = [
    "PosteriorChannel",
    "finite_k_top_b_precision",
    "large_k_top_tail_precision",
    "large_k_top_tail_limit",
    "relative_boost_error",
    "top_b_precision_from_samples",
    "monte_carlo_top_b_precision",
    "finite_k_top_b_precision_monte_carlo",
]
