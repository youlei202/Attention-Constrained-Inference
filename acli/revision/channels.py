"""Exactly calibrated posterior-score channels.

The channel objects in this module describe the *marginal* distribution of a
posterior score ``eta = P(T=1 | score)``.  Consequently calibration is simply
``E[eta] = p``, mutual information is ``E[d2(eta || p)]``, and labels can be
sampled conditionally as ``T | eta ~ Bernoulli(eta)``.

Continuous families use the construction

``eta_epsilon(g) = expit(intercept + epsilon * g)``

with a numerically solved intercept.  The supplied base distributions are all
centered and variance-normalized so ``epsilon`` has a comparable local meaning
across families.  Discrete channels implement the same public protocol and
account for score ties exactly when computing AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm, t as student_t

from .information import binary_kl_bits, h2


CONTINUOUS_FAMILIES = ("gaussian", "student_t5", "pareto4", "pareto5", "uniform")
DISCRETE_FAMILIES = ("rare_spike", "three_point", "two_point_frontier")
SUPPORTED_FAMILIES = CONTINUOUS_FAMILIES + DISCRETE_FAMILIES

_QUAD_EPSABS = 5e-13
_QUAD_EPSREL = 5e-12
_CALIBRATION_TOL = 1e-10


def _validate_prior(p: float) -> float:
    prior = float(p)
    if not np.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("p must be finite and strictly between zero and one")
    return prior


def _validate_epsilon(epsilon: float) -> float:
    strength = float(epsilon)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    return strength


def _return_scalar_if_scalar(result: np.ndarray, value: object) -> Union[float, np.ndarray]:
    if np.ndim(value) == 0:
        return float(np.asarray(result))
    return result


def _validate_unit_interval(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _base_ppf(family: str, probability: object) -> np.ndarray:
    u = np.asarray(probability, dtype=np.float64)
    if family == "gaussian":
        return np.asarray(norm.ppf(u), dtype=np.float64)
    if family == "student_t5":
        return np.asarray(student_t.ppf(u, df=5) * np.sqrt(3.0 / 5.0), dtype=np.float64)
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(shape / ((shape - 1.0) ** 2 * (shape - 2.0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.power(1.0 - u, -1.0 / shape)
        return np.asarray((raw - mean) / standard_deviation, dtype=np.float64)
    if family == "uniform":
        return np.asarray(np.sqrt(3.0) * (2.0 * u - 1.0), dtype=np.float64)
    raise ValueError(f"unknown continuous family {family!r}")


def _base_cdf(family: str, score: float) -> float:
    x = float(score)
    if family == "gaussian":
        return float(norm.cdf(x))
    if family == "student_t5":
        return float(student_t.cdf(x / np.sqrt(3.0 / 5.0), df=5))
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(shape / ((shape - 1.0) ** 2 * (shape - 2.0)))
        raw = mean + standard_deviation * x
        if raw <= 1.0:
            return 0.0
        return float(1.0 - raw ** (-shape))
    if family == "uniform":
        lower = -np.sqrt(3.0)
        upper = np.sqrt(3.0)
        return float(np.clip((x - lower) / (upper - lower), 0.0, 1.0))
    raise ValueError(f"unknown continuous family {family!r}")


def _base_pdf(family: str, score: float) -> float:
    x = float(score)
    if family == "gaussian":
        return float(norm.pdf(x))
    if family == "student_t5":
        scale = np.sqrt(3.0 / 5.0)
        return float(student_t.pdf(x / scale, df=5) / scale)
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(shape / ((shape - 1.0) ** 2 * (shape - 2.0)))
        raw = mean + standard_deviation * x
        if raw < 1.0:
            return 0.0
        return float(shape * raw ** (-shape - 1.0) * standard_deviation)
    if family == "uniform":
        if -np.sqrt(3.0) <= x <= np.sqrt(3.0):
            return float(1.0 / (2.0 * np.sqrt(3.0)))
        return 0.0
    raise ValueError(f"unknown continuous family {family!r}")


def _base_support(family: str) -> Tuple[float, float]:
    if family in ("gaussian", "student_t5"):
        return -np.inf, np.inf
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(shape / ((shape - 1.0) ** 2 * (shape - 2.0)))
        return float((1.0 - mean) / standard_deviation), np.inf
    if family == "uniform":
        return -np.sqrt(3.0), np.sqrt(3.0)
    raise ValueError(f"unknown continuous family {family!r}")


def _sample_base(family: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        return np.asarray(rng.standard_normal(n), dtype=np.float64)
    if family == "student_t5":
        return np.asarray(rng.standard_t(5, n) * np.sqrt(3.0 / 5.0), dtype=np.float64)
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(shape / ((shape - 1.0) ** 2 * (shape - 2.0)))
        raw = rng.pareto(shape, n) + 1.0
        return np.asarray((raw - mean) / standard_deviation, dtype=np.float64)
    if family == "uniform":
        return np.asarray(rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n), dtype=np.float64)
    raise ValueError(f"unknown continuous family {family!r}")


def _rng_from_arguments(
    rng: Optional[np.random.Generator], seed: Optional[int]
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _quad_unit_interval(
    function: Callable[[float], float],
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    points: Sequence[float] = (),
) -> float:
    """High-accuracy deterministic integration under a quantile transform."""

    if lower == upper:
        return 0.0
    boundaries = [float(lower)]
    boundaries.extend(sorted({float(x) for x in points if lower < x < upper}))
    boundaries.append(float(upper))
    total = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        value, _ = quad(
            function,
            left,
            right,
            epsabs=max(_QUAD_EPSABS * (right - left), 5e-15),
            epsrel=_QUAD_EPSREL,
            limit=300,
        )
        total += value
    return float(total)


def _transition_probability(family: str, intercept: float, epsilon: float) -> float:
    if epsilon == 0.0:
        return 0.5
    return _base_cdf(family, -intercept / epsilon)


def _continuous_expectation(
    family: str,
    intercept: float,
    epsilon: float,
    transform: Callable[[float], float] = lambda eta: eta,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    if epsilon == 0.0:
        return float((upper - lower) * transform(float(expit(intercept))))

    # In Pareto score coordinates the polynomial tail is awkward for QUADPACK
    # at some intermediate root-solver iterates.  With raw Pareto X=exp(t),
    # the measure is simply shape*exp(-shape*t) dt, giving an exponentially
    # decaying and warning-free integral without changing the mathematics.
    if family in ("pareto4", "pareto5"):
        shape = float(family[-1])
        mean = shape / (shape - 1.0)
        standard_deviation = np.sqrt(
            shape / ((shape - 1.0) ** 2 * (shape - 2.0))
        )
        t_lower = 0.0 if lower == 0.0 else -np.log1p(-lower) / shape
        t_upper = np.inf if upper == 1.0 else -np.log1p(-upper) / shape

        def pareto_integrand(log_raw_score: float) -> float:
            if log_raw_score >= 700.0:
                posterior = 1.0
            else:
                base_score = (np.exp(log_raw_score) - mean) / standard_deviation
                posterior = float(expit(intercept + epsilon * base_score))
            return float(transform(posterior)) * shape * np.exp(-shape * log_raw_score)

        split_points = []
        for base_score in (0.0, -intercept / epsilon):
            raw_score = mean + standard_deviation * base_score
            if raw_score > 1.0:
                log_raw_score = float(np.log(raw_score))
                if t_lower < log_raw_score < t_upper:
                    split_points.append(log_raw_score)
        boundaries = [t_lower, *sorted(set(split_points)), t_upper]
        total = 0.0
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            value, _ = quad(
                pareto_integrand,
                left,
                right,
                epsabs=_QUAD_EPSABS,
                epsrel=_QUAD_EPSREL,
                limit=300,
            )
            total += value
        return float(total)

    support_lower, support_upper = _base_support(family)
    score_lower = support_lower if lower == 0.0 else float(_base_ppf(family, lower))
    score_upper = support_upper if upper == 1.0 else float(_base_ppf(family, upper))

    def integrand(base_score: float) -> float:
        posterior = float(expit(intercept + epsilon * base_score))
        return float(transform(posterior)) * _base_pdf(family, base_score)

    split_points = []
    transition_score = -intercept / epsilon
    if score_lower < transition_score < score_upper:
        split_points.append(transition_score)
    if score_lower < 0.0 < score_upper:
        split_points.append(0.0)
    boundaries = [score_lower, *sorted(set(split_points)), score_upper]
    total = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        value, _ = quad(
            integrand,
            left,
            right,
            epsabs=_QUAD_EPSABS,
            epsrel=_QUAD_EPSREL,
            limit=300,
        )
        total += value
    return float(total)


@lru_cache(maxsize=4096)
def _calibrate_logistic_intercept_cached(p: float, epsilon: float, family: str) -> float:
    if family not in CONTINUOUS_FAMILIES:
        raise ValueError(f"unknown continuous family {family!r}")
    if epsilon == 0.0:
        return float(logit(p))

    def residual(intercept: float) -> float:
        return _continuous_expectation(family, intercept, epsilon) - p

    center = float(logit(p))
    width = max(4.0, 2.0 * epsilon)
    lower = center - width
    upper = center + width
    for _ in range(80):
        if residual(lower) <= 0.0:
            break
        width *= 2.0
        lower = center - width
    else:  # pragma: no cover - defensive failure for pathological platforms
        raise RuntimeError("could not bracket calibrated intercept from below")
    for _ in range(80):
        if residual(upper) >= 0.0:
            break
        width *= 2.0
        upper = center + width
    else:  # pragma: no cover
        raise RuntimeError("could not bracket calibrated intercept from above")

    intercept = float(
        brentq(residual, lower, upper, xtol=5e-14, rtol=4.0 * np.finfo(float).eps)
    )
    error = abs(residual(intercept))
    if error > _CALIBRATION_TOL:
        raise RuntimeError(f"posterior calibration failed: absolute error {error:.3e}")
    return intercept


def calibrate_logistic_intercept(p: float, epsilon: float, family: str = "gaussian") -> float:
    """Solve ``E[expit(a + epsilon G)] = p`` for the intercept ``a``."""

    prior = _validate_prior(p)
    strength = _validate_epsilon(epsilon)
    if family not in CONTINUOUS_FAMILIES:
        raise ValueError(f"family must be one of {CONTINUOUS_FAMILIES}")
    return _calibrate_logistic_intercept_cached(prior, strength, family)


@dataclass(frozen=True)
class ContinuousPosteriorChannel:
    """A calibrated logistic-posterior channel with a continuous base score."""

    p: float
    epsilon: float
    intercept: float
    family: str = "gaussian"

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _validate_prior(self.p))
        object.__setattr__(self, "epsilon", _validate_epsilon(self.epsilon))
        if self.family not in CONTINUOUS_FAMILIES:
            raise ValueError(f"family must be one of {CONTINUOUS_FAMILIES}")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        if self.calibration_error > _CALIBRATION_TOL:
            raise ValueError(
                "intercept is not calibrated: "
                f"abs(E[eta]-p)={self.calibration_error:.3e}"
            )

    def posterior(self, base_score: object) -> Union[float, np.ndarray]:
        """Evaluate the posterior at one or more base-score values."""

        score = np.asarray(base_score, dtype=np.float64)
        values = np.asarray(expit(self.intercept + self.epsilon * score), dtype=np.float64)
        return _return_scalar_if_scalar(values, base_score)

    score = posterior

    @property
    def mean_posterior(self) -> float:
        return _continuous_expectation(self.family, self.intercept, self.epsilon)

    @property
    def mean_eta(self) -> float:
        return self.mean_posterior

    @property
    def calibration_error(self) -> float:
        return abs(self.mean_posterior - self.p)

    def sample_eta(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw ``n`` posterior scores from the marginal score population."""

        size = int(n)
        if size < 0 or size != n:
            raise ValueError("n must be a nonnegative integer")
        generator = _rng_from_arguments(rng, seed)
        return np.asarray(self.posterior(_sample_base(self.family, size, generator)))

    sample_posterior = sample_eta

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw calibrated posterior scores and their Bernoulli labels."""

        generator = _rng_from_arguments(rng, seed)
        eta = self.sample_eta(n, rng=generator)
        labels = generator.binomial(1, eta).astype(np.int8)
        return labels, eta

    def posterior_quantile(self, probability: object) -> Union[float, np.ndarray]:
        u = _validate_unit_interval(probability, name="probability")
        if self.epsilon == 0.0:
            values = np.full_like(u, self.p, dtype=np.float64)
        else:
            values = np.asarray(self.posterior(_base_ppf(self.family, u)), dtype=np.float64)
        return _return_scalar_if_scalar(values, probability)

    quantile = posterior_quantile
    posterior_ppf = posterior_quantile

    def base_quantile(self, probability: object) -> Union[float, np.ndarray]:
        """Quantile of the centered, variance-one base score ``G``."""

        u = _validate_unit_interval(probability, name="probability")
        values = _base_ppf(self.family, u)
        return _return_scalar_if_scalar(values, probability)

    def base_top_tail_mean(self, alpha: float) -> float:
        """Return ``E[G | G is in its upper alpha tail]``."""

        fraction = float(alpha)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if fraction == 1.0:
            return 0.0
        threshold = float(self.base_quantile(1.0 - fraction))
        if self.family == "gaussian":
            return float(norm.pdf(threshold) / fraction)
        if self.family == "student_t5":
            scale = np.sqrt(3.0 / 5.0)
            raw_threshold = threshold / scale
            truncated_first_moment = (
                (5.0 + raw_threshold * raw_threshold)
                / 4.0
                * student_t.pdf(raw_threshold, df=5)
            )
            return float(scale * truncated_first_moment / fraction)
        if self.family in ("pareto4", "pareto5"):
            shape = float(self.family[-1])
            mean = shape / (shape - 1.0)
            standard_deviation = np.sqrt(
                shape / ((shape - 1.0) ** 2 * (shape - 2.0))
            )
            raw_threshold = mean + standard_deviation * threshold
            conditional_raw_mean = shape * raw_threshold / (shape - 1.0)
            return float((conditional_raw_mean - mean) / standard_deviation)
        if self.family == "uniform":
            return float((threshold + np.sqrt(3.0)) / 2.0)
        raise AssertionError("unreachable family")

    def top_tail_precision(self, alpha: float) -> float:
        """Mean posterior among the largest ``alpha`` fraction of scores."""

        fraction = float(alpha)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if fraction == 1.0:
            return self.mean_posterior
        value = _continuous_expectation(
            self.family,
            self.intercept,
            self.epsilon,
            lower=1.0 - fraction,
            upper=1.0,
        )
        return float(value / fraction)

    top_tail_mean = top_tail_precision

    def mutual_information_bits(self) -> float:
        value = _continuous_expectation(
            self.family,
            self.intercept,
            self.epsilon,
            transform=lambda eta: float(binary_kl_bits(eta, self.p)),
        )
        return float(np.clip(value, 0.0, float(h2(self.p))))

    information_bits = mutual_information_bits

    @property
    def J_bits(self) -> float:
        return self.mutual_information_bits()

    def auc(self) -> float:
        """Exact deterministic AUC of the continuous posterior score."""

        if self.epsilon == 0.0:
            return 0.5

        def integrand(base_score: float) -> float:
            eta = float(self.posterior(base_score))
            rank = 2.0 * _base_cdf(self.family, base_score) - 1.0
            return rank * eta * _base_pdf(self.family, base_score)

        lower, upper = _base_support(self.family)
        transition = -self.intercept / self.epsilon
        boundaries = [lower]
        boundaries.extend(
            sorted(x for x in {0.0, transition} if lower < x < upper)
        )
        boundaries.append(upper)
        rank_covariance = 0.0
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            value, _ = quad(
                integrand,
                left,
                right,
                epsabs=_QUAD_EPSABS,
                epsrel=_QUAD_EPSREL,
                limit=300,
            )
            rank_covariance += value
        value = 0.5 + rank_covariance / (2.0 * self.p * (1.0 - self.p))
        return float(np.clip(value, 0.5, 1.0))

    realized_auc = auc

    def configuration(self) -> Mapping[str, Any]:
        return {
            "family": self.family,
            "p": self.p,
            "epsilon": self.epsilon,
            "intercept": self.intercept,
        }


def make_continuous_channel(
    family: str,
    p: float,
    epsilon: float,
) -> ContinuousPosteriorChannel:
    """Construct an exactly calibrated continuous posterior-score channel."""

    intercept = calibrate_logistic_intercept(p, epsilon, family)
    return ContinuousPosteriorChannel(
        p=float(p), epsilon=float(epsilon), intercept=intercept, family=family
    )


def _prepare_discrete_distribution(
    values: object, probabilities: object
) -> Tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(values, dtype=np.float64)
    masses = np.asarray(probabilities, dtype=np.float64)
    if eta.ndim != 1 or masses.ndim != 1 or eta.size == 0 or eta.shape != masses.shape:
        raise ValueError("posterior_values and probabilities must be equal-length 1D arrays")
    if not np.all(np.isfinite(eta)) or np.any((eta < 0.0) | (eta > 1.0)):
        raise ValueError("posterior_values must lie in [0, 1]")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(masses.sum())
    if total <= 0.0 or abs(total - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    masses = masses / total
    positive = masses > 0.0
    eta = eta[positive]
    masses = masses[positive]

    order = np.argsort(eta, kind="stable")
    eta = eta[order]
    masses = masses[order]
    unique, inverse = np.unique(eta, return_inverse=True)
    combined = np.zeros(unique.size, dtype=np.float64)
    np.add.at(combined, inverse, masses)
    unique.setflags(write=False)
    combined.setflags(write=False)
    return unique, combined


@dataclass(frozen=True)
class DiscretePosteriorChannel:
    """A finite calibrated posterior-score distribution."""

    p: float
    posterior_values: np.ndarray
    probabilities: np.ndarray
    family: str = "discrete"
    epsilon: Optional[float] = None
    intercept: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _validate_prior(self.p))
        values, masses = _prepare_discrete_distribution(
            self.posterior_values, self.probabilities
        )
        object.__setattr__(self, "posterior_values", values)
        object.__setattr__(self, "probabilities", masses)
        if self.epsilon is not None:
            object.__setattr__(self, "epsilon", _validate_epsilon(self.epsilon))
        if self.intercept is not None and not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite when provided")
        if self.calibration_error > _CALIBRATION_TOL:
            raise ValueError(
                "discrete posterior is not calibrated: "
                f"abs(E[eta]-p)={self.calibration_error:.3e}"
            )

    @property
    def values(self) -> np.ndarray:
        return self.posterior_values

    @property
    def mean_posterior(self) -> float:
        return float(np.dot(self.probabilities, self.posterior_values))

    @property
    def mean_eta(self) -> float:
        return self.mean_posterior

    @property
    def calibration_error(self) -> float:
        return abs(self.mean_posterior - self.p)

    def sample_eta(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        size = int(n)
        if size < 0 or size != n:
            raise ValueError("n must be a nonnegative integer")
        generator = _rng_from_arguments(rng, seed)
        indices = generator.choice(self.posterior_values.size, size=size, p=self.probabilities)
        return self.posterior_values[indices]

    sample_posterior = sample_eta

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        generator = _rng_from_arguments(rng, seed)
        eta = self.sample_eta(n, rng=generator)
        labels = generator.binomial(1, eta).astype(np.int8)
        return labels, eta

    def posterior_quantile(self, probability: object) -> Union[float, np.ndarray]:
        u = _validate_unit_interval(probability, name="probability")
        cumulative = np.cumsum(self.probabilities)
        indices = np.searchsorted(cumulative, u, side="left")
        indices = np.clip(indices, 0, self.posterior_values.size - 1)
        values = self.posterior_values[indices]
        return _return_scalar_if_scalar(np.asarray(values), probability)

    quantile = posterior_quantile
    posterior_ppf = posterior_quantile

    def top_tail_precision(self, alpha: float) -> float:
        """Top-tail mean, fractionally allocating an atom at the cutoff."""

        fraction = float(alpha)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        remaining = fraction
        numerator = 0.0
        for eta, mass in zip(self.posterior_values[::-1], self.probabilities[::-1]):
            selected = min(remaining, float(mass))
            numerator += selected * float(eta)
            remaining -= selected
            if remaining <= 5e-16:
                break
        return float(numerator / fraction)

    top_tail_mean = top_tail_precision

    def mutual_information_bits(self) -> float:
        divergence = np.asarray(binary_kl_bits(self.posterior_values, self.p))
        return float(np.dot(self.probabilities, divergence))

    information_bits = mutual_information_bits

    @property
    def J_bits(self) -> float:
        return self.mutual_information_bits()

    def auc(self) -> float:
        """AUC with exactly one-half credit for every discrete score tie."""

        conditional_one = self.probabilities * self.posterior_values / self.p
        conditional_zero = self.probabilities * (1.0 - self.posterior_values) / (1.0 - self.p)
        strictly_greater = float(
            np.sum(np.tril(np.outer(conditional_one, conditional_zero), k=-1))
        )
        ties = float(np.dot(conditional_one, conditional_zero))
        return float(np.clip(strictly_greater + 0.5 * ties, 0.5, 1.0))

    realized_auc = auc

    def configuration(self) -> Mapping[str, Any]:
        return {
            "family": self.family,
            "p": self.p,
            "posterior_values": self.posterior_values.tolist(),
            "probabilities": self.probabilities.tolist(),
            "epsilon": self.epsilon,
            "intercept": self.intercept,
        }


def _standardize_discrete_scores(scores: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    mean = float(np.dot(scores, probabilities))
    centered = scores - mean
    variance = float(np.dot(centered * centered, probabilities))
    if variance <= 0.0:
        raise ValueError("discrete base scores must have positive variance")
    return centered / np.sqrt(variance)


def _calibrate_discrete_intercept(
    p: float, epsilon: float, scores: np.ndarray, probabilities: np.ndarray
) -> float:
    if epsilon == 0.0:
        return float(logit(p))

    def residual(intercept: float) -> float:
        return float(np.dot(probabilities, expit(intercept + epsilon * scores)) - p)

    center = float(logit(p))
    width = max(4.0, 2.0 * epsilon * float(np.max(np.abs(scores))))
    lower = center - width
    upper = center + width
    while residual(lower) > 0.0:
        width *= 2.0
        lower = center - width
    while residual(upper) < 0.0:
        width *= 2.0
        upper = center + width
    return float(brentq(residual, lower, upper, xtol=5e-14, rtol=4 * np.finfo(float).eps))


def calibrated_discrete_logistic_channel(
    p: float,
    epsilon: float,
    scores: Sequence[float],
    probabilities: Sequence[float],
    *,
    family: str = "discrete_logistic",
    standardize: bool = True,
) -> DiscretePosteriorChannel:
    """Create a calibrated logistic channel on a finite base-score support."""

    prior = _validate_prior(p)
    strength = _validate_epsilon(epsilon)
    score_array = np.asarray(scores, dtype=np.float64)
    masses = np.asarray(probabilities, dtype=np.float64)
    if score_array.ndim != 1 or masses.ndim != 1 or score_array.shape != masses.shape:
        raise ValueError("scores and probabilities must be equal-length 1D arrays")
    if score_array.size < 2 or not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain at least two finite values")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(masses.sum())
    if total <= 0.0 or abs(total - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    masses = masses / total
    if standardize:
        score_array = _standardize_discrete_scores(score_array, masses)
    intercept = _calibrate_discrete_intercept(prior, strength, score_array, masses)
    posterior = np.asarray(expit(intercept + strength * score_array), dtype=np.float64)
    return DiscretePosteriorChannel(
        p=prior,
        posterior_values=posterior,
        probabilities=masses,
        family=family,
        epsilon=strength,
        intercept=intercept,
    )


def rare_spike_channel(
    p: float,
    epsilon: float,
    *,
    spike_probability: float = 0.01,
) -> DiscretePosteriorChannel:
    """A two-level discrete base score with a rare high-score atom."""

    spike = float(spike_probability)
    if not np.isfinite(spike) or not 0.0 < spike < 1.0:
        raise ValueError("spike_probability must lie in (0, 1)")
    return calibrated_discrete_logistic_channel(
        p,
        epsilon,
        scores=(0.0, 1.0),
        probabilities=(1.0 - spike, spike),
        family="rare_spike",
    )


def three_point_channel(
    p: float,
    epsilon: float,
    *,
    probabilities: Sequence[float] = (0.45, 0.50, 0.05),
    scores: Sequence[float] = (-1.0, 0.0, 3.0),
) -> DiscretePosteriorChannel:
    """A calibrated asymmetric three-level posterior-score channel."""

    return calibrated_discrete_logistic_channel(
        p,
        epsilon,
        scores=scores,
        probabilities=probabilities,
        family="three_point",
    )


def two_point_frontier_channel(
    p: float,
    alpha: float,
    *,
    q: Optional[float] = None,
    J_bits: Optional[float] = None,
) -> DiscretePosteriorChannel:
    """Two-posterior channel attaining the binary-selection frontier.

    Exactly one of ``q`` (the posterior on the selected ``alpha`` branch) or
    ``J_bits`` (the desired selection information) must be supplied.
    """

    prior = _validate_prior(p)
    fraction = float(alpha)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if (q is None) == (J_bits is None):
        raise ValueError("provide exactly one of q or J_bits")
    upper = min(1.0, prior / fraction)

    def q0(selected: float) -> float:
        return (prior - fraction * selected) / (1.0 - fraction)

    def information(selected: float) -> float:
        return float(
            fraction * binary_kl_bits(selected, prior)
            + (1.0 - fraction) * binary_kl_bits(q0(selected), prior)
        )

    if J_bits is not None:
        target = float(J_bits)
        ceiling = information(upper)
        if not np.isfinite(target) or target < 0.0 or target > ceiling + 2e-12:
            raise ValueError(f"J_bits must lie in [0, {ceiling}]")
        if target <= 1e-16:
            selected = prior
        elif ceiling - target <= 2e-14:
            selected = upper
        else:
            selected = float(
                brentq(lambda value: information(value) - target, prior, upper, xtol=5e-14)
            )
    else:
        selected = float(q)
        if not np.isfinite(selected) or selected < prior or selected > upper:
            raise ValueError(f"q must lie in [{prior}, {upper}]")

    other = q0(selected)
    return DiscretePosteriorChannel(
        p=prior,
        posterior_values=np.array([other, selected]),
        probabilities=np.array([1.0 - fraction, fraction]),
        family="two_point_frontier",
    )


def make_score_channel(
    family: str,
    p: float,
    epsilon: Optional[float] = None,
    **kwargs: Any,
) -> Union[ContinuousPosteriorChannel, DiscretePosteriorChannel]:
    """Unified factory for every supported major-revision score family."""

    if family in CONTINUOUS_FAMILIES:
        if epsilon is None:
            raise ValueError("continuous families require epsilon")
        if kwargs:
            raise TypeError(f"unexpected channel arguments: {sorted(kwargs)}")
        return make_continuous_channel(family, p, epsilon)
    if family == "rare_spike":
        if epsilon is None:
            raise ValueError("rare_spike requires epsilon")
        return rare_spike_channel(p, epsilon, **kwargs)
    if family == "three_point":
        if epsilon is None:
            raise ValueError("three_point requires epsilon")
        return three_point_channel(p, epsilon, **kwargs)
    if family == "two_point_frontier":
        if epsilon is not None:
            raise ValueError("two_point_frontier uses q or J_bits, not epsilon")
        return two_point_frontier_channel(p, **kwargs)
    raise ValueError(f"unknown family {family!r}; expected one of {SUPPORTED_FAMILIES}")


def _solve_continuous_strength(
    family: str,
    p: float,
    target: float,
    statistic: Callable[[ContinuousPosteriorChannel], float],
    baseline: float,
    theoretical_ceiling: float,
) -> float:
    if family not in CONTINUOUS_FAMILIES:
        raise ValueError(f"family must be one of {CONTINUOUS_FAMILIES}")
    prior = _validate_prior(p)
    desired = float(target)
    if not np.isfinite(desired) or desired < baseline or desired >= theoretical_ceiling:
        raise ValueError(
            f"target must lie in [{baseline}, {theoretical_ceiling}) for a finite epsilon"
        )
    if desired == baseline:
        return 0.0

    def residual(epsilon: float) -> float:
        return statistic(make_continuous_channel(family, prior, epsilon)) - desired

    upper = 0.25
    for _ in range(30):
        if residual(upper) >= 0.0:
            break
        upper *= 2.0
    else:
        raise ValueError("target is above the numerically reachable channel ceiling")
    return float(brentq(residual, 0.0, upper, xtol=2e-12, rtol=2e-12))


def _solve_discrete_strength(
    family: str,
    p: float,
    target: float,
    statistic: Callable[[DiscretePosteriorChannel], float],
    baseline: float,
    theoretical_ceiling: float,
    channel_kwargs: Mapping[str, Any],
) -> float:
    if family not in ("rare_spike", "three_point"):
        raise ValueError("epsilon calibration supports rare_spike and three_point")
    prior = _validate_prior(p)
    desired = float(target)
    if not np.isfinite(desired) or desired < baseline or desired >= theoretical_ceiling:
        raise ValueError(
            f"target must lie in [{baseline}, {theoretical_ceiling}) for a finite epsilon"
        )
    if desired == baseline:
        return 0.0

    def channel_at(epsilon: float) -> DiscretePosteriorChannel:
        channel = make_score_channel(family, prior, epsilon, **dict(channel_kwargs))
        assert isinstance(channel, DiscretePosteriorChannel)
        return channel

    def residual(epsilon: float) -> float:
        return statistic(channel_at(epsilon)) - desired

    upper = 0.25
    previous = baseline
    for _ in range(24):
        current = statistic(channel_at(upper))
        if current >= desired:
            break
        # Once every logistic category is saturated, further doubling cannot
        # improve a target above this discrete channel's intrinsic ceiling.
        if upper >= 256.0 and current - previous < 2e-13:
            raise ValueError("target is above this discrete channel's attainable ceiling")
        previous = current
        upper *= 2.0
    else:
        raise ValueError("target is above this discrete channel's attainable ceiling")
    return float(brentq(residual, 0.0, upper, xtol=2e-12, rtol=2e-12))


def calibrate_epsilon_for_information(
    family: str,
    p: float,
    J_bits: float,
    **channel_kwargs: Any,
) -> float:
    """Find ``epsilon`` yielding the requested mutual information in bits."""

    if family in CONTINUOUS_FAMILIES:
        if channel_kwargs:
            raise TypeError(f"unexpected channel arguments: {sorted(channel_kwargs)}")
        return _solve_continuous_strength(
            family,
            p,
            J_bits,
            statistic=lambda channel: channel.mutual_information_bits(),
            baseline=0.0,
            theoretical_ceiling=float(h2(_validate_prior(p))),
        )
    return _solve_discrete_strength(
        family,
        p,
        J_bits,
        statistic=lambda channel: channel.mutual_information_bits(),
        baseline=0.0,
        theoretical_ceiling=float(h2(_validate_prior(p))),
        channel_kwargs=channel_kwargs,
    )


def calibrate_epsilon_for_auc(
    family: str,
    p: float,
    target_auc: float,
    **channel_kwargs: Any,
) -> float:
    """Find ``epsilon`` yielding the requested AUC."""

    if family in CONTINUOUS_FAMILIES:
        if channel_kwargs:
            raise TypeError(f"unexpected channel arguments: {sorted(channel_kwargs)}")
        return _solve_continuous_strength(
            family,
            p,
            target_auc,
            statistic=lambda channel: channel.auc(),
            baseline=0.5,
            theoretical_ceiling=1.0,
        )
    return _solve_discrete_strength(
        family,
        p,
        target_auc,
        statistic=lambda channel: channel.auc(),
        baseline=0.5,
        theoretical_ceiling=1.0,
        channel_kwargs=channel_kwargs,
    )


def make_continuous_channel_for_information(
    family: str, p: float, J_bits: float
) -> ContinuousPosteriorChannel:
    epsilon = calibrate_epsilon_for_information(family, p, J_bits)
    return make_continuous_channel(family, p, epsilon)


def make_continuous_channel_for_auc(
    family: str, p: float, target_auc: float
) -> ContinuousPosteriorChannel:
    epsilon = calibrate_epsilon_for_auc(family, p, target_auc)
    return make_continuous_channel(family, p, epsilon)


def make_channel_for_information(
    family: str,
    p: float,
    J_bits: float,
    **channel_kwargs: Any,
) -> Union[ContinuousPosteriorChannel, DiscretePosteriorChannel]:
    """Construct any supported family at a common target information level."""

    if family == "two_point_frontier":
        return two_point_frontier_channel(p, J_bits=J_bits, **channel_kwargs)
    epsilon = calibrate_epsilon_for_information(family, p, J_bits, **channel_kwargs)
    return make_score_channel(family, p, epsilon, **channel_kwargs)


def make_channel_for_auc(
    family: str,
    p: float,
    target_auc: float,
    **channel_kwargs: Any,
) -> Union[ContinuousPosteriorChannel, DiscretePosteriorChannel]:
    """Construct a logistic continuous/discrete family at a target AUC."""

    if family == "two_point_frontier":
        raise ValueError("two_point_frontier is parameterized by q or J_bits, not epsilon")
    epsilon = calibrate_epsilon_for_auc(family, p, target_auc, **channel_kwargs)
    return make_score_channel(family, p, epsilon, **channel_kwargs)


__all__ = [
    "CONTINUOUS_FAMILIES",
    "DISCRETE_FAMILIES",
    "SUPPORTED_FAMILIES",
    "ContinuousPosteriorChannel",
    "DiscretePosteriorChannel",
    "calibrate_logistic_intercept",
    "make_continuous_channel",
    "calibrated_discrete_logistic_channel",
    "rare_spike_channel",
    "three_point_channel",
    "two_point_frontier_channel",
    "make_score_channel",
    "calibrate_epsilon_for_information",
    "calibrate_epsilon_for_auc",
    "make_continuous_channel_for_information",
    "make_continuous_channel_for_auc",
    "make_channel_for_information",
    "make_channel_for_auc",
]
