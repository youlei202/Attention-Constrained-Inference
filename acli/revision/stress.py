r"""Exact finite-channel audits for information accumulation.

These helpers are intentionally independent of Monte Carlo simulation.  They
enumerate output *types* to compute the iid verification profile and enumerate
small observed hit/miss transcripts to audit the identity

.. math::

    I(\Theta; \text{verification transcript})
      = \mathbb{E}[\Phi(N_B)].

The stress runners merely sample finite priors/channels; each sampled case is
then evaluated deterministically.  Explicit RNG seeds make failures exactly
reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Optional, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

from .accumulation import (
    PROFILE_TOLERANCE,
    assert_valid_profile,
    expected_profile_poisson_binomial,
    poisson_binomial_pmf,
    profile_violations,
)


_LOG_2 = float(np.log(2.0))


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        adjective = "nonnegative" if allow_zero else "positive"
        raise TypeError(f"{name} must be a {adjective} integer")
    value = int(value)
    lower = 0 if allow_zero else 1
    if value < lower:
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {adjective}")
    return value


def _probability_vector(values: Sequence[float], name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(np.sum(probabilities))
    if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must sum to one (got {total:.17g})")
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return probabilities / total


def _finite_channel(channel: Sequence[Sequence[float]], n_states: int) -> np.ndarray:
    matrix = np.asarray(channel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_states or matrix.shape[1] == 0:
        raise ValueError(
            "channel must have shape (len(prior), n_outputs) with n_outputs >= 1"
        )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("channel must contain finite nonnegative probabilities")
    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("each channel row must sum to one")
    if np.any(row_sums <= 0.0):
        raise ValueError("each channel row must have positive mass")
    return matrix / row_sums[:, None]


def entropy_bits(probabilities: Sequence[float]) -> float:
    """Shannon entropy of a finite probability vector, in bits."""

    p = _probability_vector(probabilities, "probabilities")
    return float(-np.sum(xlogy(p, p)) / _LOG_2)


@lru_cache(maxsize=None)
def _count_vectors(n: int, alphabet_size: int) -> np.ndarray:
    """All weak compositions of ``n`` into ``alphabet_size`` cells."""

    rows: list[tuple[int, ...]] = []

    def append_compositions(remaining: int, cells: int, prefix: tuple[int, ...]) -> None:
        if cells == 1:
            rows.append(prefix + (remaining,))
            return
        for first in range(remaining + 1):
            append_compositions(remaining - first, cells - 1, prefix + (first,))

    append_compositions(n, alphabet_size, ())
    result = np.asarray(rows, dtype=np.int64)
    result.setflags(write=False)
    return result


def _conditional_count_log_probabilities(channel: np.ndarray, n: int) -> np.ndarray:
    """Log P(output count vector | theta), one row per theta."""

    counts = _count_vectors(n, channel.shape[1])
    log_coefficient = gammaln(n + 1.0) - np.sum(gammaln(counts + 1.0), axis=1)
    result = np.empty((channel.shape[0], counts.shape[0]), dtype=np.float64)
    for state, row in enumerate(channel):
        positive = row > 0.0
        log_probability = log_coefficient.copy()
        if np.any(positive):
            log_probability += counts[:, positive] @ np.log(row[positive])
        impossible = np.any(counts[:, ~positive] > 0, axis=1) if np.any(~positive) else False
        if np.ndim(impossible):
            log_probability[impossible] = -np.inf
        result[state] = log_probability
    return result


def _mutual_information_from_log_conditionals(
    prior: np.ndarray,
    log_conditionals: np.ndarray,
) -> float:
    active = prior > 0.0
    active_prior = prior[active]
    active_log_conditional = log_conditionals[active]
    log_marginal = logsumexp(
        np.log(active_prior)[:, None] + active_log_conditional,
        axis=0,
    )
    information_nats = 0.0
    for probability, log_conditional in zip(active_prior, active_log_conditional):
        possible = np.isfinite(log_conditional)
        conditional_probability = np.exp(log_conditional[possible])
        information_nats += probability * float(
            np.sum(
                conditional_probability
                * (log_conditional[possible] - log_marginal[possible])
            )
        )
    return float(information_nats / _LOG_2)


def iid_channel_profile(
    prior: Sequence[float],
    channel: Sequence[Sequence[float]],
    n_max: int,
) -> np.ndarray:
    """Exact ``I(Theta; Y^n)`` profile for a finite iid channel.

    Rows of ``channel`` are ``P(Y=y | Theta=theta)``.  Sequence outcomes are
    aggregated by their count vectors, reducing ``|Y|**n`` transcripts to
    ``binomial(n+|Y|-1, |Y|-1)`` output types without changing mutual
    information.
    """

    n_max = _positive_integer(n_max, "n_max", allow_zero=True)
    prior_array = _probability_vector(prior, "prior")
    channel_array = _finite_channel(channel, prior_array.size)
    ceiling = entropy_bits(prior_array)

    profile = np.empty(n_max + 1, dtype=np.float64)
    profile[0] = 0.0
    for n in range(1, n_max + 1):
        log_conditionals = _conditional_count_log_probabilities(channel_array, n)
        information = _mutual_information_from_log_conditionals(
            prior_array, log_conditionals
        )
        profile[n] = np.clip(information, 0.0, ceiling)
    return profile


# Descriptive alias retained for experiment code that emphasizes verification.
finite_iid_verification_profile = iid_channel_profile


def _transcript_mutual_information(
    prior: np.ndarray,
    channel: np.ndarray,
    hit_probabilities: np.ndarray,
    *,
    max_transcripts: int,
) -> tuple[float, int]:
    """Directly enumerate observed ``miss`` or ``(hit, output)`` symbols."""

    alphabet_size = channel.shape[1] + 1
    n_transcripts = alphabet_size ** hit_probabilities.size
    if n_transcripts > max_transcripts:
        raise ValueError(
            f"audit would enumerate {n_transcripts} transcripts; "
            f"increase max_transcripts (currently {max_transcripts}) explicitly"
        )

    conditional = np.ones((prior.size, 1), dtype=np.float64)
    for probability in hit_probabilities:
        local = np.concatenate(
            (
                np.full((prior.size, 1), 1.0 - probability, dtype=np.float64),
                probability * channel,
            ),
            axis=1,
        )
        conditional = (conditional[:, :, None] * local[:, None, :]).reshape(
            prior.size, -1
        )

    marginal = prior @ conditional
    information_nats = 0.0
    for state, state_probability in enumerate(prior):
        if state_probability == 0.0:
            continue
        possible = conditional[state] > 0.0
        information_nats += state_probability * float(
            np.sum(
                conditional[state, possible]
                * (
                    np.log(conditional[state, possible])
                    - np.log(marginal[possible])
                )
            )
        )
    return float(information_nats / _LOG_2), int(n_transcripts)


def audit_transcript_vs_expected_profile(
    prior: Sequence[float],
    channel: Sequence[Sequence[float]],
    probabilities: Sequence[float],
    *,
    max_transcripts: int = 2_000_000,
) -> dict[str, float | int]:
    """Audit exact transcript MI against ``E[Phi(N_B)]``.

    For selected item ``i``, the hit indicator is independently Bernoulli with
    probability ``probabilities[i]`` and is observed as part of verification.
    A hit produces one draw from the common finite verification channel; a miss
    produces a target-independent symbol.  The first value below enumerates all
    such observed transcripts directly, while the second groups them only by
    the Poisson-binomial hit count.
    """

    max_transcripts = _positive_integer(max_transcripts, "max_transcripts")
    prior_array = _probability_vector(prior, "prior")
    channel_array = _finite_channel(channel, prior_array.size)
    hit_probabilities = np.asarray(probabilities, dtype=np.float64)
    if hit_probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if (
        not np.all(np.isfinite(hit_probabilities))
        or np.any(hit_probabilities < 0.0)
        or np.any(hit_probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")

    exact_mi, n_transcripts = _transcript_mutual_information(
        prior_array,
        channel_array,
        hit_probabilities,
        max_transcripts=max_transcripts,
    )
    profile = iid_channel_profile(prior_array, channel_array, hit_probabilities.size)
    # The analytic iid profile must itself obey diminishing returns and the
    # finite target-entropy ceiling before it is used by the identity audit.
    assert_valid_profile(profile, entropy_bits(prior_array))
    expected_mi = expected_profile_poisson_binomial(profile, hit_probabilities)
    signed_error = exact_mi - expected_mi
    return {
        "exact_transcript_mi": float(exact_mi),
        "expected_profile_mi": float(expected_mi),
        "expected_phi_n": float(expected_mi),
        "signed_error": float(signed_error),
        "absolute_error": abs(float(signed_error)),
        "budget": int(hit_probabilities.size),
        "n_transcripts": n_transcripts,
    }


# Short aliases make the helper convenient in experiment and test code.
exact_transcript_profile_audit = audit_transcript_vs_expected_profile
audit_transcript_identity = audit_transcript_vs_expected_profile


def _draw_random_channel(
    rng: np.random.Generator,
    max_states: int,
    max_outputs: int,
    trial: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_states = int(rng.integers(2, max_states + 1))
    n_outputs = int(rng.integers(2, max_outputs + 1))
    prior_concentration = 10.0 ** float(rng.uniform(-1.0, 1.0))
    prior = rng.dirichlet(np.full(n_states, prior_concentration))

    rows = []
    for state in range(n_states):
        row_concentration = 10.0 ** float(rng.uniform(-1.0, 1.0))
        row = rng.dirichlet(np.full(n_outputs, row_concentration))
        # Exercise exact zero/one branches regularly, while retaining random
        # dense channels in most trials.
        if trial % 17 == 0 and state == trial % n_states:
            row = np.zeros(n_outputs, dtype=np.float64)
            row[int(rng.integers(n_outputs))] = 1.0
        rows.append(row)
    return prior, np.asarray(rows, dtype=np.float64)


def random_iid_channel_stress(
    n_channels: int = 50_000,
    *,
    n_max: int = 6,
    seed: int = 20260209,
    max_states: int = 4,
    max_outputs: int = 3,
    atol: float = PROFILE_TOLERANCE,
) -> dict[str, object]:
    """Stress monotonicity, concavity, and entropy ceilings on random channels.

    The summary records maxima and the full worst sampled case, rather than only
    a pass/fail flag.  It is JSON-serializable and deterministic for a fixed
    seed and NumPy version.
    """

    n_channels = _positive_integer(n_channels, "n_channels")
    n_max = _positive_integer(n_max, "n_max", allow_zero=True)
    max_states = _positive_integer(max_states, "max_states")
    max_outputs = _positive_integer(max_outputs, "max_outputs")
    if max_states < 2 or max_outputs < 2:
        raise ValueError("max_states and max_outputs must each be at least two")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    maxima = {
        "monotonicity_violation": 0.0,
        "concavity_violation": 0.0,
        "entropy_ceiling_violation": 0.0,
        "origin_abs_error": 0.0,
    }
    failure_count = 0
    worst_score = -1.0
    worst_case: dict[str, object] = {}

    for trial in range(n_channels):
        prior, channel = _draw_random_channel(
            rng, max_states, max_outputs, trial
        )
        profile = iid_channel_profile(prior, channel, n_max)
        diagnostics = profile_violations(profile, entropy_bits(prior))
        current = {name: float(diagnostics[name]) for name in maxima}
        for name, value in current.items():
            maxima[name] = max(maxima[name], value)
        score = max(current.values())
        if score > worst_score:
            worst_score = score
            worst_case = {
                "trial": trial,
                "prior": prior.tolist(),
                "channel": channel.tolist(),
                "profile": profile.tolist(),
                **current,
            }
        if any(value > atol for value in current.values()):
            failure_count += 1

    return {
        "seed": int(seed),
        "n_channels": n_channels,
        "n_max": n_max,
        "atol": float(atol),
        "failure_count": failure_count,
        "passed": failure_count == 0,
        "max_monotonicity_violation": maxima["monotonicity_violation"],
        "max_concavity_violation": maxima["concavity_violation"],
        "max_entropy_ceiling_violation": maxima["entropy_ceiling_violation"],
        "max_origin_abs_error": maxima["origin_abs_error"],
        "worst_case": worst_case,
    }


# Alias matching the runbook wording.
random_finite_iid_verification_channel_stress = random_iid_channel_stress


def random_transcript_identity_stress(
    n_instances: int = 2_000,
    *,
    max_budget: int = 5,
    seed: int = 20260210,
    max_states: int = 3,
    max_outputs: int = 3,
) -> dict[str, object]:
    """Run small exact transcript-vs-``E[Phi(N_B)]`` identity audits."""

    n_instances = _positive_integer(n_instances, "n_instances")
    max_budget = _positive_integer(max_budget, "max_budget", allow_zero=True)
    max_states = _positive_integer(max_states, "max_states")
    max_outputs = _positive_integer(max_outputs, "max_outputs")
    if max_states < 2 or max_outputs < 2:
        raise ValueError("max_states and max_outputs must each be at least two")

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    max_absolute_error = 0.0
    worst_case: dict[str, object] = {}
    for trial in range(n_instances):
        prior, channel = _draw_random_channel(
            rng, max_states, max_outputs, trial
        )
        budget = int(rng.integers(max_budget + 1))
        probabilities = rng.beta(0.7, 0.7, size=budget)
        if budget and trial % 11 == 0:
            probabilities[trial % budget] = float((trial // 11) % 2)
        audit = audit_transcript_vs_expected_profile(
            prior,
            channel,
            probabilities,
        )
        error = float(audit["absolute_error"])
        if trial == 0 or error > max_absolute_error:
            max_absolute_error = error
            worst_case = {
                "trial": trial,
                "prior": prior.tolist(),
                "channel": channel.tolist(),
                "probabilities": probabilities.tolist(),
                **audit,
            }

    return {
        "seed": int(seed),
        "n_instances": n_instances,
        "max_budget": max_budget,
        "max_absolute_error": max_absolute_error,
        "worst_case": worst_case,
    }


__all__ = [
    "audit_transcript_identity",
    "audit_transcript_vs_expected_profile",
    "entropy_bits",
    "exact_transcript_profile_audit",
    "finite_iid_verification_profile",
    "iid_channel_profile",
    "random_finite_iid_verification_channel_stress",
    "random_iid_channel_stress",
    "random_transcript_identity_stress",
]
