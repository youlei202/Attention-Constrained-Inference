from __future__ import annotations

import itertools

import numpy as np
import pytest

from acli.revision.accumulation import (
    PROFILE_TOLERANCE,
    assert_valid_profile,
    component_target_profile,
    concave_interpolation,
    decoupled_profile,
    expected_profile_poisson_binomial,
    phi_bsc,
    phi_bsc_profile,
    poisson_binomial_pmf,
    profile_margins,
    profile_violations,
)
from acli.revision.stress import (
    audit_transcript_vs_expected_profile,
    entropy_bits,
    iid_channel_profile,
    random_iid_channel_stress,
    random_transcript_identity_stress,
)


def _binary_entropy(p: float) -> float:
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


@pytest.mark.parametrize("beta", [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0])
def test_balanced_bsc_one_hit_and_boundaries(beta: float) -> None:
    expected = 1.0 - _binary_entropy(beta)
    assert phi_bsc(1, 0.5, beta) == pytest.approx(expected, abs=2e-14)
    assert phi_bsc(0, 0.5, beta) == 0.0
    assert phi_bsc(10, 0.0, beta) == 0.0
    assert phi_bsc(10, 1.0, beta) == 0.0


@pytest.mark.parametrize("pi", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("beta", [0.05, 0.1, 0.2, 0.4])
def test_bsc_profiles_are_monotone_concave_and_entropy_bounded(
    pi: float, beta: float
) -> None:
    profile = phi_bsc_profile(256, pi, beta)
    ceiling = _binary_entropy(pi)
    margins = profile_margins(profile)

    assert np.min(margins) >= -PROFILE_TOLERANCE
    assert np.max(np.diff(margins)) <= PROFILE_TOLERANCE
    assert np.max(profile) <= ceiling + PROFILE_TOLERANCE
    assert_valid_profile(profile, ceiling)


def test_bsc_output_relabeling_does_not_change_information() -> None:
    for n in range(8):
        assert phi_bsc(n, 0.2, 0.1) == pytest.approx(
            phi_bsc(n, 0.2, 0.9), abs=2e-14
        )


def test_component_profile_has_expected_occupancy_formula() -> None:
    d = 7
    beta = 0.1
    profile = component_target_profile(32, d, beta)
    scalar = phi_bsc_profile(32, 0.5, beta)

    assert np.array_equal(component_target_profile(32, 1, beta), scalar)
    assert profile[1] == pytest.approx(scalar[1], abs=2e-14)
    assert profile[2] == pytest.approx(
        2.0 * (1.0 - 1.0 / d) * scalar[1] + scalar[2] / d,
        abs=2e-14,
    )
    assert_valid_profile(profile, float(d))
    assert np.all(profile <= np.arange(profile.size) * scalar[1] + 2e-12)


def test_decoupled_profile_is_linear_and_unbounded_by_shared_entropy() -> None:
    profile = decoupled_profile(10, 0.37)
    assert np.allclose(profile, np.arange(11) * 0.37, atol=0.0, rtol=0.0)
    assert np.allclose(profile_margins(profile), 0.37)
    assert profile[-1] > 1.0


def test_concave_interpolation_scalar_and_array() -> None:
    profile = np.array([0.0, 0.8, 1.2, 1.4])
    assert concave_interpolation(profile, 1.25) == pytest.approx(0.9)
    values = concave_interpolation(profile, np.array([[0.0, 1.5], [2.0, 3.0]]))
    assert np.allclose(values, [[0.0, 1.0], [1.2, 1.4]])
    with pytest.raises(ValueError, match="must lie"):
        concave_interpolation(profile, 3.1)
    with pytest.raises(ValueError, match="concavity"):
        concave_interpolation([0.0, 0.1, 0.3], 1.0)


def _brute_poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    result = np.zeros(probabilities.size + 1)
    for outcomes in itertools.product((0, 1), repeat=probabilities.size):
        mass = 1.0
        for outcome, probability in zip(outcomes, probabilities):
            mass *= probability if outcome else 1.0 - probability
        result[sum(outcomes)] += mass
    return result


def test_poisson_binomial_pmf_matches_brute_force_and_sums_to_one() -> None:
    probabilities = np.array([0.0, 0.13, 0.41, 0.88, 1.0])
    pmf = poisson_binomial_pmf(probabilities)
    assert np.allclose(pmf, _brute_poisson_binomial(probabilities), atol=2e-16)
    assert np.sum(pmf) == pytest.approx(1.0, abs=2e-16)
    assert np.all(pmf >= 0.0)
    assert np.array_equal(poisson_binomial_pmf([]), np.array([1.0]))


def test_expected_profile_uses_nonidentical_probabilities_exactly() -> None:
    probabilities = np.array([0.12, 0.37, 0.91])
    profile = phi_bsc_profile(probabilities.size, 0.2, 0.1)
    expected = expected_profile_poisson_binomial(profile, probabilities)
    brute_pmf = _brute_poisson_binomial(probabilities)
    assert expected == pytest.approx(float(brute_pmf @ profile), abs=2e-16)

    # Replacing the vector by its mean generally changes the distribution and
    # is intentionally not what the exact helper computes.
    binomial_approximation = poisson_binomial_pmf(
        np.full(probabilities.size, probabilities.mean())
    ) @ profile
    assert abs(expected - binomial_approximation) > 1e-5


def test_generic_iid_channel_profile_matches_bsc_profile() -> None:
    pi = 0.2
    beta = 0.13
    prior = [1.0 - pi, pi]
    channel = [[1.0 - beta, beta], [beta, 1.0 - beta]]
    generic = iid_channel_profile(prior, channel, 9)
    specialized = phi_bsc_profile(9, pi, beta)
    assert np.allclose(generic, specialized, atol=2e-14, rtol=0.0)
    assert_valid_profile(generic, entropy_bits(prior))


@pytest.mark.parametrize(
    "prior,channel,probabilities",
    [
        ([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [0.2, 0.7, 1.0]),
        ([0.2, 0.3, 0.5], [[1, 0], [0.4, 0.6], [0.1, 0.9]], [0.0, 0.3]),
        ([0.1, 0.9], [[0.3, 0.2, 0.5], [0.8, 0.1, 0.1]], []),
    ],
)
def test_exact_transcript_mi_equals_expected_phi_of_hit_count(
    prior, channel, probabilities
) -> None:
    audit = audit_transcript_vs_expected_profile(prior, channel, probabilities)
    assert audit["absolute_error"] <= 2e-14
    assert audit["budget"] == len(probabilities)


def test_random_finite_channel_stress_is_seeded_and_records_maxima() -> None:
    first = random_iid_channel_stress(80, n_max=6, seed=481)
    second = random_iid_channel_stress(80, n_max=6, seed=481)
    assert first == second
    assert first["passed"]
    assert first["failure_count"] == 0
    assert first["max_monotonicity_violation"] <= PROFILE_TOLERANCE
    assert first["max_concavity_violation"] <= PROFILE_TOLERANCE
    assert first["max_entropy_ceiling_violation"] <= PROFILE_TOLERANCE
    assert first["worst_case"]["trial"] >= 0


def test_random_transcript_identity_stress_is_at_machine_precision() -> None:
    summary = random_transcript_identity_stress(60, max_budget=4, seed=991)
    assert summary["max_absolute_error"] <= 2e-14
    assert summary["worst_case"]["trial"] >= 0


def test_profile_validation_and_probability_input_errors_are_explicit() -> None:
    diagnostics = profile_violations([0.0, 0.4, 0.9], entropy_ceiling=0.8)
    assert diagnostics["concavity_violation"] == pytest.approx(0.1)
    assert diagnostics["entropy_ceiling_violation"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="invalid accumulation profile"):
        assert_valid_profile([0.0, 0.4, 0.9], entropy_ceiling=0.8)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        poisson_binomial_pmf([0.2, 1.1])
    with pytest.raises(ValueError, match="entries through"):
        expected_profile_poisson_binomial([0.0, 0.1], [0.2, 0.3])
