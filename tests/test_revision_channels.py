from __future__ import annotations

import numpy as np
import pytest

from acli.revision.channels import (
    CONTINUOUS_FAMILIES,
    DiscretePosteriorChannel,
    calibrate_epsilon_for_auc,
    calibrate_epsilon_for_information,
    make_continuous_channel,
    make_score_channel,
    rare_spike_channel,
    three_point_channel,
    two_point_frontier_channel,
)
from acli.revision.information import binary_kl_bits


@pytest.mark.parametrize("family", CONTINUOUS_FAMILIES)
def test_continuous_families_are_exactly_calibrated(family: str) -> None:
    channel = make_continuous_channel(family, p=0.01, epsilon=0.4)
    assert channel.calibration_error <= 1e-10
    assert channel.mean_eta == pytest.approx(0.01, abs=1e-10)
    assert 0.0 < channel.mutual_information_bits() < 1.0
    assert 0.5 < channel.auc() < 1.0
    assert channel.top_tail_precision(0.01) > channel.p

    quantiles = channel.posterior_quantile(np.linspace(0.001, 0.999, 51))
    assert np.all(np.diff(quantiles) >= 0.0)
    samples = channel.sample_eta(64, seed=123)
    np.testing.assert_array_equal(samples, channel.sample_posterior(64, seed=123))


def test_zero_strength_channel_is_prior_everywhere() -> None:
    channel = make_continuous_channel("gaussian", p=0.2, epsilon=0.0)
    np.testing.assert_allclose(channel.posterior_quantile([0.0, 0.5, 1.0]), 0.2, atol=1e-15)
    assert channel.mutual_information_bits() == pytest.approx(0.0, abs=1e-15)
    assert channel.auc() == 0.5
    assert channel.top_tail_precision(0.001) == pytest.approx(0.2, abs=1e-15)


def test_information_and_auc_target_calibration() -> None:
    epsilon_j = calibrate_epsilon_for_information("gaussian", 0.01, 0.001)
    channel_j = make_continuous_channel("gaussian", 0.01, epsilon_j)
    assert channel_j.mutual_information_bits() == pytest.approx(0.001, abs=2e-10)

    epsilon_auc = calibrate_epsilon_for_auc("gaussian", 0.01, 0.70)
    channel_auc = make_continuous_channel("gaussian", 0.01, epsilon_auc)
    assert channel_auc.auc() == pytest.approx(0.70, abs=2e-10)
    assert channel_auc.calibration_error <= 1e-10


@pytest.mark.parametrize(
    "channel",
    [
        rare_spike_channel(0.01, 0.4),
        three_point_channel(0.01, 0.4),
        two_point_frontier_channel(0.01, 0.1, q=0.05),
    ],
)
def test_discrete_families_have_exact_mean_and_protocol(channel: DiscretePosteriorChannel) -> None:
    assert channel.calibration_error <= 1e-10
    assert np.dot(channel.posterior_values, channel.probabilities) == pytest.approx(
        channel.p, abs=1e-12
    )
    assert channel.top_tail_precision(1.0) == pytest.approx(channel.p, abs=1e-12)
    assert 0.5 <= channel.auc() <= 1.0
    assert np.all(np.diff(channel.posterior_quantile(np.linspace(0.0, 1.0, 101))) >= 0.0)
    np.testing.assert_array_equal(channel.sample_eta(100, seed=4), channel.sample_eta(100, seed=4))


def test_discrete_auc_awards_half_credit_to_ties() -> None:
    # Marginal P(eta=0.25)=P(eta=0.75)=1/2 and p=1/2.  Conditional score
    # masses are (0.25, 0.75) for T=1 and (0.75, 0.25) for T=0.
    channel = DiscretePosteriorChannel(
        p=0.5,
        posterior_values=np.array([0.25, 0.75]),
        probabilities=np.array([0.5, 0.5]),
    )
    strictly_greater = 0.75 * 0.75
    ties = 0.25 * 0.75 + 0.75 * 0.25
    assert channel.auc() == pytest.approx(strictly_greater + 0.5 * ties, abs=1e-15)


def test_two_point_frontier_target_information_and_fractional_tail_atom() -> None:
    direct = two_point_frontier_channel(0.05, 0.2, q=0.18)
    target = direct.mutual_information_bits()
    inverted = two_point_frontier_channel(0.05, 0.2, J_bits=target)
    assert inverted.posterior_values[-1] == pytest.approx(0.18, abs=2e-12)
    assert inverted.mutual_information_bits() == pytest.approx(target, abs=2e-12)

    # Selecting half of the high atom still yields its posterior exactly.
    assert inverted.top_tail_precision(0.1) == pytest.approx(0.18, abs=2e-12)


def test_discrete_information_matches_weighted_binary_kl() -> None:
    channel = three_point_channel(0.1, 0.7)
    expected = np.dot(
        channel.probabilities,
        binary_kl_bits(channel.posterior_values, channel.p),
    )
    assert channel.mutual_information_bits() == pytest.approx(expected, abs=1e-15)


def test_unified_factory_supports_all_required_names() -> None:
    for family in CONTINUOUS_FAMILIES:
        assert make_score_channel(family, 0.05, 0.2).family == family
    assert make_score_channel("rare_spike", 0.05, 0.2).family == "rare_spike"
    assert make_score_channel("three_point", 0.05, 0.2).family == "three_point"
    frontier = make_score_channel("two_point_frontier", 0.05, alpha=0.1, q=0.2)
    assert frontier.family == "two_point_frontier"
