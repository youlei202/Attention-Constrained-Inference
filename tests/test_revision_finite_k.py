import math

import numpy as np
import pytest

from acli.revision.finite_k import (
    finite_k_top_b_precision,
    large_k_top_tail_precision,
    monte_carlo_top_b_precision,
    relative_boost_error,
    top_b_precision_from_samples,
)


class UniformPosteriorChannel:
    """Posterior eta ~ Uniform(0, 1/2), hence E[eta] = p = 1/4."""

    p = 0.25

    @staticmethod
    def posterior_quantile(u):
        return 0.5 * np.asarray(u)

    @staticmethod
    def sample_eta(n, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 0.5, size=n)


class TwoPointPosteriorChannel:
    posterior_values = np.array([0.1, 0.8])
    probabilities = np.array([0.8, 0.2])
    p = 0.24

    @staticmethod
    def sample_posterior(n, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        return rng.choice(
            TwoPointPosteriorChannel.posterior_values,
            size=n,
            p=TwoPointPosteriorChannel.probabilities,
        )


def test_continuous_finite_k_matches_uniform_order_statistic_formula():
    channel = UniformPosteriorChannel()
    K = 20
    B = 4
    # E[U_(r)] = r/(K+1), averaged over r=K-B+1,...,K, and eta=U/2.
    expected = 0.5 * (2 * K - B + 1) / (2 * (K + 1))
    actual = finite_k_top_b_precision(channel, K, B)
    assert actual == pytest.approx(expected, abs=3e-10)


def test_selecting_all_records_recovers_calibrated_prior():
    channel = UniformPosteriorChannel()
    assert finite_k_top_b_precision(channel, K=200, B=200) == channel.p


def test_discrete_finite_k_is_exact_with_ties():
    channel = TwoPointPosteriorChannel()
    K = 5
    # For B=1 the maximum is high iff at least one high atom was sampled.
    expected = 0.1 + (0.8 - 0.1) * (1.0 - 0.8**K)
    assert finite_k_top_b_precision(channel, K=K, B=1) == pytest.approx(
        expected, abs=2e-15
    )


def test_large_k_top_tail_is_quantile_integral_for_continuous_channel():
    channel = UniformPosteriorChannel()
    alpha = 0.2
    expected = 0.5 * (1.0 - alpha / 2.0)
    assert large_k_top_tail_precision(channel, alpha) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.1, 0.8),
        (0.2, 0.8),
        (0.3, (0.2 * 0.8 + 0.1 * 0.1) / 0.3),
        (1.0, 0.24),
    ],
)
def test_large_k_discrete_tail_handles_fractional_boundary_atom(alpha, expected):
    assert large_k_top_tail_precision(TwoPointPosteriorChannel(), alpha) == pytest.approx(
        expected, abs=2e-15
    )


def test_relative_boost_error_uses_boost_not_precision_as_denominator():
    assert relative_boost_error(0.39, 0.4, 0.2) == pytest.approx(0.05)
    assert relative_boost_error(0.2, 0.2, 0.2) == 0.0
    assert math.isinf(relative_boost_error(0.21, 0.2, 0.2))


def test_sample_top_b_helper_uses_partition_and_matches_sort(monkeypatch):
    samples = np.array([[0.2, 0.9, 0.1, 0.7], [0.4, 0.3, 0.8, 0.6]])
    original = np.argpartition
    called = False

    def recording_argpartition(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "argpartition", recording_argpartition)
    actual = top_b_precision_from_samples(samples, B=2, axis=1)
    expected = np.mean(np.sort(samples, axis=1)[:, -2:], axis=1)
    assert called
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_deterministic_finite_k_agrees_with_monte_carlo_within_three_se():
    channel = UniformPosteriorChannel()
    deterministic = finite_k_top_b_precision(channel, K=30, B=6)
    estimate, standard_error = monte_carlo_top_b_precision(
        channel,
        K=30,
        B=6,
        n_trials=6_000,
        seed=731,
        batch_size=500,
    )
    assert abs(estimate - deterministic) <= 3.0 * standard_error + 2e-12


def test_protocol_fallback_accepts_channel_top_tail_method():
    class MethodOnlyChannel:
        @staticmethod
        def top_tail_precision(alpha):
            return 0.3 + alpha

    assert large_k_top_tail_precision(MethodOnlyChannel(), 0.1) == pytest.approx(0.4)


@pytest.mark.parametrize(("K", "B"), [(0, 1), (10, 0), (4, 5)])
def test_invalid_k_b_are_rejected(K, B):
    with pytest.raises(ValueError):
        finite_k_top_b_precision(UniformPosteriorChannel(), K, B)
