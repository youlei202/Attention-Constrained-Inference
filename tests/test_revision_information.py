from __future__ import annotations

import json

import numpy as np
import pytest

from acli.revision.information import (
    binary_kl_bits,
    entropy_bits,
    h2,
    mutual_information_from_joint,
    posterior_mutual_information_bits,
)
from acli.revision.reproducibility import (
    canonical_configuration,
    make_rng,
    stable_task_seed,
    task_rng,
)


def test_h2_boundaries_symmetry_and_bits() -> None:
    np.testing.assert_array_equal(h2(np.array([0.0, 1.0])), np.array([0.0, 0.0]))
    assert h2(0.5) == pytest.approx(1.0, abs=1e-15)
    assert h2(0.17) == pytest.approx(h2(0.83), abs=2e-15)
    with pytest.raises(ValueError):
        h2(-1e-8)


def test_binary_kl_bits_known_values_and_boundary_conventions() -> None:
    assert binary_kl_bits(0.5, 0.5) == pytest.approx(0.0, abs=1e-15)
    assert binary_kl_bits(1.0, 0.5) == pytest.approx(1.0, abs=1e-15)
    assert binary_kl_bits(0.0, 0.5) == pytest.approx(1.0, abs=1e-15)
    assert binary_kl_bits(0.0, 0.0) == 0.0
    assert binary_kl_bits(1.0, 1.0) == 0.0
    assert np.isinf(binary_kl_bits(0.1, 0.0))
    assert np.isinf(binary_kl_bits(0.9, 1.0))

    q = np.array([0.1, 0.3, 0.9])
    expected = q * np.log2(q / 0.3) + (1.0 - q) * np.log2((1.0 - q) / 0.7)
    np.testing.assert_allclose(binary_kl_bits(q, 0.3), expected, atol=2e-15)


def test_entropy_bits_uses_zero_log_zero_and_validates_mass() -> None:
    assert entropy_bits([0.5, 0.5, 0.0]) == pytest.approx(1.0, abs=1e-15)
    rows = entropy_bits([[0.5, 0.5], [1.0, 0.0]], axis=1)
    np.testing.assert_allclose(rows, [1.0, 0.0], atol=1e-15)
    with pytest.raises(ValueError, match="sum to one"):
        entropy_bits([0.2, 0.2])
    with pytest.raises(ValueError, match="nonnegative"):
        entropy_bits([1.1, -0.1])


def test_mutual_information_from_joint_independent_and_perfect() -> None:
    independent = np.array([[0.12, 0.18], [0.28, 0.42]])
    assert mutual_information_from_joint(independent) == pytest.approx(0.0, abs=2e-15)
    perfect_fair_bit = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert mutual_information_from_joint(perfect_fair_bit) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError, match="two-dimensional"):
        mutual_information_from_joint([0.5, 0.5])


def test_posterior_information_identity_for_finite_distribution() -> None:
    eta = np.array([0.1, 0.5, 0.9])
    weights = np.array([0.25, 0.5, 0.25])
    direct = np.dot(weights, binary_kl_bits(eta, 0.5))
    assert posterior_mutual_information_bits(eta, 0.5, weights) == pytest.approx(
        direct, abs=1e-15
    )


def test_stable_task_seed_uses_full_canonical_configuration() -> None:
    config_a = {"alpha": np.float64(0.01), "families": ["gaussian", "pareto4"]}
    config_reordered = {"families": ["gaussian", "pareto4"], "alpha": 0.01}
    first = stable_task_seed(2026, "finite_k", "cell", config_a)
    second = stable_task_seed(2026, "finite_k", "cell", config_reordered)
    assert first == second
    assert first != stable_task_seed(2026, "finite_k", "cell", {**config_a, "K": 1000})
    assert first != stable_task_seed(2026, "frontier", "cell", config_a)

    encoded = canonical_configuration(config_a)
    assert json.loads(encoded)["alpha"] == 0.01


def test_rng_is_pcg64dxsm_and_reproducible() -> None:
    one = task_rng(17, "stage", "task", {"p": 0.01})
    two = task_rng(17, "stage", "task", {"p": 0.01})
    assert isinstance(one.bit_generator, np.random.PCG64DXSM)
    np.testing.assert_array_equal(one.integers(0, 2**31, size=20), two.integers(0, 2**31, size=20))
    assert isinstance(make_rng(3).bit_generator, np.random.PCG64DXSM)
