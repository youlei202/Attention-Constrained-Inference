import math

import pytest

from acli.revision.frontier import (
    binary_selection_information,
    feasible_q_bounds,
    q0_from_q,
    q_one_branch,
    q_one_branch_raw,
    q_pinsker_clipped,
    q_pinsker_raw,
    q_star,
)


@pytest.mark.parametrize(
    ("p", "alpha", "expected_upper"),
    [
        (0.1, 0.02, 1.0),
        (0.01, 0.1, 0.1),
        (0.1, 0.1, 1.0),
    ],
)
def test_feasible_bounds_are_exact(p, alpha, expected_upper):
    assert feasible_q_bounds(p, alpha) == pytest.approx(
        (p, expected_upper), rel=2e-16, abs=0.0
    )


def test_q0_mass_conservation_and_ceiling_branch():
    p = 0.01
    alpha = 0.1
    q = 0.075
    q0 = q0_from_q(q, p, alpha)
    assert alpha * q + (1.0 - alpha) * q0 == pytest.approx(p, abs=2e-16)
    assert q0_from_q(p / alpha, p, alpha) == pytest.approx(0.0, abs=2e-16)


def test_binary_selection_information_is_zero_only_at_prior():
    p = 0.03
    alpha = 0.2
    assert binary_selection_information(p, p, alpha) == pytest.approx(0.0, abs=2e-16)
    assert binary_selection_information(0.08, p, alpha) > 0.0


@pytest.mark.parametrize("p", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("alpha", [1e-4, 0.003, 0.03, 0.3])
@pytest.mark.parametrize("fraction", [1e-5, 1e-3, 0.1, 0.7])
def test_q_star_inverts_psi_or_hits_ceiling(p, alpha, fraction):
    lower, upper = feasible_q_bounds(p, alpha)
    ceiling = binary_selection_information(upper, p, alpha)
    J = fraction * ceiling
    q = q_star(p, alpha, J)
    assert lower <= q <= upper
    assert abs(binary_selection_information(q, p, alpha) - J) <= 2e-10


def test_q_star_clips_when_information_exceeds_feasibility_ceiling():
    p = 0.01
    alpha = 0.1
    upper = feasible_q_bounds(p, alpha)[1]
    ceiling = binary_selection_information(upper, p, alpha)
    assert q_star(p, alpha, ceiling) == upper
    assert q_star(p, alpha, ceiling + 0.25) == upper


def test_pinsker_variants_and_legacy_one_branch_are_distinct():
    p = 0.02
    alpha = 0.1
    J = 0.005
    expected_one_branch = p + math.sqrt(math.log(2.0) * J / (2.0 * alpha))
    expected_two_branch = p + math.sqrt(
        math.log(2.0) * J * (1.0 - alpha) / (2.0 * alpha)
    )
    assert q_one_branch_raw(p, alpha, J) == pytest.approx(expected_one_branch)
    assert q_one_branch(p, alpha, J, clip=False) == pytest.approx(expected_one_branch)
    assert q_pinsker_raw(p, alpha, J) == pytest.approx(expected_two_branch)
    assert q_pinsker_raw(p, alpha, J) < q_one_branch_raw(p, alpha, J)


def test_pinsker_and_one_branch_default_clip_to_feasibility():
    p = 0.01
    alpha = 0.2
    J = 10.0
    ceiling = feasible_q_bounds(p, alpha)[1]
    assert q_one_branch(p, alpha, J) == ceiling
    assert q_pinsker_clipped(p, alpha, J) == ceiling
    assert q_pinsker_raw(p, alpha, J) > ceiling


@pytest.mark.parametrize(
    "call",
    [
        lambda: feasible_q_bounds(0.0, 0.1),
        lambda: feasible_q_bounds(0.1, 1.0),
        lambda: q0_from_q(0.09, 0.1, 0.2),
        lambda: q_star(0.1, 0.2, -1e-4),
    ],
)
def test_invalid_frontier_inputs_fail_loudly(call):
    with pytest.raises(ValueError):
        call()
