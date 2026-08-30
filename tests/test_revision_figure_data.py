"""Schema and manifest contracts for the major-revision figure pipeline.

These tests deliberately validate data contracts without requiring notebook
execution or rendered PDF/PNG files.  Formal artifact existence belongs to
``acli.revision.validation`` after the paper notebook has run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from acli.revision.suite import STAGES
from acli.revision.validation import METADATA_COLUMNS, required_tables


ROOT = Path(__file__).resolve().parents[1]


def _columns(*names: str) -> frozenset[str]:
    return frozenset((*names, *METADATA_COLUMNS))


# Minimal stable contracts: experiment tables may add compatibility aliases,
# but removing any of these fields would break a paper figure, numerical
# audit, or provenance check.
REQUIRED_SCHEMAS: dict[str, frozenset[str]] = {
    "result/table/05_finite_length_simulation_validation.csv": _columns(
        "scenario",
        "family",
        "target_auc",
        "realized_auc",
        "epsilon",
        "intercept",
        "calibration_error",
        "p",
        "J_bits",
        "K",
        "B",
        "alpha",
        "mc_gain_bits",
        "mc_gain_se_bits",
        "exact_finite_K_precision",
        "exact_finite_K_gain_bits",
        "large_K_top_tail_precision",
        "large_K_top_tail_gain_bits",
        "sharp_binary_KL_precision",
        "two_branch_pinsker_precision",
        "original_pinsker_jakob_raw_precision",
        "original_pinsker_jakob_raw_gain_bits",
        "original_pinsker_jakob_clipped_precision",
        "original_pinsker_jakob_clipped_gain_bits",
        "original_pinsker_jakob_pool_clipped_gain_bits",
        "finite_pool_oracle_gain_bits",
        "baseline_random_gain_bits",
    ),
    "result/table/06_accumulation_profiles.csv": _columns(
        "curve_type",
        "pi",
        "beta",
        "dimension",
        "n",
        "phi_bits",
        "delta_bits",
        "entropy_ceiling_bits",
    ),
    "result/table/06_redundancy_representative_cases.csv": _columns(
        "case",
        "budget",
        "beta",
        "selected_probabilities",
        "mean_hit_probability",
        "expected_hits",
        "exact_gain_bits",
        "binomial_approximation_gain_bits",
        "binomial_approximation_abs_error",
        "decoupled_gain_bits",
    ),
    "result/table/06_accumulation_stress_summary.csv": _columns(
        "random_channels",
        "bruteforce_instances",
        "random_channel_seed",
        "transcript_seed",
        "channel_failure_count",
        "max_monotonicity_violation",
        "max_concavity_violation",
        "max_entropy_violation",
        "max_origin_error",
        "max_transcript_identity_error",
        "worst_channel_case",
        "worst_transcript_case",
    ),
    "result/table/07_frontier_comparison.csv": _columns(
        "p",
        "alpha",
        "J_bits",
        "q_feasible_lower",
        "q_feasible_upper",
        "q_star",
        "q_star_information_bits",
        "q_star_residual",
        "ceiling_information_bits",
        "ceiling_hit",
        "q_one_branch_raw",
        "q_one_branch_clipped",
        "q_two_branch_pinsker_raw",
        "q_two_branch_pinsker_clipped",
    ),
    "result/table/07_same_J_tail_comparison.csv": _columns(
        "family",
        "p",
        "alpha",
        "target_J_bits",
        "actual_J_bits",
        "J_abs_mismatch_bits",
        "epsilon",
        "calibration_error",
        "AUC",
        "q_alpha",
        "J_tail_bits",
        "J_tail_over_J",
        "q_star",
        "tail_boost_fraction_of_sharp",
    ),
    "result/table/07_constant_gap.csv": _columns(
        "p",
        "alpha",
        "J_bits",
        "q_star",
        "q_one_branch_clipped",
        "q_two_branch_pinsker_clipped",
        "sharp_boost",
        "one_branch_additive_gap",
        "two_branch_additive_gap",
    ),
    "result/table/07_tail_metric_summary.csv": _columns(
        "family",
        "p",
        "actual_J_bits",
        "AUC",
        "mean_J_tail_over_J",
        "median_J_tail_over_J",
        "mean_tail_boost_fraction_of_sharp",
        "tail_coverage_rank",
    ),
    "result/table/08_finite_K_convergence.csv": _columns(
        "p",
        "family",
        "J_fraction_h2",
        "target_J_bits",
        "actual_J_bits",
        "epsilon",
        "calibration_error",
        "alpha",
        "K",
        "B",
        "realized_alpha",
        "q_alpha_fraction",
        "quadrature_order",
        "q_K_B",
        "q_alpha",
        "relative_boost_error",
    ),
    "result/table/08_finite_K_summary.csv": _columns(
        "K",
        "n_configurations",
        "median_relative_boost_error",
        "p90_relative_boost_error",
        "max_relative_boost_error",
        "median_loglog_slope",
        "mc_audit_configuration",
        "mc_audit_seed",
        "mc_audit_reps",
        "mc_audit_deterministic_precision",
        "mc_audit_estimate_precision",
        "mc_audit_se",
        "mc_audit_abs_difference",
        "mc_audit_within_3se",
    ),
    "result/table/09_weak_screening_validity.csv": _columns(
        "family",
        "p",
        "alpha",
        "J_fraction_h2",
        "target_J_bits",
        "actual_J_bits",
        "epsilon",
        "intercept",
        "realized_auc",
        "tail_locality",
        "exact_q_alpha",
        "weak_epsilon_q",
        "weak_J_q",
        "relative_boost_error_epsilon",
        "relative_boost_error_J",
        "calibration_error",
        "software",
    ),
    "result/table/09_calibration_audit.csv": _columns(
        "source",
        "family",
        "p",
        "target_metric",
        "target_value",
        "target_J_bits",
        "actual_J_bits",
        "J_fraction_h2",
        "target_auc",
        "realized_auc",
        "epsilon",
        "intercept",
        "mean_eta",
        "calibration_error",
        "software",
    ),
    "result/table/09_auc_epsilon_J_mapping.csv": _columns(
        "mapping_kind",
        "family",
        "p",
        "target_auc",
        "realized_auc",
        "epsilon",
        "intercept",
        "J_bits",
        "J_fraction_h2",
        "mean_eta",
        "calibration_error",
        "software",
    ),
}


_TEXT_COLUMNS = {
    "case",
    "configuration",
    "curve_type",
    "family",
    "git_commit",
    "mapping_kind",
    "method",
    "scenario",
    "selected_probabilities",
    "software",
    "source",
    "target_metric",
    "worst_channel_case",
    "worst_transcript_case",
}


def _synthetic_frame(required: frozenset[str]) -> pd.DataFrame:
    row: dict[str, object] = {}
    for column in required:
        if column == "configuration":
            row[column] = "{}"
        elif column in {"selected_probabilities", "worst_channel_case", "worst_transcript_case"}:
            row[column] = "[]" if column == "selected_probabilities" else "{}"
        elif column in _TEXT_COLUMNS:
            row[column] = f"synthetic_{column}"
        elif column == "seed":
            row[column] = 7
        else:
            row[column] = 0.25
    return pd.DataFrame([row])


def _assert_table_contract(
    frame: pd.DataFrame,
    required: frozenset[str],
    *,
    label: str,
) -> None:
    assert not frame.empty, f"{label} must contain at least one row"
    missing = sorted(required.difference(frame.columns))
    assert not missing, f"{label} is missing required columns: {missing}"
    assert not frame.loc[:, sorted(required)].isna().any(axis=None), (
        f"{label} contains missing values in required columns"
    )

    for column in METADATA_COLUMNS:
        values = frame[column].astype(str).str.strip()
        assert values.ne("").all(), f"{label}:{column} must be nonempty"

    numeric = frame.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all(), (
        f"{label} contains non-finite numeric values"
    )


def test_paper_figure_manifest_maps_exactly_figures_one_through_eight():
    path = ROOT / "paper_figures/manifest.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    figures = payload.get("figures")
    assert isinstance(figures, list)
    assert [item["paper_figure"] for item in figures] == [
        f"Fig. {number}" for number in range(1, 9)
    ]

    stems = [str(item["output"]) for item in figures]
    assert len(stems) == len(set(stems)) == 8
    assert all(
        stem and Path(stem).name == stem and not Path(stem).suffix for stem in stems
    )


@pytest.mark.parametrize(
    ("relative", "required"),
    REQUIRED_SCHEMAS.items(),
    ids=lambda value: Path(value).name if isinstance(value, str) else None,
)
def test_each_schema_contract_accepts_a_small_valid_synthetic_frame(relative, required):
    _assert_table_contract(_synthetic_frame(required), required, label=relative)


def test_schema_contract_rejects_missing_or_nonfinite_required_data():
    relative, required = next(iter(REQUIRED_SCHEMAS.items()))
    valid = _synthetic_frame(required)
    missing = valid.drop(columns=["method"])
    with pytest.raises(AssertionError, match="missing required columns"):
        _assert_table_contract(missing, required, label=relative)

    nonfinite = valid.copy()
    nonfinite.loc[0, "alpha"] = np.inf
    with pytest.raises(AssertionError, match="non-finite numeric values"):
        _assert_table_contract(nonfinite, required, label=relative)


def test_generated_revision_csvs_obey_contract_when_present():
    for relative, required in REQUIRED_SCHEMAS.items():
        path = ROOT / relative
        if not path.exists():
            continue
        assert path.is_file() and path.stat().st_size > 0, f"empty generated table: {relative}"
        _assert_table_contract(pd.read_csv(path), required, label=relative)
    # It is valid for a clean checkout to have no generated tables.  The
    # synthetic-frame tests above still exercise every schema in that case.


def test_suite_stage_mapping_contains_every_required_csv_exactly_once():
    mapped = [relative for _, _, outputs in STAGES for relative in outputs]
    expected = list(REQUIRED_SCHEMAS)
    assert len(mapped) == len(set(mapped)), "suite stage outputs must not be duplicated"
    assert set(mapped) == set(expected)
    assert required_tables() == mapped

    stage_names = [name for name, _, _ in STAGES]
    assert stage_names == [
        "finite_length_validation",
        "shared_target_accumulation",
        "sharp_frontier_and_tail_metrics",
        "finite_K_convergence",
        "weak_screening_validity",
    ]
