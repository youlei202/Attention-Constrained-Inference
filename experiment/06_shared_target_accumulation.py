#!/usr/bin/env python3
"""Shared-target accumulation profiles and exact identity stress audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.accumulation import (
    component_target_profile,
    decoupled_profile,
    expected_profile_poisson_binomial,
    phi_bsc_profile,
    profile_margins,
)
from acli.revision.information import h2
from acli.revision.reproducibility import canonical_configuration, stable_task_seed
from acli.revision.stress import random_iid_channel_stress, random_transcript_identity_stress
from acli.revision.suite import git_commit, load_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def provenance(master_seed: int, task: str, configuration: dict, method: str) -> dict:
    return {
        "seed": stable_task_seed(master_seed, "06_accumulation", task, configuration),
        "configuration": canonical_configuration(configuration),
        "git_commit": git_commit(ROOT),
        "method": method,
    }


def profile_rows(config: dict) -> list[dict]:
    master_seed = int(config["master_seed"])
    n_max = int(config["max_n"])
    rows: list[dict] = []
    for pi in (0.5, 0.2, 0.05):
        for beta in (0.05, 0.10, 0.20, 0.30, 0.40):
            task_config = {"curve_type": "shared_bsc", "pi": pi, "beta": beta, "n_max": n_max}
            profile = phi_bsc_profile(n_max, pi, beta)
            margins = np.concatenate(([0.0], profile_margins(profile)))
            meta = provenance(master_seed, "shared_bsc", task_config, "exact_count_sufficient_statistic")
            for n, (value, margin) in enumerate(zip(profile, margins)):
                rows.append(
                    {
                        "curve_type": "shared_bsc",
                        "pi": pi,
                        "beta": beta,
                        "dimension": 1,
                        "n": n,
                        "phi_bits": value,
                        "delta_bits": margin,
                        "entropy_ceiling_bits": float(h2(pi)),
                        **meta,
                    }
                )

    component_beta = 0.10
    for dimension in (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048):
        task_config = {
            "curve_type": "component_target",
            "dimension": dimension,
            "beta": component_beta,
            "n_max": n_max,
        }
        profile = component_target_profile(n_max, dimension, component_beta)
        margins = np.concatenate(([0.0], profile_margins(profile)))
        meta = provenance(master_seed, "component_target", task_config, "exact_binomial_occupancy")
        for n, (value, margin) in enumerate(zip(profile, margins)):
            rows.append(
                {
                    "curve_type": "component_target",
                    "pi": 0.5,
                    "beta": component_beta,
                    "dimension": dimension,
                    "n": n,
                    "phi_bits": value,
                    "delta_bits": margin,
                    "entropy_ceiling_bits": float(dimension),
                    **meta,
                }
            )

    i_ver = 1.0 - float(h2(component_beta))
    task_config = {"curve_type": "decoupled", "i_ver": i_ver, "n_max": n_max}
    profile = decoupled_profile(n_max, i_ver)
    margins = np.concatenate(([0.0], profile_margins(profile)))
    meta = provenance(master_seed, "decoupled", task_config, "exact_linear_profile")
    for n, (value, margin) in enumerate(zip(profile, margins)):
        rows.append(
            {
                "curve_type": "decoupled",
                "pi": 0.5,
                "beta": component_beta,
                "dimension": 0,
                "n": n,
                "phi_bits": value,
                "delta_bits": margin,
                "entropy_ceiling_bits": float(n_max * i_ver),
                **meta,
            }
        )
    return rows


def redundancy_rows(config: dict) -> list[dict]:
    master_seed = int(config["master_seed"])
    beta = 0.1
    budget = 8
    phi = phi_bsc_profile(budget, 0.5, beta)
    cases = {
        "homogeneous": np.full(budget, 0.18),
        "heterogeneous": np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.60, 0.90]),
        "concentrated": np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.08, 0.80, 1.0]),
    }
    rows: list[dict] = []
    for name, probabilities in cases.items():
        exact_gain = expected_profile_poisson_binomial(phi, probabilities)
        mean_probability = float(np.mean(probabilities))
        approximation_pmf = binom.pmf(np.arange(budget + 1), budget, mean_probability)
        approximation_gain = float(np.dot(phi, approximation_pmf))
        task_config = {
            "case": name,
            "beta": beta,
            "probabilities": probabilities.tolist(),
        }
        rows.append(
            {
                "case": name,
                "budget": budget,
                "beta": beta,
                "selected_probabilities": canonical_configuration(probabilities.tolist()),
                "mean_hit_probability": mean_probability,
                "expected_hits": float(np.sum(probabilities)),
                "exact_gain_bits": exact_gain,
                "binomial_approximation_gain_bits": approximation_gain,
                "binomial_approximation_abs_error": abs(approximation_gain - exact_gain),
                "decoupled_gain_bits": float(np.sum(probabilities) * (1.0 - h2(beta))),
                **provenance(
                    master_seed,
                    "redundancy_case",
                    task_config,
                    "exact_poisson_binomial_with_labeled_binomial_approximation",
                ),
            }
        )
    return rows


def stress_rows(config: dict) -> list[dict]:
    master_seed = int(config["master_seed"])
    n_channels = int(config["random_channels"])
    n_instances = int(config["bruteforce_instances"])
    channel_configuration = {"n_channels": n_channels, "n_max": 6, "max_states": 4, "max_outputs": 3}
    transcript_configuration = {"n_instances": n_instances, "max_budget": 5, "max_states": 3, "max_outputs": 3}
    channel_seed = stable_task_seed(master_seed, "06_accumulation", "random_channel_stress", channel_configuration)
    transcript_seed = stable_task_seed(master_seed, "06_accumulation", "transcript_identity_stress", transcript_configuration)
    channel = random_iid_channel_stress(n_channels, n_max=6, seed=channel_seed)
    transcript = random_transcript_identity_stress(n_instances, max_budget=5, seed=transcript_seed)
    combined_configuration = {
        "random_channel_stress": channel_configuration,
        "transcript_identity_stress": transcript_configuration,
    }
    return [
        {
            "random_channels": n_channels,
            "bruteforce_instances": n_instances,
            "random_channel_seed": channel_seed,
            "transcript_seed": transcript_seed,
            "channel_failure_count": int(channel["failure_count"]),
            "max_monotonicity_violation": float(channel["max_monotonicity_violation"]),
            "max_concavity_violation": float(channel["max_concavity_violation"]),
            "max_entropy_violation": float(channel["max_entropy_ceiling_violation"]),
            "max_origin_error": float(channel["max_origin_abs_error"]),
            "max_transcript_identity_error": float(transcript["max_absolute_error"]),
            "worst_channel_case": json.dumps(channel["worst_case"], sort_keys=True, separators=(",", ":")),
            "worst_transcript_case": json.dumps(transcript["worst_case"], sort_keys=True, separators=(",", ":")),
            **provenance(master_seed, "stress_summary", combined_configuration, "exact_enumeration_stress"),
        }
    ]


def main() -> None:
    args = parse_args()
    config, _ = load_profile(args.profile)
    outputs = ROOT / "result/table"
    print(f"[06] profile={config['profile']} max_n={config['max_n']}", flush=True)
    atomic_csv(pd.DataFrame(profile_rows(config)), outputs / "06_accumulation_profiles.csv")
    atomic_csv(pd.DataFrame(redundancy_rows(config)), outputs / "06_redundancy_representative_cases.csv")
    atomic_csv(pd.DataFrame(stress_rows(config)), outputs / "06_accumulation_stress_summary.csv")
    print("[06] accumulation outputs complete", flush=True)


if __name__ == "__main__":
    main()
