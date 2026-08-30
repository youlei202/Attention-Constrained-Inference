"""Stable task-level randomness for reproducible experiment sweeps."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


_HASH_DOMAIN = "acli.revision.task-seed.v1"


def _json_compatible(value: Any) -> Any:
    """Convert common scientific-Python configuration values to JSON data."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("configuration mapping keys must be strings")
            converted[key] = _json_compatible(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted_items = [_json_compatible(item) for item in value]
        return sorted(
            converted_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("configuration floats must be finite")
        return value
    raise TypeError(f"configuration value of type {type(value).__name__} is not supported")


def canonical_configuration(configuration: Any) -> str:
    """Serialize a complete task configuration in a stable canonical form."""

    return json.dumps(
        _json_compatible(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_task_seed(
    master_seed: int,
    stage: str,
    task: str,
    configuration: Any,
) -> int:
    """Derive a deterministic 128-bit task seed with SHA-256.

    The derivation includes the master seed, stage, task name, and full
    canonical parameter configuration.  It intentionally does not use
    Python's process-randomized :func:`hash`.
    """

    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise TypeError("master_seed must be an integer")
    if int(master_seed) < 0:
        raise ValueError("master_seed must be nonnegative")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a nonempty string")

    payload = canonical_configuration(
        {
            "domain": _HASH_DOMAIN,
            "master_seed": int(master_seed),
            "stage": stage,
            "task": task,
            "configuration": configuration,
        }
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


# Friendly aliases used by experiment scripts and external callers.
derive_task_seed = stable_task_seed
task_seed = stable_task_seed


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy ``Generator`` backed specifically by PCG64DXSM."""

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")
        if int(seed) < 0:
            raise ValueError("seed must be nonnegative")
        seed = int(seed)
    return np.random.Generator(np.random.PCG64DXSM(seed))


def task_rng(
    master_seed: int,
    stage: str,
    task: str,
    configuration: Any,
) -> np.random.Generator:
    """Construct the PCG64DXSM generator for one fully specified task."""

    return make_rng(stable_task_seed(master_seed, stage, task, configuration))


def task_reproducibility_fields(
    master_seed: int,
    stage: str,
    task: str,
    configuration: Any,
) -> Mapping[str, Any]:
    """Return seed/configuration fields suitable for result-table rows."""

    serialized = canonical_configuration(configuration)
    return {
        "seed": stable_task_seed(master_seed, stage, task, configuration),
        "configuration": serialized,
    }


__all__ = [
    "canonical_configuration",
    "stable_task_seed",
    "derive_task_seed",
    "task_seed",
    "make_rng",
    "task_rng",
    "task_reproducibility_fields",
]
