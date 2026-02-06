"""Utilities: seeding, device selection, small math helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Set as many seeds as possible for strict reproducibility.

    This sets:
    - Python random
    - numpy RNG
    - torch RNG (if torch is installed)
    and configures deterministic torch behavior when possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device(prefer_cuda: bool = True) -> str:
    """Return a compute device string.

    - If torch is available and CUDA is available and prefer_cuda=True -> "cuda"
    - Else -> "cpu"
    """
    if not prefer_cuda:
        return "cpu"
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def h2(x: float) -> float:
    """Binary entropy in bits."""
    x = min(max(x, 1e-12), 1 - 1e-12)
    return float(-(x * np.log2(x) + (1 - x) * np.log2(1 - x)))


@dataclass(frozen=True)
class RunMeta:
    seed: int
    device: str
    timestamp: str
    git_commit: Optional[str] = None

    @staticmethod
    def now(seed: int, device: str) -> "RunMeta":
        import datetime

        return RunMeta(
            seed=seed,
            device=device,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            git_commit=_try_get_git_commit(),
        )


def _try_get_git_commit() -> Optional[str]:
    """Try to read current git commit hash if repo is a git checkout."""
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None
