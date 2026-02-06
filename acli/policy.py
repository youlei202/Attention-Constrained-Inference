"""Verification policies.

Policies map screening scores to a set of indices to verify under budget B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

import numpy as np


class Policy:
    """Abstract policy."""

    def select(self, scores: np.ndarray, B: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return indices of selected records (length B)."""
        raise NotImplementedError


@dataclass(frozen=True)
class TopBPolicy(Policy):
    """Verify the top-B records by score."""

    def select(self, scores: np.ndarray, B: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if B <= 0:
            return np.array([], dtype=np.int64)
        if B >= len(scores):
            return np.arange(len(scores), dtype=np.int64)
        idx = np.argpartition(scores, -B)[-B:]
        # optional: return in descending score order for determinism
        idx = idx[np.argsort(scores[idx])[::-1]]
        return idx.astype(np.int64)


@dataclass(frozen=True)
class RandomPolicy(Policy):
    """Verify B records uniformly at random (baseline)."""

    def select(self, scores: np.ndarray, B: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if B <= 0:
            return np.array([], dtype=np.int64)
        if B >= len(scores):
            return np.arange(len(scores), dtype=np.int64)
        return rng.choice(len(scores), size=B, replace=False).astype(np.int64)
