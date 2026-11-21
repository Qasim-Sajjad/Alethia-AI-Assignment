"""Adaptive per-second frame sampling (decided in specs/10-sampling.md: adapt:2:opb0.05:r7x0.15).

Shared by the pipeline and tools/simulate.py so both use identical logic.

Stage 1: K1 evenly spaced frames. If they disagree on any object or on the people count, or any
         target-class detection has confidence within BAND of 0.4 -> stage 2.
Stage 2: union with K2 evenly spaced frames. If an object's frame fraction is within MARGIN of 0.5,
         or the people median is held by fewer than half of the frames -> stage 3.
Stage 3: all frames of the second.
"""
from __future__ import annotations

import numpy as np

from . import CLASS_NAMES, PERSON
from .aggregate import median_floor

NC = len(CLASS_NAMES)
K1, K2, BAND, MARGIN = 2, 7, 0.05, 0.15


def pick(n: int, k: int) -> np.ndarray:
    """k evenly spaced frame offsets centred in a second of n frames."""
    if n <= k:
        return np.arange(n)
    return np.unique(((np.arange(k) + 0.5) * n / k).astype(int))


def decide(c: np.ndarray) -> tuple[frozenset, int]:
    """Brief's rules applied to the per-frame counts c (m, 5) of the frames that were run."""
    m = len(c)
    if m == 0:
        return frozenset(), 0
    hits = (c > 0).sum(axis=0)
    objs = frozenset(CLASS_NAMES[j] for j in range(NC) if 2 * hits[j] >= m)
    return objs, median_floor(c[:, PERSON])


def stage1_uncertain(c: np.ndarray, near: np.ndarray, band: float | None = BAND,
                     use_o: bool = True, use_p: bool = True) -> bool:
    hits = (c > 0).sum(axis=0)
    if use_o and np.any((hits > 0) & (hits < len(c))):
        return True
    if use_p and c[:, PERSON].min() != c[:, PERSON].max():
        return True
    if band is not None and np.any(near <= band):
        return True
    return False


def ambiguous(c: np.ndarray, margin: float = MARGIN) -> bool:
    """After stage 2: still too close to call?"""
    m = len(c)
    frac = (c > 0).sum(axis=0) / m
    if np.any(np.abs(frac - 0.5) < margin):
        return True
    med = median_floor(c[:, PERSON])
    return (c[:, PERSON] == med).sum() / m < 0.5


def stage2_frames(n: int) -> np.ndarray:
    return np.union1d(pick(n, K1), pick(n, K2))
