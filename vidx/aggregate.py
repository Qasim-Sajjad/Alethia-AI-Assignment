"""Per-second aggregation rules from the brief (section 5).

- object present in a second  <=>  found in at least half of that second's frames
- people = median of per-frame person counts, rounded down at a .5 tie
- one line per second from 0 up to and including the second of the last frame;
  a second with no frames gets no objects and 0 people
"""
from __future__ import annotations

import numpy as np

from . import CLASS_NAMES, PERSON


def median_floor(values: np.ndarray) -> int:
    v = np.sort(values)
    n = len(v)
    if n == 0:
        return 0
    if n % 2:
        return int(v[n // 2])
    return int((int(v[n // 2 - 1]) + int(v[n // 2])) // 2)


def aggregate(sec: np.ndarray, counts: np.ndarray) -> list[tuple[int, list[str], int]]:
    """sec: (n,) int second index per frame; counts: (n, 5) detections per class per frame.

    Returns [(second, objects, people), ...] for every second 0..max(sec).
    """
    if len(sec) == 0:
        return []
    out = []
    last = int(sec.max())
    order = np.argsort(sec, kind="stable")
    sec_sorted = sec[order]
    counts_sorted = counts[order]
    bounds = np.searchsorted(sec_sorted, np.arange(last + 2))
    for s in range(last + 1):
        c = counts_sorted[bounds[s]:bounds[s + 1]]
        n = len(c)
        if n == 0:
            out.append((s, [], 0))
            continue
        hits = (c > 0).sum(axis=0)
        objs = [CLASS_NAMES[k] for k in range(len(CLASS_NAMES)) if 2 * hits[k] >= n]
        out.append((s, objs, median_floor(c[:, PERSON])))
    return out
