"""Simulate frame-sampling strategies against the every-frame reference cache (no GPU needed).

For each second we pick which frames a strategy would have run YOLO on, aggregate only those
frames with the brief's rules, and score against the reference built from all frames.

Strategies
  fixed:k        k evenly spaced frames per second (centred in the second)
  adapt:k:T      k coarse frames per second; if any trigger in T fires, run ALL frames of that second
                 triggers:  o = samples disagree on an object (0 < hits < k)
                            p = sampled people counts not all equal
                            b<d> = a sampled detection has conf within d of 0.4 (e.g. b0.1)
                            n = coarse result differs from the previous or next second's coarse result
                                (both seconds are refined)

Usage:
  python tools/simulate.py --ref-dir cache/ref --split train-00000      # tune set
  python tools/simulate.py --ref-dir cache/ref --split train-00001      # confirm set
  python tools/simulate.py --ref-dir cache/ref --strategies fixed:3 adapt:3:op
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score  # noqa: E402
from vidx import CLASS_NAMES, CONF, PERSON  # noqa: E402
from vidx.aggregate import aggregate, median_floor  # noqa: E402

NC = len(CLASS_NAMES)


@dataclass
class Video:
    vid: str
    sec: np.ndarray        # (n,) second per frame
    counts: np.ndarray     # (n, 5) detections >= CONF per class
    near: np.ndarray       # (n,) min |conf - CONF| over target-class detections in the frame (inf if none)
    bounds: np.ndarray     # frame index boundaries per second (sorted by sec)
    ref: dict


def load(ref_dir: Path, split: str | None) -> list[Video]:
    vids = []
    for npz in sorted(ref_dir.glob("*.npz")):
        if split and not npz.name.startswith(split):
            continue
        z = np.load(npz)
        if str(z["error"]):
            continue
        sec = z["sec"]
        n = len(sec)
        m = z["det_conf"] >= CONF
        counts = np.zeros((n, NC), np.int16)
        np.add.at(counts, (z["det_frame"][m], z["det_cls"][m].astype(int)), 1)
        near = np.full(n, np.inf)
        np.minimum.at(near, z["det_frame"], np.abs(z["det_conf"] - CONF))
        assert np.all(np.diff(sec) >= 0), "frames not in second order"
        bounds = np.searchsorted(sec, np.arange(int(sec.max()) + 2))
        ref = {s: (frozenset(o), p) for s, o, p in aggregate(sec, counts)}
        vids.append(Video(npz.stem.replace("__", ":"), sec, counts, near, bounds, ref))
    return vids


def pick(n: int, k: int) -> np.ndarray:
    """k evenly spaced frame offsets centred in a second of n frames."""
    if n <= k:
        return np.arange(n)
    return np.unique(((np.arange(k) + 0.5) * n / k).astype(int))


def decide(c: np.ndarray) -> tuple[frozenset, int]:
    m = len(c)
    if m == 0:
        return frozenset(), 0
    hits = (c > 0).sum(axis=0)
    objs = frozenset(CLASS_NAMES[j] for j in range(NC) if 2 * hits[j] >= m)
    return objs, median_floor(c[:, PERSON])


def ambiguous(c: np.ndarray, margin: float) -> bool:
    """After stage 2: still too close to call? object fraction near 0.5, or people median weakly supported."""
    m = len(c)
    frac = (c > 0).sum(axis=0) / m
    if np.any(np.abs(frac - 0.5) < margin):
        return True
    med = median_floor(c[:, PERSON])
    return (c[:, PERSON] == med).sum() / m < 0.5


def parse(strategy: str):
    """kind:k[:triggers][:rK[xM]]  e.g. adapt:2:op:r7x0.2 -> refine to 7 frames, then all if ambiguous(margin 0.2)."""
    parts = strategy.split(":")
    kind, k = parts[0], int(parts[1])
    trig = parts[2] if len(parts) > 2 else ""
    refine_k, margin = None, None
    if len(parts) > 3:
        r = parts[3][1:]
        if "x" in r:
            r, mg = r.split("x")
            margin = float(mg)
        refine_k = int(r)
    band = None
    if "b" in trig:
        i = trig.index("b")
        j = i + 1
        while j < len(trig) and (trig[j].isdigit() or trig[j] == "."):
            j += 1
        band = float(trig[i + 1:j])
        trig = trig[:i] + trig[j:]
    return kind, k, set(trig), band, refine_k, margin


def run_video(v: Video, strategy: str):
    kind, k, trig, band, refine_k, margin = parse(strategy)
    n_sec = len(v.bounds) - 1
    coarse, refine = [], np.zeros(n_sec, bool)
    used = 0
    for s in range(n_sec):
        a, b = v.bounds[s], v.bounds[s + 1]
        idx = a + pick(b - a, k)
        c = v.counts[idx]
        coarse.append(decide(c))
        used += len(idx)
        if kind == "adapt" and len(idx) < b - a:
            hits = (c > 0).sum(axis=0)
            if "o" in trig and np.any((hits > 0) & (hits < len(idx))):
                refine[s] = True
            if "p" in trig and c[:, PERSON].min() != c[:, PERSON].max():
                refine[s] = True
            if band is not None and np.any(v.near[idx] <= band):
                refine[s] = True
    if kind == "adapt" and "n" in trig:
        for s in range(n_sec - 1):
            if coarse[s] != coarse[s + 1]:
                refine[s] = refine[s + 1] = True
    pred = {}
    for s in range(n_sec):
        a, b = v.bounds[s], v.bounds[s + 1]
        if refine[s]:
            idx = set((a + pick(b - a, k)).tolist())
            if refine_k is not None and refine_k < b - a:
                idx2 = idx | set((a + pick(b - a, refine_k)).tolist())
                c2 = v.counts[sorted(idx2)]
                if margin is not None and ambiguous(c2, margin):
                    used += (b - a) - len(idx)
                    pred[s] = decide(v.counts[a:b])
                else:
                    used += len(idx2) - len(idx)
                    pred[s] = decide(c2)
            else:
                used += (b - a) - len(idx)
                pred[s] = decide(v.counts[a:b])
        else:
            pred[s] = coarse[s]
    return pred, used, int(refine.sum()), n_sec


def evaluate(videos: list[Video], strategy: str) -> dict:
    ref = {v.vid: v.ref for v in videos}
    pred, used, refined, secs = {}, 0, 0, 0
    total = sum(len(v.sec) for v in videos)
    for v in videos:
        p, u, r, n = run_video(v, strategy)
        pred[v.vid] = p
        used += u; refined += r; secs += n  # noqa: E702
    res = score(ref, pred)
    pc = {c: f for c, f in res["per_class_f1"].items() if f is not None}
    return {
        "strategy": strategy,
        "frames_pct": 100 * used / total,
        "refined_pct": 100 * refined / max(secs, 1),
        "f1_pooled": res["f1"],
        "f1_min_class": min(pc.values()) if pc else 1.0,
        "worst_class": min(pc, key=pc.get) if pc else "-",
        "exact": res["exact"],
        "within1": res["within1"],
        "per_class": pc,
    }


DEFAULT = (
    [f"fixed:{k}" for k in (1, 2, 3, 5, 7, 10)]
    + [f"adapt:{k}:op" for k in (2, 3, 5)]
    + [f"adapt:{k}:opn" for k in (1, 2, 3, 5)]
    + [f"adapt:{k}:opb0.05" for k in (2, 3, 5)]
    + [f"adapt:{k}:opb0.1" for k in (2, 3, 5)]
    + [f"adapt:{k}:opnb0.05" for k in (1, 2, 3)]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="cache/ref")
    ap.add_argument("--split", default=None, help="only videos whose cache name starts with this")
    ap.add_argument("--strategies", nargs="+", default=DEFAULT)
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args()
    videos = load(Path(args.ref_dir), args.split)
    print(f"{len(videos)} videos, {sum(len(v.sec) for v in videos)} frames, "
          f"{sum(len(v.ref) for v in videos)} seconds")
    hdr = f"{'strategy':22s} {'frames%':>8s} {'refine%':>8s} {'F1pool':>7s} {'F1min':>7s} {'worst':>10s} " \
          f"{'exact':>7s} {'±1':>7s}  per-class F1 (person car dog laptop phone)"
    print(hdr)
    out = []
    for s in args.strategies:
        r = evaluate(videos, s)
        out.append(r)
        pcs = " ".join(f"{r['per_class'].get(c, float('nan')):.3f}" for c in CLASS_NAMES)
        print(f"{s:22s} {r['frames_pct']:8.1f} {r['refined_pct']:8.1f} {r['f1_pooled']:7.4f} "
              f"{r['f1_min_class']:7.4f} {r['worst_class']:>10s} {r['exact']:7.4f} {r['within1']:7.4f}  {pcs}",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
