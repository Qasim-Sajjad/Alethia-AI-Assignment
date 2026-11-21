"""Per-frame agreement between a pipeline dump (e.g. FP16 / TensorRT) and the FP32 every-frame reference.

For every frame the pipeline actually ran, compare the per-class counts at conf >= 0.4 with the
reference counts for the same frame, and the confidence shift of matched detections.

Usage: python tools/compare_dumps.py --ref-dir cache/ref --dump-dir cache/fast_fp16_dump
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx import CLASS_NAMES, CONF  # noqa: E402


def counts(n, f, c, cf):
    m = cf >= CONF
    out = np.zeros((n, len(CLASS_NAMES)), np.int16)
    np.add.at(out, (f[m], c[m].astype(int)), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="cache/ref")
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()
    frames = diff_frames = 0
    cls_diff = np.zeros(len(CLASS_NAMES), int)
    shifts = []
    for d in sorted(Path(args.dump_dir).glob("*.npz")):
        z = np.load(d)
        r = np.load(Path(args.ref_dir) / d.name)
        n = len(r["sec"])
        ran = z["ran"]
        a = counts(n, z["det_frame"], z["det_cls"], z["det_conf"])[ran]
        b = counts(n, r["det_frame"], r["det_cls"], r["det_conf"])[ran]
        frames += len(ran)
        neq = a != b
        diff_frames += int(neq.any(axis=1).sum())
        cls_diff += neq.any(axis=0) * 0 + neq.sum(axis=0)
        # confidence shift: per frame and class, compare sorted confidences when counts at 0.25 match
        for fi in ran[:: max(len(ran) // 200, 1)]:
            for k in range(len(CLASS_NAMES)):
                x = np.sort(z["det_conf"][(z["det_frame"] == fi) & (z["det_cls"] == k)])
                y = np.sort(r["det_conf"][(r["det_frame"] == fi) & (r["det_cls"] == k)])
                if len(x) == len(y) and len(x):
                    shifts.append(np.abs(x - y))
    s = np.concatenate(shifts) if shifts else np.zeros(1)
    print(f"frames compared: {frames}")
    print(f"frames whose 0.4-counts differ from reference: {diff_frames} ({100 * diff_frames / max(frames, 1):.2f}%)")
    print("per-class count differences:", dict(zip(CLASS_NAMES, cls_diff.tolist())))
    print(f"|conf shift| of matched detections: median {np.median(s):.4f}, p95 {np.percentile(s, 95):.4f}, "
          f"p99 {np.percentile(s, 99):.4f}, max {s.max():.4f}")


if __name__ == "__main__":
    main()
