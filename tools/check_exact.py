"""Exactness checks for the fast path against the reference path.

1. vidx.letterbox.letterbox == ultralytics LetterBox(auto=True) for many shapes (pixel-equal).
2. Detector.infer on frames letterboxed by us == cached reference detections (from model.predict)
   for the first N frames of a video.

Usage: python tools/check_exact.py --ref-dir cache/ref --parquet finevideo/data/train-00000-of-01357.parquet --row 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx import CONF  # noqa: E402
from vidx.io import decode_frames  # noqa: E402
from vidx.letterbox import letterbox  # noqa: E402


def check_letterbox():
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(640, auto=True, stride=32)
    rng = np.random.default_rng(0)
    shapes = [(360, 640), (480, 270), (360, 202), (640, 360), (360, 480), (360, 626), (272, 480), (236, 416),
              (360, 526), (360, 204), (1080, 1920), (720, 1280), (1920, 1080), (101, 333), (17, 900), (640, 640)]
    bad = 0
    for h, w in shapes:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        a, b = letterbox(img), lb(image=img)
        if a.shape != b.shape or not np.array_equal(a, b):
            bad += 1
            print(f"  MISMATCH {h}x{w}: ours {a.shape} vs ultralytics {b.shape}")
    print(f"letterbox: {len(shapes) - bad}/{len(shapes)} shapes pixel-identical")


def check_detector(ref_dir: Path, parquet: Path, row: int, n: int, batch: int):
    from vidx.detector import Detector

    det = Detector("weights/yolo11l.pt")
    z = np.load(ref_dir / f"{parquet.name}__{row}.npz")
    mp4 = pq.read_table(parquet, columns=["mp4"]).column(0)[row].as_py()
    frames = []
    for i, (_, img) in enumerate(decode_frames(mp4)):
        if i >= n:
            break
        frames.append(letterbox(img))
    got = []
    for a in range(0, len(frames), batch):
        got += det.infer(np.stack(frames[a:a + batch]))
    same_counts = 0
    max_dconf = 0.0
    for fi, (cls, cf) in enumerate(got):
        m = z["det_frame"] == fi
        rc, rf = z["det_cls"][m], z["det_conf"][m]
        a = np.bincount(cls[cf >= CONF], minlength=5)
        b = np.bincount(rc[rf >= CONF], minlength=5)
        same_counts += int(np.array_equal(a, b))
        if len(cf) == len(rf):
            oa, ob = np.sort(cf), np.sort(rf)
            max_dconf = max(max_dconf, float(np.abs(oa - ob).max()) if len(oa) else 0.0)
        else:
            print(f"  frame {fi}: {len(cf)} dets vs ref {len(rf)}")
    print(f"detector (batch {batch}): {same_counts}/{len(got)} frames with identical counts at 0.4; "
          f"max |conf diff| {max_dconf:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="cache/ref")
    ap.add_argument("--parquet", default="finevideo/data/train-00000-of-01357.parquet")
    ap.add_argument("--row", type=int, default=13)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    check_letterbox()
    for b in (32, 7, 64):
        check_detector(Path(args.ref_dir), Path(args.parquet), args.row, args.n, b)


if __name__ == "__main__":
    main()
