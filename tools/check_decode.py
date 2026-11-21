"""Decoder sanity checks.

1. PyAV bgr24 frames vs OpenCV VideoCapture frames for the same video (pixel diff).
2. Rotation: make a copy of a video with a display-rotation tag via ffmpeg, then compare
   our PyAV decode + rotate_frame against ffmpeg's autorotated decode and OpenCV's.

Usage: python tools/check_decode.py --parquet finevideo/data/train-00000-of-01357.parquet --row 0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx.io import decode_frames  # noqa: E402

N = 120


def cv2_frames(path: str, n: int):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def pyav_frames(mp4: bytes, n: int):
    out = []
    for _, img in decode_frames(mp4):
        out.append(img)
        if len(out) >= n:
            break
    return out


def ffmpeg_frames(path: str, n: int):
    probe = cv2.VideoCapture(path)
    probe.release()
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    # get autorotated size from ffprobe-less approach: try both orientations
    return raw


def compare(a, b, label):
    if len(a) != len(b):
        print(f"{label}: frame count differs {len(a)} vs {len(b)}")
    diffs = []
    for x, y in zip(a, b):
        if x.shape != y.shape:
            print(f"{label}: shape differs {x.shape} vs {y.shape}")
            return
        diffs.append(np.abs(x.astype(int) - y.astype(int)))
    mx = max(int(d.max()) for d in diffs)
    mean = float(np.mean([d.mean() for d in diffs]))
    identical = sum(int(d.max() == 0) for d in diffs)
    print(f"{label}: {identical}/{len(diffs)} frames identical, max abs diff {mx}, mean abs diff {mean:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()
    t = pq.read_table(args.parquet, columns=["mp4"])
    mp4 = t.column(0)[args.row].as_py()
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "src.mp4"
    src.write_bytes(mp4)

    a = pyav_frames(mp4, N)
    b = cv2_frames(str(src), N)
    compare(a, b, "PyAV vs OpenCV (no rotation)")

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping rotation check")
        return
    h, w = a[0].shape[:2]
    for rot in (90, 180, 270):
        dst = tmp / f"rot{rot}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rot), "-i", str(src),
                        "-c", "copy", str(dst)], check=True)
        ours = pyav_frames(dst.read_bytes(), 5)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dst), "-frames:v", "5", "-f", "rawvideo",
                              "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
        oh, ow = ours[0].shape[:2]
        ff = np.frombuffer(raw, np.uint8).reshape(-1, oh, ow, 3) if len(raw) == 5 * oh * ow * 3 else None
        cv = cv2_frames(str(dst), 5)
        print(f"rotation tag {rot}: ours {ours[0].shape}, opencv {cv[0].shape if cv else None}")
        if ff is not None:
            compare(ours, list(ff), f"  ours vs ffmpeg autorotate ({rot})")
        else:
            print(f"  ffmpeg output size mismatch (raw {len(raw)} bytes) -> our shape is wrong for {rot}")
        if cv:
            compare(ours, cv, f"  ours vs OpenCV ({rot})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
