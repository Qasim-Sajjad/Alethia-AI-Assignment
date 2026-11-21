"""Build a folder of damaged / unusual Parquet inputs to test run.py's error handling.

Writes into --out (default cache/robust):
  robust-00000-of-00003.parquet  rows:
    0 good          a normal short video
    1 truncated     first 40% of an mp4's bytes
    2 tail_cut      last 30% of bytes removed
    3 corrupt_mid   random bytes over 10% of the middle of the file
    4 empty         b""
    5 null          None
    6 garbage       random bytes, not an mp4
    7 rotated90     good video re-muxed with a 90-degree display rotation tag
    8 res_change    two clips of different resolutions concatenated into one stream
    9 audio_only    mp4 with an audio stream and no video
   10 zero_frames   mp4 with a video stream but no frames
   11 one_frame     a single-frame video
   12 vfr           variable frame rate video
  robust-00001-of-00003.parquet  a normal file (2 good videos) to check the run continues
  robust-00002-of-00003.parquet  a truncated Parquet file (unreadable)

Needs ffmpeg on PATH. Usage: python tools/make_robustness_set.py --src finevideo/data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS = ["good", "truncated", "tail_cut", "corrupt_mid", "empty", "null", "garbage", "rotated90",
        "res_change", "audio_only", "zero_frames", "one_frame", "vfr"]


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="finevideo/data")
    ap.add_argument("--out", default="cache/robust")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src = sorted(Path(args.src).glob("*.parquet"))
    t0 = pq.read_table(src[0], columns=["mp4"]).column(0)
    t1 = pq.read_table(src[-1], columns=["mp4"]).column(0)
    good = t0[32].as_py()       # 21 s, 202x360
    good2 = t1[0].as_py()       # 31 s, 270x480
    good3 = t1[22].as_py()      # 31 s, 202x360, laptops
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.mp4").write_bytes(good)
    (tmp / "b.mp4").write_bytes(good2)

    rows: dict[str, bytes | None] = {}
    rows["good"] = good
    rows["truncated"] = good[: int(len(good) * 0.4)]
    rows["tail_cut"] = good[: int(len(good) * 0.7)]
    b = bytearray(good)
    a = len(b) // 2
    b[a:a + len(b) // 10] = rng.integers(0, 256, len(b) // 10, dtype=np.uint8).tobytes()
    rows["corrupt_mid"] = bytes(b)
    rows["empty"] = b""
    rows["null"] = None
    rows["garbage"] = rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes()

    ff("-display_rotation", 90, "-i", tmp / "a.mp4", "-c", "copy", tmp / "rot.mp4")
    rows["rotated90"] = (tmp / "rot.mp4").read_bytes()

    # resolution change: re-encode 5 s of each clip at its own size, concat the H.264 streams with -c copy
    ff("-i", tmp / "a.mp4", "-t", 5, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", 30, "-bsf:v",
       "h264_mp4toannexb", "-f", "mpegts", tmp / "p1.ts")
    ff("-i", tmp / "b.mp4", "-t", 5, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", 30, "-bsf:v",
       "h264_mp4toannexb", "-f", "mpegts", tmp / "p2.ts")
    ff("-i", f"concat:{tmp / 'p1.ts'}|{tmp / 'p2.ts'}", "-c", "copy", tmp / "res.mp4")
    rows["res_change"] = (tmp / "res.mp4").read_bytes()

    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "aac", tmp / "audio.mp4")
    rows["audio_only"] = (tmp / "audio.mp4").read_bytes()
    ff("-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30", "-frames:v", 0, "-c:v", "libx264", tmp / "zero.mp4")
    rows["zero_frames"] = (tmp / "zero.mp4").read_bytes()
    ff("-i", tmp / "a.mp4", "-frames:v", 1, "-an", "-c:v", "libx264", tmp / "one.mp4")
    rows["one_frame"] = (tmp / "one.mp4").read_bytes()
    ff("-i", tmp / "b.mp4", "-t", 8, "-an", "-vf", "setpts='if(lt(N,60),PTS,PTS+0.3*(N-60)/TB/30)'",
       "-fps_mode", "vfr", "-c:v", "libx264", tmp / "vfr.mp4")
    rows["vfr"] = (tmp / "vfr.mp4").read_bytes()

    def write(name, vals):
        tbl = pa.table({"mp4": pa.array(vals, type=pa.binary()), "json": pa.array(["{}"] * len(vals))})
        pq.write_table(tbl, out / name)

    write("robust-00000-of-00003.parquet", [rows[k] for k in ROWS])
    write("robust-00001-of-00003.parquet", [good2, good3])
    write("robust-00002-of-00003.parquet", [good, good2])
    p = out / "robust-00002-of-00003.parquet"
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # unreadable Parquet (footer missing)
    shutil.rmtree(tmp, ignore_errors=True)
    for i, k in enumerate(ROWS):
        v = rows[k]
        print(f"row {i:2d} {k:12s} {len(v) if v is not None else 'None'} bytes")


if __name__ == "__main__":
    main()
