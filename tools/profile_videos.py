"""Profile the videos in a folder of FineVideo Parquet files.

Demuxes packets only (no full decode) to get codec, resolution, fps, duration,
keyframe spacing and timestamp regularity; decodes one frame to read rotation.

Usage: python tools/profile_videos.py --input finevideo/data --out cache/profile.csv
"""
import argparse
import csv
import io
import statistics
from pathlib import Path

import av
import pyarrow.parquet as pq


def profile_one(mp4: bytes) -> dict:
    with av.open(io.BytesIO(mp4)) as c:
        vs = c.streams.video[0]
        info = {
            "codec": vs.codec_context.name,
            "width": vs.codec_context.width,
            "height": vs.codec_context.height,
            "pix_fmt": vs.codec_context.pix_fmt,
            "avg_fps": float(vs.average_rate) if vs.average_rate else None,
            "time_base": str(vs.time_base),
            "has_audio": len(c.streams.audio) > 0,
        }
        pts, keys = [], []
        for pkt in c.demux(vs):
            if pkt.pts is None:
                continue
            pts.append(pkt.pts)
            if pkt.is_keyframe:
                keys.append(pkt.pts)
        pts.sort()
        tb = float(vs.time_base)
        deltas = [(b - a) * tb for a, b in zip(pts, pts[1:])]
        info["n_packets"] = len(pts)
        info["duration_s"] = round((pts[-1] - pts[0]) * tb, 3) if pts else 0
        info["n_keyframes"] = len(keys)
        info["gop_s"] = round(info["duration_s"] / max(len(keys), 1), 2)
        if deltas:
            md = statistics.median(deltas)
            info["median_dt"] = round(md, 5)
            info["vfr_frac"] = round(sum(abs(d - md) > 0.25 * md for d in deltas) / len(deltas), 4)
        c.seek(0)
        rot = 0
        for f in c.decode(vs):
            rot = f.rotation
            break
        info["rotation"] = rot
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for pf in sorted(Path(args.input).glob("*.parquet")):
        f = pq.ParquetFile(pf)
        idx = 0
        for batch in f.iter_batches(batch_size=1, columns=["mp4"]):
            mp4 = batch.column(0)[0].as_py()
            rec = {"video": f"{pf.name}:{idx}", "mb": round(len(mp4) / 2**20, 1)}
            try:
                rec.update(profile_one(mp4))
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:200]
            rows.append(rec)
            print(rec, flush=True)
            idx += 1
    keys = sorted({k for r in rows for k in r})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
