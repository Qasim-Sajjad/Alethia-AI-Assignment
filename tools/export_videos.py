"""Export videos from Parquet for visual inspection.

For each requested row it writes, into --out:
  <file>_row<NNN>_raw.mp4         the original mp4 bytes, untouched
  <file>_row<NNN>_annotated.mp4   every frame with a header showing frame number, timestamp, second,
                                  this frame's detections (conf >= 0.4, from the reference cache) and
                                  the per-second result; with --boxes, YOLO11l boxes are drawn too
  <file>_row<NNN>_seconds.jsonl   the per-second reference results for that video

The .npz cache holds detections (class + confidence per frame), not pixels, so frames are decoded
again from the Parquet file and the cached results are drawn on top.

Usage:
  python tools/export_videos.py --parquet finevideo/data/train-00000-of-01357.parquet --rows 13 16 \
      --ref-dir cache/ref --out finevideo/raw_videos --boxes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx import CLASS_NAMES, COCO_IDS, CONF, IMGSZ, IOU  # noqa: E402
from vidx.aggregate import aggregate  # noqa: E402
from vidx.io import decode_frames  # noqa: E402

COLORS = [(0, 200, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]  # BGR per class
HEADER = 78


def per_frame_counts(z) -> np.ndarray:
    m = z["det_conf"] >= CONF
    counts = np.zeros((len(z["sec"]), len(CLASS_NAMES)), np.int16)
    np.add.at(counts, (z["det_frame"][m], z["det_cls"][m].astype(int)), 1)
    return counts


def fmt_counts(row) -> str:
    parts = [f"{CLASS_NAMES[k]} x{int(row[k])}" for k in range(len(CLASS_NAMES)) if row[k] > 0]
    return ", ".join(parts) if parts else "-"


def put(img, text, y, scale=0.45, color=(255, 255, 255)):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def open_writer(path: Path, w: int, h: int, fps: float):
    """H.264 via PyAV if available, else mp4v via OpenCV."""
    try:
        av.codec.Codec("libx264", "w")  # raises if this PyAV build has no x264 encoder
        c = av.open(str(path), "w")
        s = c.add_stream("libx264", rate=round(fps * 1000) / 1000 if fps else 30)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "20", "preset": "veryfast"}

        def write(img):
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                c.mux(p)

        def close():
            for p in s.encode():
                c.mux(p)
            c.close()

        return write, close
    except Exception:  # noqa: BLE001
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30, (w, h))

        def close():
            vw.release()
            to_h264(path)

        return vw.write, close


def to_h264(path: Path):
    """Re-encode an mp4v file to H.264 with the system ffmpeg, if present, for wider player support."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:v", "libx264", "-crf", "20",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)


def export(parquet: Path, row: int, ref_dir: Path | None, out: Path, model):
    stem = f"{parquet.name.split('-of-')[0]}_row{row:03d}"
    mp4 = pq.read_table(parquet, columns=["mp4"]).column(0)[row].as_py()
    (out / f"{stem}_raw.mp4").write_bytes(mp4)
    print(f"wrote {stem}_raw.mp4 ({len(mp4) / 2**20:.1f} MB)")
    if ref_dir is None:
        return
    npz = ref_dir / f"{parquet.name}__{row}.npz"
    if not npz.exists():
        print(f"  no cache {npz.name}; skipping annotated export")
        return
    z = np.load(npz)
    counts = per_frame_counts(z)
    seconds = {s: (o, p) for s, o, p in aggregate(z["sec"], counts)}
    with open(out / f"{stem}_seconds.jsonl", "w", encoding="utf-8") as fh:
        for s, (o, p) in seconds.items():
            fh.write(json.dumps({"video": f"{parquet.name}:{row}", "second": s, "objects": o, "people": p}) + "\n")

    with av.open(__import__("io").BytesIO(mp4)) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate or 30)
    lut = {cid: i for i, cid in enumerate(COCO_IDS)}
    writer = None
    for fi, (sec, img) in enumerate(decode_frames(mp4)):
        h, w = img.shape[:2]
        if writer is None:
            W, H = w + (w % 2), h + HEADER + ((h + HEADER) % 2)
            write, close = open_writer(out / f"{stem}_annotated.mp4", W, H, fps)
            writer = True
        if model is not None:
            r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
            for (x1, y1, x2, y2), cls, cf in zip(r.boxes.xyxy.cpu().numpy().astype(int),
                                                 r.boxes.cls.cpu().numpy().astype(int),
                                                 r.boxes.conf.cpu().numpy()):
                if cls not in lut:
                    continue
                k = lut[cls]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[k], 1)
                cv2.putText(img, f"{CLASS_NAMES[k]} {cf:.2f}", (x1 + 2, max(y1 - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS[k], 1, cv2.LINE_AA)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[HEADER:HEADER + h, :w] = img
        objs, people = seconds.get(sec, ([], 0))
        put(canvas, f"{parquet.name.split('-of-')[0]}:{row}  frame {fi}  t={fi / fps:6.2f}s  SECOND {sec}", 16)
        put(canvas, f"frame dets: {fmt_counts(counts[fi]) if fi < len(counts) else '?'}", 36, 0.4)
        put(canvas, f"second {sec} result: {', '.join(objs) or 'none'} | people={people}", 56, 0.4,
            (0, 255, 255))
        write(canvas)
    if writer:
        close()
        print(f"wrote {stem}_annotated.mp4 and {stem}_seconds.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="finevideo/raw_videos")
    ap.add_argument("--ref-dir", default=None, help="reference cache; enables the annotated export")
    ap.add_argument("--boxes", action="store_true", help="re-run YOLO11l to draw boxes (needs --ref-dir)")
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = None
    if args.boxes:
        from ultralytics import YOLO

        model = YOLO(args.weights)
    for r in args.rows:
        export(Path(args.parquet), r, Path(args.ref_dir) if args.ref_dir else None, out, model)


if __name__ == "__main__":
    main()
