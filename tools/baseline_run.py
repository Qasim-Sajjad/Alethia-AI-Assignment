"""BASELINE (first working version, kept for timing comparison). Video indexing: which of five objects appear in each second, and how many people.

Baseline version: runs YOLO11l on EVERY decoded frame (exactly the grader's
reference procedure). Optionally dumps per-frame detections for building the
reference cache used by tools/score.py.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl
    python run.py --input ... --output ... --dump-dir cache/ref   # also cache per-frame detections
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidx import CLASS_NAMES, COCO_IDS, CONF, IMGSZ, IOU
from vidx.aggregate import aggregate
from vidx.io import decode_frames, iter_videos, list_parquet

DUMP_CONF = 0.25  # cache detections down to this confidence (counts still use CONF)
BATCH = 32


def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def detect_batch(model, frames: list[np.ndarray], conf: float):
    """Run YOLO on a list of same-shape BGR frames with Ultralytics defaults.

    Returns list of (cls_idx_in_CLASS_NAMES ndarray, conf ndarray) per frame.
    """
    res = model.predict(frames, imgsz=IMGSZ, conf=conf, iou=IOU, verbose=False)
    out = []
    lut = {c: i for i, c in enumerate(COCO_IDS)}
    for r in res:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        keep = np.isin(cls, COCO_IDS)
        out.append((np.array([lut[c] for c in cls[keep]], dtype=np.int8), cf[keep].astype(np.float32)))
    return out


def process_video(model, mp4: bytes, conf_run: float):
    """Decode every frame and detect. Returns (sec, counts, dets, nframes) or raises.

    dets: list of (frame_idx, cls_idx, conf) arrays for caching.
    On failure mid-video, the exception carries the partial result in .partial.
    """
    secs: list[int] = []
    counts: list[np.ndarray] = []
    d_frame, d_cls, d_conf = [], [], []
    buf: list[np.ndarray] = []
    buf_secs: list[int] = []

    def flush():
        if not buf:
            return
        # Ultralytics letterboxes a batch to one shape; frames of one video share a shape,
        # but guard against mid-stream resolution changes by splitting on shape.
        groups: list[tuple[int, int]] = []
        start = 0
        for i in range(1, len(buf) + 1):
            if i == len(buf) or buf[i].shape != buf[start].shape:
                groups.append((start, i))
                start = i
        for a, b in groups:
            for j, (cls, cf) in enumerate(detect_batch(model, buf[a:b], conf_run)):
                fi = len(secs)
                secs.append(buf_secs[a + j])
                m = cf >= CONF
                counts.append(np.bincount(cls[m], minlength=len(CLASS_NAMES)).astype(np.int16))
                d_frame.append(np.full(len(cls), fi, dtype=np.int32))
                d_cls.append(cls)
                d_conf.append(cf)
        buf.clear()
        buf_secs.clear()

    def result():
        sec = np.array(secs, dtype=np.int32)
        cnt = np.stack(counts) if counts else np.zeros((0, len(CLASS_NAMES)), np.int16)
        dets = (
            np.concatenate(d_frame) if d_frame else np.zeros(0, np.int32),
            np.concatenate(d_cls) if d_cls else np.zeros(0, np.int8),
            np.concatenate(d_conf) if d_conf else np.zeros(0, np.float32),
        )
        return sec, cnt, dets

    try:
        for s, img in decode_frames(mp4):
            buf.append(img)
            buf_secs.append(s)
            if len(buf) >= BATCH:
                flush()
        flush()
    except Exception as e:  # noqa: BLE001
        buf.clear()
        e.partial = result()  # type: ignore[attr-defined]
        raise
    return result()


def write_lines(fh, vid: str, sec: np.ndarray, cnt: np.ndarray):
    for s, objs, people in aggregate(sec, cnt):
        fh.write(json.dumps({"video": vid, "second": s, "objects": objs, "people": people}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    ap.add_argument("--dump-dir", default=None, help="cache per-frame detections here (npz per video)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos (testing)")
    args = ap.parse_args()

    model = load_model(args.weights)
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    conf_run = DUMP_CONF if dump else CONF

    t0 = time.perf_counter()
    n_done = 0
    total_frames = 0
    with open(args.output, "w", encoding="utf-8") as fh:
        for pf in list_parquet(args.input):
            try:
                videos = iter_videos(pf)
                for vid, mp4, err in videos:
                    if args.limit is not None and n_done >= args.limit:
                        break
                    n_done += 1
                    tv = time.perf_counter()
                    if err:
                        fh.write(json.dumps({"video": vid, "error": err}) + "\n")
                        continue
                    npz = dump / (vid.replace(":", "__") + ".npz") if dump else None
                    error = None
                    try:
                        sec, cnt, dets = process_video(model, mp4, conf_run)
                    except Exception as e:  # noqa: BLE001
                        sec, cnt, dets = getattr(e, "partial", (np.zeros(0, np.int32), None, None))
                        if cnt is None:
                            cnt = np.zeros((0, len(CLASS_NAMES)), np.int16)
                        error = f"{type(e).__name__}: {str(e)[:120]}"
                    write_lines(fh, vid, sec, cnt)
                    if error:
                        fh.write(json.dumps({"video": vid, "error": error}) + "\n")
                    fh.flush()
                    total_frames += len(sec)
                    if npz is not None and dets is not None:
                        np.savez_compressed(
                            npz, sec=sec, counts=cnt, det_frame=dets[0], det_cls=dets[1], det_conf=dets[2],
                            error=np.array(error or ""),
                        )
                    dt = time.perf_counter() - tv
                    print(f"{vid}: {len(sec)} frames, {dt:.1f}s ({len(sec)/max(dt,1e-9):.1f} fps)"
                          f"{' ERROR ' + error if error else ''}", flush=True)
            except Exception as e:  # noqa: BLE001  (unreadable parquet file)
                fh.write(json.dumps({"video": f"{pf.name}:0", "error": f"unreadable parquet: {str(e)[:100]}"}) + "\n")
                print(f"{pf.name}: unreadable ({e})", file=sys.stderr, flush=True)
            if args.limit is not None and n_done >= args.limit:
                break
    el = time.perf_counter() - t0
    print(f"done: {n_done} videos, {total_frames} frames, {el:.1f}s ({total_frames/max(el,1e-9):.1f} fps)")


if __name__ == "__main__":
    main()

