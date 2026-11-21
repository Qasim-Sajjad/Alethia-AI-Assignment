"""Video indexing at speed: which of five objects appear in each second, and how many people.

    python run.py --input <folder of .parquet files> --output results.jsonl

Fast pipeline (specs/20-pipeline.md): parallel decode workers with adaptive per-second frame sampling
(specs/10-sampling.md) feeding one GPU process that batches YOLO11l across videos.
The first, every-frame version is kept as tools/baseline_run.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vidx.pipeline import Config, run


def main():
    d = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="auto",
                    help="auto = weights/yolo11l.engine if built by prepare.py, else weights/yolo11l.pt")
    ap.add_argument("--workers", type=int, default=d.workers, help="decode worker processes")
    ap.add_argument("--batch", type=int, default=None, help="max frames per GPU batch (default 64; 32 with --fp32)")
    ap.add_argument("--sampling", choices=["adaptive", "all"], default=d.sampling)
    ap.add_argument("--fp32", action="store_true", help="FP32 inference for the .pt model (default FP16)")
    ap.add_argument("--decode-threads", type=int, default=d.decode_threads)
    ap.add_argument("--dump-dir", default=None, help="save per-frame detections of frames that were run")
    ap.add_argument("--resume", action="store_true", help="skip videos already in --output")
    ap.add_argument("--timings", default=None, help="write stage timings JSON here (default: <output>.timings.json)")
    args = ap.parse_args()

    weights = args.weights
    if weights == "auto":
        weights = "weights/yolo11l.engine" if Path("weights/yolo11l.engine").exists() else "weights/yolo11l.pt"
    is_engine = weights.endswith(".engine")
    batch = args.batch or (32 if args.fp32 and not is_engine else 64)
    cfg = Config(weights=weights, workers=args.workers, batch=batch, sampling=args.sampling,
                 half=not args.fp32 and not is_engine, decode_threads=args.decode_threads,
                 dump_dir=args.dump_dir)
    print(f"model: {weights} ({'TensorRT' if is_engine else 'FP16' if cfg.half else 'FP32'}), "
          f"workers {cfg.workers}, batch {cfg.batch}, sampling {cfg.sampling}", file=sys.stderr)
    report = run(args.input, args.output, cfg, resume=args.resume)
    report["weights"] = weights
    tpath = Path(args.timings or (args.output + ".timings.json"))
    tpath.write_text(json.dumps(report, indent=2))
    print(f"done: {report['videos']} videos, {report['frames_decoded']} frames decoded, "
          f"{report['frames_inferred']} inferred ({report['inferred_pct']:.1f}%), "
          f"{report['wall_s']:.1f}s wall, {report['video_fps']:.1f} video-fps, "
          f"{report['error_lines']} error lines; timings -> {tpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
