"""Preparation step (run once, with internet, untimed): download yolo11l.pt and build a TensorRT engine.

    python prepare.py

Produces weights/yolo11l.pt and, if TensorRT is available, weights/yolo11l.engine:
FP16, dynamic shapes (so frames keep Ultralytics' minimal 640xH letterbox, e.g. 640x384), max batch
BATCH. run.py uses the engine automatically when it exists and falls back to the .pt model (FP16).
A TensorRT engine is specific to the GPU and TensorRT version, so it must be built on the machine
that runs run.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WEIGHTS = Path("weights/yolo11l.pt")
ENGINE = Path("weights/yolo11l.engine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="max batch the engine accepts")
    ap.add_argument("--no-engine", action="store_true", help="only download the .pt weights")
    args = ap.parse_args()

    from ultralytics import YOLO

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))  # downloads the official weights if missing
    print(f"weights: {WEIGHTS} ({WEIGHTS.stat().st_size / 2**20:.1f} MB)")
    if args.no_engine:
        return
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("TensorRT not installed; run.py will use the .pt model in FP16", file=sys.stderr)
        return
    try:
        out = model.export(format="engine", half=True, dynamic=True, batch=args.batch, imgsz=640,
                           workspace=8, simplify=True, verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f"TensorRT export failed ({type(e).__name__}: {e}); run.py will use the .pt model in FP16",
              file=sys.stderr)
        ENGINE.unlink(missing_ok=True)
        return
    out = Path(out)
    if out.resolve() != ENGINE.resolve():
        shutil.move(str(out), ENGINE)
    print(f"engine: {ENGINE} ({ENGINE.stat().st_size / 2**20:.1f} MB), max batch {args.batch}")
    # sanity check: the engine loads and runs one batch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np

    from vidx.detector import Detector

    det = Detector(str(ENGINE))
    det.infer(np.full((4, 384, 640, 3), 114, np.uint8))
    print("engine check: ok (fp16 =", det.fp16, ")")


if __name__ == "__main__":
    main()
