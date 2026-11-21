"""YOLO11l detection on already-letterboxed BGR batches, matching Ultralytics' predictor exactly.

Ultralytics predictor steps reproduced here (engine/predictor.py, models/yolo/detect/predict.py):
  uint8 BHWC -> device -> BCHW -> flip(1) BGR->RGB -> float (or half) / 255 -> AutoBackend model
  -> nms.non_max_suppression(conf, iou=0.7, classes=None, agnostic=False, max_det=300, nc=0)
The AutoBackend (fused Conv+BN) is taken from a predictor set up by one warm-up predict() call, so
weights and fusion are identical to model.predict().
"""
from __future__ import annotations

import numpy as np
import torch

from . import COCO_IDS, IMGSZ, IOU

DET_CONF = 0.25  # NMS confidence floor; counts use CONF=0.4 (same >=0.4 set, see specs)


class Detector:
    def __init__(self, weights: str, half: bool = False, device: str | None = None):
        from ultralytics import YOLO

        self.yolo = YOLO(weights)
        dummy = np.zeros((360, 640, 3), np.uint8)
        kw = {"half": True} if half else {}
        self.yolo.predict(dummy, imgsz=IMGSZ, conf=DET_CONF, iou=IOU, verbose=False, device=device, **kw)
        self.pred = self.yolo.predictor
        self.model = self.pred.model
        self.device = self.pred.device
        self.fp16 = bool(self.model.fp16)
        self.end2end = getattr(self.model, "end2end", False)
        from ultralytics.utils import nms

        self._nms = nms.non_max_suppression
        lut = np.full(1000, -1, np.int16)
        for i, c in enumerate(COCO_IDS):
            lut[c] = i
        self.lut = lut

    @torch.inference_mode()
    def infer(self, batch: np.ndarray):
        """batch: (B, H, W, 3) uint8 BGR, letterboxed. Returns list of (cls5 int8[], conf float32[])."""
        im = torch.from_numpy(batch).to(self.device, non_blocking=True)
        im = im.permute(0, 3, 1, 2).contiguous().flip(1)
        im = (im.half() if self.fp16 else im.float()).div_(255)
        preds = self.model(im)
        # max_time_img raised from 0.05: Ultralytics' NMS returns EMPTY results for the rest of a batch
        # once its wall-clock budget is exceeded. A one-frame reference never hits it; a large batch
        # on a busy GPU can. It is a safety cutoff, not a detection parameter.
        dets = self._nms(preds, DET_CONF, IOU, None, False, max_det=300, nc=0, end2end=self.end2end,
                         max_time_img=10.0)
        out = []
        for d in dets:
            d = d.cpu().numpy()
            cls = self.lut[d[:, 5].astype(int)]
            keep = cls >= 0
            out.append((cls[keep].astype(np.int8), d[keep, 4].astype(np.float32)))
        return out
