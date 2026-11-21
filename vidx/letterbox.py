"""Letterbox exactly as Ultralytics' predictor does for same-shape batches of a .pt model.

Mirrors ultralytics.data.augment.LetterBox(new_shape=640, auto=True, stride=32, scaleup=True,
center=True, padding_value=114, interpolation=cv2.INTER_LINEAR). Kept dependency-free (no torch
import) so decode workers stay light; tools/check_letterbox.py asserts pixel equality.
"""
from __future__ import annotations

import cv2
import numpy as np

IMGSZ = 640
STRIDE = 32
PAD = 114


def letterbox_params(h: int, w: int) -> tuple[tuple[int, int], int, int, int, int]:
    r = min(IMGSZ / h, IMGSZ / w)
    new_unpad = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = IMGSZ - new_unpad[0], IMGSZ - new_unpad[1]
    dw, dh = np.mod(dw, STRIDE), np.mod(dh, STRIDE)
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return new_unpad, top, bottom, left, right


def output_shape(h: int, w: int) -> tuple[int, int]:
    (uw, uh), top, bottom, left, right = letterbox_params(h, w)
    return uh + top + bottom, uw + left + right


def letterbox(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_unpad, top, bottom, left, right = letterbox_params(h, w)
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
