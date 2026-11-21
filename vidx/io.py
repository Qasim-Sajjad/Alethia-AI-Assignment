"""Reading videos out of Parquet files and decoding them to BGR frames."""
from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow.parquet as pq


def list_parquet(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*.parquet"))


def iter_videos(path: Path) -> Iterator[tuple[str, bytes | None, str | None]]:
    """Yield (video_id, mp4_bytes, error) for each row; reads one row at a time."""
    f = pq.ParquetFile(path)
    idx = 0
    for batch in f.iter_batches(batch_size=1, columns=["mp4"]):
        col = batch.column(0)
        for i in range(len(col)):
            vid = f"{path.name}:{idx}"
            val = col[i].as_py()
            if not val:
                yield vid, None, "empty mp4 bytes"
            else:
                yield vid, val, None
            idx += 1


def rotate_frame(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply display rotation the way ffmpeg autorotate does.

    PyAV's VideoFrame.rotation is the display-matrix angle in degrees (CCW).
    tools/check_decode.py confirms this matches ffmpeg autorotate and OpenCV
    pixel-for-pixel for tags 90/180/270.
    """
    r = int(round(rotation)) % 360
    if r == 0:
        return img
    if r == 90:
        return np.ascontiguousarray(np.rot90(img, 1))
    if r == 180:
        return np.ascontiguousarray(img[::-1, ::-1])
    if r == 270:
        return np.ascontiguousarray(np.rot90(img, -1))
    raise ValueError(f"unsupported rotation {rotation}")


def to_bgr(frame: av.VideoFrame) -> np.ndarray:
    """Decoded frame -> BGR uint8 with display rotation applied (same path as the reference)."""
    img = frame.to_ndarray(format="bgr24")
    if frame.rotation:
        img = rotate_frame(img, frame.rotation)
    return img


def decode_frames(mp4: bytes) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (second_index, bgr_frame) for every decoded frame."""
    for sec, frame in iter_frames(mp4):
        yield sec, to_bgr(frame)


def iter_frames(mp4: bytes, threads: int | None = None) -> Iterator[tuple[int, av.VideoFrame]]:
    """Yield (second_index, VideoFrame) for every decoded frame, without colour conversion.

    second_index = floor(t - t_first) using exact pts arithmetic.
    threads: decoder threads (None = FFmpeg auto).
    """
    with av.open(io.BytesIO(mp4)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        if threads is not None:
            vs.codec_context.thread_count = threads
        tb = vs.time_base
        rate = vs.average_rate or vs.guessed_rate or 30
        pts0 = None
        last_t = None
        for i, frame in enumerate(c.decode(vs)):
            if frame.pts is not None and tb is not None:
                t = Fraction(frame.pts) * tb
            elif last_t is not None:
                t = last_t + 1 / Fraction(rate)
            else:
                t = Fraction(0)
            if pts0 is None:
                pts0 = t
            last_t = t
            rel = t - pts0
            yield max(int(rel // 1), 0), frame
