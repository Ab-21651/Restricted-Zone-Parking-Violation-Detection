"""Frame sources and simple performance meters."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def open_source(uri: str | int) -> Iterator[np.ndarray]:
    """Yield BGR frames from an image, video file, webcam index or RTSP URL."""
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)

    if isinstance(uri, str) and Path(uri).suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(uri)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {uri}")
        yield img
        return

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {uri}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def source_info(uri: str | int) -> dict:
    """Frame size, FPS and frame count of a video source (best effort)."""
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)
    cap = cv2.VideoCapture(uri)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def iter_frames(
    uri: str | int, process_fps: float = 0.0, live: bool | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_seconds, frame)``.

    * Files: ``t`` is the video timestamp, and frames are subsampled to about
      ``process_fps`` (0 = every frame), so dwell timing follows video time.
    * Live sources (webcam / RTSP): ``t`` is wall-clock time since start and
      every frame read is yielded; the processing speed itself limits the rate.
    """
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)
    if live is None:
        live = isinstance(uri, int) or (isinstance(uri, str) and uri.lower().startswith(("rtsp://", "http://", "udp://")))

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {uri}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = 1 if (live or process_fps <= 0) else max(1, round(src_fps / process_fps))
    t_start = time.perf_counter()
    idx = -1
    try:
        while True:
            if stride > 1:
                # grab() skips decoding of frames we will not analyse
                for _ in range(stride - 1):
                    if not cap.grab():
                        return
                    idx += 1
            ok, frame = cap.read()
            if not ok:
                return
            idx += 1
            t = (time.perf_counter() - t_start) if live else idx / src_fps
            yield idx, t, frame
    finally:
        cap.release()


class FpsMeter:
    """Exponential moving average of end-to-end loop FPS."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.fps = 0.0
        self._last: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-9)
            self.fps = inst if self.fps == 0.0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps
