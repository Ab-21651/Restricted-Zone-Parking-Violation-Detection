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
