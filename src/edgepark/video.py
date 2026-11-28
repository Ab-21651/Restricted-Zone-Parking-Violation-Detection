"""Frame sources and simple performance meters."""

from __future__ import annotations

import threading
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


def is_live(uri: str | int) -> bool:
    if isinstance(uri, int) or (isinstance(uri, str) and uri.isdigit()):
        return True
    return isinstance(uri, str) and uri.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


class LatestFrameReader:
    """Reads a live stream in a background thread and keeps only the newest frame.

    A real-time edge node must analyse the *current* scene: if inference is
    slower than the camera, older frames are dropped instead of queueing up
    (which would make the system fall further and further behind). The stream
    is reopened automatically if it drops.
    """

    def __init__(self, uri: str | int, reconnect_s: float = 2.0) -> None:
        self.uri = int(uri) if isinstance(uri, str) and uri.isdigit() else uri
        self.reconnect_s = reconnect_s
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self.read_count = 0  # frames received from the camera
        self.fps = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            cap = cv2.VideoCapture(self.uri)
            if not cap.isOpened():
                print(f"[camera] cannot open {self.uri}, retrying in {self.reconnect_s}s", flush=True)
                time.sleep(self.reconnect_s)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            print(f"[camera] connected to {self.uri} ({self.fps:.1f} fps)", flush=True)
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    print("[camera] stream lost, reconnecting", flush=True)
                    break
                with self._cond:
                    self._frame, self._ts = frame, time.time()
                    self.read_count += 1
                    self._cond.notify_all()
            cap.release()
            if not self._stop:
                time.sleep(self.reconnect_s)

    def get(self, last_count: int, timeout: float = 10.0) -> tuple[int, float, np.ndarray] | None:
        """Wait for a frame newer than ``last_count``; returns (count, capture_ts, frame)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self.read_count > last_count or self._stop, timeout):
                return None
            if self._stop:
                return None
            return self.read_count, self._ts, self._frame

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()


def iter_frames(
    uri: str | int, process_fps: float = 0.0, live: bool | None = None, stats: dict | None = None
) -> Iterator[tuple[int, float, float, np.ndarray]]:
    """Yield ``(frame_index, t_seconds, capture_wall_ts, frame)``.

    * Files: ``t`` is the video timestamp, and frames are subsampled to about
      ``process_fps`` (0 = every frame), so dwell timing follows video time.
      ``capture_wall_ts`` is the moment the frame was decoded.
    * Live sources (webcam / RTSP): newest-frame-only reading; ``t`` is
      wall-clock time since start; gaps in ``frame_index`` are dropped frames.
      If ``stats`` is given it is updated with frames received / analysed.
    """
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)
    if live is None:
        live = is_live(uri)

    if live:
        reader = LatestFrameReader(uri)
        t0, last = time.time(), 0
        try:
            while True:
                got = reader.get(last)
                if got is None:
                    print("[camera] no frames for 10 s", flush=True)
                    continue
                last, ts, frame = got
                if stats is not None:
                    stats.update(received=reader.read_count, camera_fps=reader.fps)
                yield last, ts - t0, ts, frame
        finally:
            reader.stop()
        return

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {uri}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = 1 if process_fps <= 0 else max(1, round(src_fps / process_fps))
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
            yield idx, idx / src_fps, time.time(), frame
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
