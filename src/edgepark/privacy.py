"""Privacy filter for images that leave the edge node (alert snapshots, shared media).

* People are found with the same COCO detector (class 0 = person) and blurred.
* For public sharing, vehicles can be pixelated too (``redact_vehicles``), which
  hides number plates without needing a separate plate detector.

A separate YOLO instance is used on purpose: calling ``predict`` on the model
that runs ``track(persist=True)`` would reuse its predictor and feed these
detections into the vehicle tracker.
"""

from __future__ import annotations

import cv2
import numpy as np


def _blur_box(img: np.ndarray, box, strength: float = 0.35) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(3, int(min(roi.shape[:2]) * strength) | 1)  # odd kernel, scales with box size
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def _pixelate_box(img: np.ndarray, box, blocks: int = 6) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, max(1, blocks * roi.shape[0] // max(roi.shape[1], 1))), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)


class PrivacyFilter:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.25) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self._model = None  # loaded on first use (alerts are rare)

    def _detect(self, frame: np.ndarray, classes: list[int]):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path, task="detect")
        r = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=classes, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return [], []
        return r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    def apply(self, frame: np.ndarray, redact_vehicles: bool = False, detect_on: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Return (filtered copy, number of regions hidden).

        ``detect_on`` lets detection run on the clean frame while blurring an
        annotated copy (boxes / labels drawn on top would confuse the detector)."""
        src = frame if detect_on is None else detect_on
        classes = [0, 2, 3, 5, 7] if redact_vehicles else [0]
        boxes, cls = self._detect(src, classes)
        out = frame.copy()
        for b, c in zip(boxes, cls):
            if c == 0:
                _blur_box(out, b)
            else:
                _pixelate_box(out, b)
        return out, len(boxes)
