"""Visualisation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

COLORS = {
    "car": (255, 180, 0),
    "motorcycle": (0, 200, 255),
    "bus": (180, 0, 255),
    "truck": (0, 140, 255),
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = COLORS.get(d.cls_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return frame
