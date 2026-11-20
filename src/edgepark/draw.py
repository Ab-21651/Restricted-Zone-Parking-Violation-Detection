"""Visualisation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .zones import Zone

COLORS = {
    "car": (255, 180, 0),
    "motorcycle": (0, 200, 255),
    "bus": (180, 0, 255),
    "truck": (0, 140, 255),
}
STATUS_COLORS = {
    "outside": (200, 200, 200),  # grey
    "moving": (255, 255, 255),  # white
    "dwelling": (0, 220, 255),  # yellow
    "violation": (0, 0, 255),  # red
}


def _label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = COLORS.get(d.cls_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, f"{d.cls_name} {d.conf:.2f}", x1, y1, color)
    return frame


def draw_zones(frame: np.ndarray, zones: list[Zone], active: set[str] | None = None) -> np.ndarray:
    """Zones are filled semi-transparently: green normally, red while a violation is open."""
    active = active or set()
    overlay = frame.copy()
    for z in zones:
        pts = z.polygon.astype(np.int32)
        color = (0, 0, 255) if z.name in active else (0, 200, 0)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, dst=frame)
    for z in zones:
        x, y = z.polygon.min(axis=0).astype(int)
        cv2.putText(frame, z.name, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_tracks(frame: np.ndarray, detections: list[Detection], states: dict, t: float) -> np.ndarray:
    """Boxes coloured by status, labelled with track id and dwell time."""
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        st = states.get(d.track_id)
        status = st.status if st else "outside"
        color = STATUS_COLORS[status]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = (int(v) for v in d.bottom_center)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        text = f"#{d.track_id} {d.cls_name}"
        if st and status != "moving":
            text += f" {st.dwell(t):.0f}s"
        if status == "violation":
            text = "VIOLATION " + text
        _label(frame, text, x1, y1, color)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    y = frame.shape[0] - 10 - 24 * (len(lines) - 1)  # bottom-left, away from labels
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return frame
