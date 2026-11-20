"""Restricted zone polygons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import PROJECT_ROOT

ZONES_DIR = PROJECT_ROOT / "configs" / "zones"


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) float32 pixel coordinates

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0


def load_zones(path: str | Path, frame_size: tuple[int, int] | None = None) -> list[Zone]:
    """Load zones from YAML. If ``frame_size`` (w, h) differs from the size the
    zones were drawn on, the polygons are rescaled."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sx = sy = 1.0
    if frame_size and data.get("frame_size"):
        dw, dh = data["frame_size"]
        sx, sy = frame_size[0] / dw, frame_size[1] / dh
    zones = []
    for z in data.get("zones", []):
        pts = np.array(z["polygon"], dtype=np.float32) * np.array([sx, sy], dtype=np.float32)
        zones.append(Zone(name=z["name"], polygon=pts))
    return zones


def save_zones(path: str | Path, zones: list[Zone], frame_size: tuple[int, int], video: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": video,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "zones": [
            {"name": z.name, "polygon": [[int(round(x)), int(round(y))] for x, y in z.polygon]} for z in zones
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def zones_path_for(video: str | Path) -> Path:
    """Default zone file for a video: configs/zones/<video stem>.yaml"""
    return ZONES_DIR / f"{Path(video).stem}.yaml"
