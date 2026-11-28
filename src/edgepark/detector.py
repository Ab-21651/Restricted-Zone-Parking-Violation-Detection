"""Vehicle detection (and tracking) using a COCO-pretrained YOLO model.

Ultralytics' ``YOLO`` class loads PyTorch (.pt) weights as well as exported
formats (ONNX, NCNN, TFLite, ...), so the same wrapper serves both the
baseline and the optimised models benchmarked later.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .config import MODELS_DIR, PROJECT_ROOT

# Inference thread count (e.g. 4 = Raspberry Pi 5 cores). Set in the Docker image.
if os.environ.get("EDGEPARK_THREADS"):
    torch.set_num_threads(int(os.environ["EDGEPARK_THREADS"]))

DEFAULT_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    cls_id: int
    cls_name: str
    conf: float
    track_id: int | None = None

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point of the vehicle (used for zone checks)."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


def resolve_model_path(path: str | Path) -> str:
    """Return a usable model path, downloading bare weight names into models/."""
    p = Path(path)
    if p.exists():
        return str(p)
    local = MODELS_DIR / p.name
    if local.exists():
        return str(local)
    if p.parent == Path(".") and p.suffix == ".pt":
        # Ultralytics downloads bare names into the current directory; move it.
        YOLO(p.name)
        MODELS_DIR.mkdir(exist_ok=True)
        if Path(p.name).exists():
            shutil.move(p.name, local)
            return str(local)
    raise FileNotFoundError(f"Model not found: {path}")


class VehicleDetector:
    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        vehicle_classes: dict[int, str] | None = None,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.model = YOLO(self.model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES

    def warmup(self, runs: int = 2) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.detect(dummy)

    def _parse(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or not len(boxes):
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, s, tid in zip(xyxy, cls, confs, ids):
            detections.append(
                Detection(
                    xyxy=tuple(float(v) for v in b),
                    cls_id=int(c),
                    cls_name=self.vehicle_classes.get(int(c), str(c)),
                    conf=float(s),
                    track_id=None if tid is None else int(tid),
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Run detection on a BGR frame. Returns (detections, latency_ms)."""
        t0 = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=list(self.vehicle_classes),
            agnostic_nms=True,
            verbose=False,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse(result), latency_ms

    def track(
        self, frame: np.ndarray, tracker: str = "configs/bytetrack.yaml", conf: float = 0.1
    ) -> tuple[list[Detection], float]:
        """Detect + track (ByteTrack). Only confirmed tracks (with an id) are returned.

        Latency covers detection and tracking together.
        """
        tracker_path = Path(tracker)
        if not tracker_path.is_absolute() and not tracker_path.exists():
            tracker_path = PROJECT_ROOT / tracker_path
        t0 = time.perf_counter()
        result = self.model.track(
            frame,
            persist=True,
            tracker=str(tracker_path),
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            classes=list(self.vehicle_classes),
            agnostic_nms=True,  # one box per vehicle even if it scores as both car and truck
            verbose=False,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return [d for d in self._parse(result) if d.track_id is not None], latency_ms
