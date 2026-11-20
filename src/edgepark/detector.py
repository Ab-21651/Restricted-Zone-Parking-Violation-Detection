"""Vehicle detection using a COCO-pretrained YOLO model.

Ultralytics' ``YOLO`` class loads PyTorch (.pt) weights as well as exported
formats (ONNX, NCNN, TFLite, ...), so the same wrapper serves both the
baseline and the optimised models benchmarked later.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import MODELS_DIR

DEFAULT_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    cls_id: int
    cls_name: str
    conf: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point of the vehicle (used for zone checks later)."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)


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

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Run detection on a BGR frame. Returns (detections, latency_ms)."""
        t0 = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=list(self.vehicle_classes),
            verbose=False,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cls, confs):
                detections.append(
                    Detection(
                        xyxy=tuple(float(v) for v in b),
                        cls_id=int(c),
                        cls_name=self.vehicle_classes.get(int(c), str(c)),
                        conf=float(s),
                    )
                )
        return detections, latency_ms
