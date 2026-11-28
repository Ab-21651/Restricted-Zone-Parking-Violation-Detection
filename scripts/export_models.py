"""Step 5: export the nano detectors to edge-friendly formats.

    python scripts/export_models.py                          # default models x formats
    python scripts/export_models.py --models yolo26n.pt --formats onnx

Exports land next to the weights in models/ (e.g. models/yolo26n.onnx,
models/yolo26n_ncnn_model/). FP32, imgsz 640 - the same settings as the
published Raspberry Pi 5 table, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import time

from ultralytics import YOLO

from edgepark.detector import resolve_model_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"])
    ap.add_argument("--formats", nargs="+", default=["onnx", "ncnn"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    for m in args.models:
        path = resolve_model_path(m)
        for fmt in args.formats:
            t0 = time.time()
            out = YOLO(path).export(format=fmt, imgsz=args.imgsz, verbose=False)
            print(f"{m:12s} -> {fmt:8s} {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
