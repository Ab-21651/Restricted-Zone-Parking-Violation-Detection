"""Step 1: run the baseline vehicle detector on an image, video, webcam or stream.

Examples:
    python scripts/run_detect.py --source data/videos/sample.mp4 --show
    python scripts/run_detect.py --source data/videos/sample.mp4 --save outputs/detect.mp4
    python scripts/run_detect.py --source 0 --show            # webcam
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2

from edgepark.config import load_config
from edgepark.detector import VehicleDetector
from edgepark.draw import draw_detections, draw_hud
from edgepark.video import FpsMeter, open_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--source", default=None, help="Override source URI from config")
    ap.add_argument("--model", default=None, help="Override model path from config")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="Display annotated frames")
    ap.add_argument("--save", default=None, help="Write annotated output (video or image path)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    ap.add_argument("--snapshot", default=None, help="Save the last annotated frame as an image")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    source = args.source or cfg["source"]["uri"]

    detector = VehicleDetector(
        args.model or mcfg["path"],
        imgsz=args.imgsz or mcfg.get("imgsz", 640),
        conf=mcfg.get("conf", 0.35),
        iou=mcfg.get("iou", 0.5),
        vehicle_classes=cfg["vehicle_classes"],
    )
    print(f"Model: {detector.model_path}  imgsz={detector.imgsz}")
    detector.warmup()

    fps_meter = FpsMeter()
    latencies: list[float] = []
    writer = None
    n = 0

    for frame in open_source(source):
        detections, latency_ms = detector.detect(frame)
        latencies.append(latency_ms)
        fps = fps_meter.tick()
        n += 1

        if args.show or args.save or args.snapshot:
            draw_detections(frame, detections)
            draw_hud(frame, [f"infer {latency_ms:.1f} ms", f"loop {fps:.1f} FPS", f"vehicles {len(detections)}"])

        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix.lower() in {".jpg", ".png"}:
                cv2.imwrite(str(out), frame)
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(frame)

        if args.show:
            cv2.imshow("edgepark - detect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and n >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    if args.snapshot and n:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.snapshot, frame)
    cv2.destroyAllWindows()

    if latencies:
        lat_sorted = sorted(latencies)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
        mean = statistics.mean(latencies)
        print(f"Frames: {n}")
        print(f"Inference latency  mean {mean:.1f} ms | median {statistics.median(latencies):.1f} ms | p95 {p95:.1f} ms")
        print(f"Inference-only FPS {1000.0 / mean:.1f}")


if __name__ == "__main__":
    main()
