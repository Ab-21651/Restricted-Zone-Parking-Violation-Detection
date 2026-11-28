"""Step 5: speed of every model x format inside the Pi-5-calibrated container.

Uses the published Ultralytics protocol (val on coco128, batch 1, FP32, 640) so
YOLO26n can be compared format-by-format with the real Raspberry Pi 5 table -
the test of whether the calibrated container predicts real-device speed (RQ4).

    python scripts/bench_pi_sim.py                       # all models x formats, 3 runs each
    python scripts/bench_pi_sim.py --cpus 0.57 --repeats 3
Writes outputs/bench/pi_sim.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from calibrate_cpus import PROJECT, run_in_container

# Published Raspberry Pi 5 inference times (ms/image), Ultralytics Raspberry Pi
# guide, YOLO26n, FP32, imgsz 640, coco128.
PI5_YOLO26N = {"pytorch": 299.09, "onnx": 125.99, "ncnn": 67.03}

FORMATS = {"pytorch": "{m}.pt", "onnx": "models/{m}.onnx", "ncnn": "models/{m}_ncnn_model"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n", "yolo26n"])
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--cpus", type=float, default=0.57)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--image", default="edgepark-node:latest")
    args = ap.parse_args()

    out = PROJECT / "outputs" / "bench" / "pi_sim.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    for m in args.models:
        for fmt in args.formats:
            model = FORMATS[fmt].format(m=m)
            if fmt != "pytorch" and not (PROJECT / model).exists():
                print(f"skip {model} (not exported)")
                continue
            runs = [run_in_container(args.cpus, model, args.image, "4g") for _ in range(args.repeats)]
            ms = sorted(r["inference_ms"] for r in runs)
            row = {
                "model": m, "format": fmt, "cpus": args.cpus,
                "inference_ms": round(statistics.median(ms), 1), "runs_ms": ms,
                "preprocess_ms": runs[0]["preprocess_ms"], "postprocess_ms": runs[0]["postprocess_ms"],
                "map50_95_coco128": runs[0]["map50_95"],
                "pi5_published_ms": PI5_YOLO26N.get(fmt) if m == "yolo26n" else None,
            }
            if row["pi5_published_ms"]:
                row["error_pct"] = round(100 * (row["inference_ms"] - row["pi5_published_ms"]) / row["pi5_published_ms"], 1)
            rows = [r for r in rows if not (r["model"] == m and r["format"] == fmt and r["cpus"] == args.cpus)] + [row]
            out.write_text(json.dumps(rows, indent=2))
            extra = f"  | real Pi 5 {row['pi5_published_ms']} ms ({row['error_pct']:+}%)" if row.get("pi5_published_ms") else ""
            print(f"{m:8s} {fmt:8s} {row['inference_ms']:7.1f} ms/im  mAP {row['map50_95_coco128']}{extra}", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
