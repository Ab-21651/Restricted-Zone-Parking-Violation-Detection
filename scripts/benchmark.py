"""Benchmark a detector (any format Ultralytics can load) on this machine / container.

Modes
  val     Same protocol as the published Ultralytics Raspberry Pi tables:
          model.val() on a dataset (default coco128.yaml), batch 1, FP32.
          Reports inference ms/image (pre/post-processing listed separately) and mAP.
  frames  Real per-frame cost in our pipeline: predict() on ISLab video frames
          (pre-loaded, so decoding is excluded), vehicle classes only.

Every run appends one JSON line to outputs/bench/results.jsonl, including the CPU
limit actually in force (read from the container's cgroup) and the thread count.

    python scripts/benchmark.py --mode val    --model yolo26n.pt --label host
    python scripts/benchmark.py --mode frames --model yolov8n.pt --frames 150
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def cpu_limit() -> float | None:
    """CPU quota of this container (cgroup v2 / v1), None if unlimited."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if q <= 0 else q / p
    except (OSError, ValueError):
        return None


def load_frames(video: str, n: int, stride: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {video}")
    return frames


def summarize(values: list[float]) -> dict:
    s = sorted(values)
    return {
        "mean": round(statistics.mean(s), 2),
        "median": round(statistics.median(s), 2),
        "p95": round(s[int(0.95 * (len(s) - 1))], 2),
        "n": len(s),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["val", "frames"], default="val")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default="coco128.yaml", help="val mode dataset")
    ap.add_argument("--video", default="data/videos/islab/ISLab-07.mp4", help="frames mode source")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EDGEPARK_THREADS", 0)),
                    help="torch threads (0 = default; env EDGEPARK_THREADS)")
    ap.add_argument("--label", default=os.environ.get("BENCH_LABEL", ""))
    ap.add_argument("--out", default="outputs/bench/results.jsonl")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    from edgepark.detector import resolve_model_path

    if args.threads:
        torch.set_num_threads(args.threads)
    model_path = resolve_model_path(args.model) if args.model.endswith(".pt") else args.model
    model = YOLO(model_path, task="detect")

    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": args.label,
        "mode": args.mode,
        "model": Path(model_path).name,
        "imgsz": args.imgsz,
        "cpu_limit": cpu_limit(),
        "torch_threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "in_container": Path("/.dockerenv").exists(),
    }

    if args.mode == "val":
        r = model.val(data=args.data, imgsz=args.imgsz, batch=1, device="cpu", plots=False, verbose=False)
        rec.update(
            data=args.data,
            preprocess_ms=round(r.speed["preprocess"], 2),
            inference_ms=round(r.speed["inference"], 2),
            postprocess_ms=round(r.speed["postprocess"], 2),
            map50_95=round(float(r.box.map), 4),
            map50=round(float(r.box.map50), 4),
        )
        print(f"[{args.label or 'run'}] {rec['model']} cpus={rec['cpu_limit']} threads={rec['torch_threads']}: "
              f"inference {rec['inference_ms']} ms/im  (pre {rec['preprocess_ms']}, post {rec['postprocess_ms']})  "
              f"mAP50-95 {rec['map50_95']}")
    else:
        frames = load_frames(args.video, args.frames + args.warmup)
        kw = dict(imgsz=args.imgsz, classes=[2, 3, 5, 7], conf=0.25, verbose=False, device="cpu")
        for f in frames[: args.warmup]:
            model.predict(f, **kw)
        pre, inf, post, total = [], [], [], []
        for f in frames[args.warmup :]:
            t0 = time.perf_counter()
            res = model.predict(f, **kw)[0]
            total.append((time.perf_counter() - t0) * 1000)
            pre.append(res.speed["preprocess"])
            inf.append(res.speed["inference"])
            post.append(res.speed["postprocess"])
        rec.update(
            video=Path(args.video).name,
            preprocess_ms=summarize(pre)["mean"],
            inference_ms=summarize(inf),
            postprocess_ms=summarize(post)["mean"],
            total_ms=summarize(total),
            fps=round(1000 / statistics.mean(total), 2),
        )
        print(f"[{args.label or 'run'}] {rec['model']} cpus={rec['cpu_limit']} threads={rec['torch_threads']}: "
              f"total {rec['total_ms']['mean']} ms/frame ({rec['fps']} FPS), inference {rec['inference_ms']['mean']} ms")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
