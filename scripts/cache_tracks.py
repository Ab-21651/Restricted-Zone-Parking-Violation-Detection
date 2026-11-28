"""Step 6a: run detection + tracking once per ISLab video and cache the result.

Every ablation / threshold variant in eval_alerts.py replays these cached frames,
so variants differ only in the alert logic, never in the detections.
Default rate 3 frames per second = what the Pi-5-calibrated edge node achieves
(3.1-3.4 FPS measured in the Docker deployment).

    python scripts/cache_tracks.py                          # all 16 videos
    python scripts/cache_tracks.py --videos ISLab-07 --model models/yolo26n.onnx
Output: outputs/tracks/<video>.jsonl  (one line per analysed frame)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from edgepark.config import load_config
from edgepark.detector import VehicleDetector
from edgepark.video import iter_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/videos/islab")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--model", default=None, help="default: model.path in config")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--out", default="outputs/tracks")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg.get("tracking", {})
    videos = sorted(Path(args.dir).glob("*.mp4"))
    if args.videos:
        videos = [v for v in videos if v.stem in args.videos]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model or cfg["model"]["path"], "fps": args.fps, "tracker": tcfg.get("tracker"),
            "track_conf": tcfg.get("conf", 0.1), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for v in videos:
        # fresh detector per video so tracker IDs / state never leak between videos
        det = VehicleDetector(meta["model"], imgsz=cfg["model"].get("imgsz", 640), iou=cfg["model"].get("iou", 0.5),
                              vehicle_classes=cfg["vehicle_classes"])
        t0, n = time.time(), 0
        with open(Path(args.out) / f"{v.stem}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": dict(meta, video=v.name)}) + "\n")
            for idx, t, _, frame in iter_frames(str(v), args.fps):
                dets, ms = det.track(frame, tcfg.get("tracker", "configs/bytetrack.yaml"), tcfg.get("conf", 0.1))
                f.write(json.dumps({"i": idx, "t": round(t, 3), "ms": round(ms, 1), "d": [
                    [round(x, 1) for x in d.xyxy] + [d.cls_id, round(d.conf, 3), d.track_id] for d in dets]}) + "\n")
                n += 1
        print(f"{v.stem}: {n} frames cached in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
