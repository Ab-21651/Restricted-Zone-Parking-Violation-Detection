"""One overview image per video, to decide where restricted zones should go.

Each image has:
  * left:  the empty scene (median of sampled frames, which removes passing
           vehicles) with a heatmap of where vehicles' ground points were.
           Bright spots = a vehicle stood there for a long time (parked).
  * right: a timeline of snapshots across the video.

    python scripts/video_overview.py                       # all videos
    python scripts/video_overview.py --videos ISLab-07 ISLab-13
Output: outputs/overview/<video>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from edgepark.config import load_config
from edgepark.detector import VehicleDetector
from edgepark.video import iter_frames, source_info


def overview(video: Path, detector: VehicleDetector, sample_fps: float, n_thumbs: int) -> np.ndarray:
    info = source_info(str(video))
    w, h, dur = info["width"], info["height"], info["frames"] / (info["fps"] or 30)
    heat = np.zeros((h, w), np.float32)
    samples, thumbs = [], []
    thumb_times = list(np.linspace(0, dur * 0.97, n_thumbs))

    for _, t, _, frame in iter_frames(str(video), sample_fps):
        dets, _ = detector.detect(frame)
        if dets:
            layer = np.zeros_like(heat)  # cv2.circle overwrites, so draw per frame then add
            for d in dets:
                x, y = (int(v) for v in d.bottom_center)
                cv2.circle(layer, (min(x, w - 1), min(y, h - 1)), max(4, int(d.height * 0.08)), 1.0, -1)
            heat += layer
        if len(samples) < 60 or np.random.rand() < 0.1:
            samples.append(cv2.resize(frame, (w, h)))
        if thumb_times and t >= thumb_times[0]:
            thumb_times.pop(0)
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
            tw = w // 2
            th_ = h // 2
            th = cv2.resize(frame, (tw, th_))
            cv2.rectangle(th, (0, 0), (90, 24), (0, 0, 0), -1)
            cv2.putText(th, f"{int(t // 60)}:{int(t % 60):02d}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            thumbs.append(th)

    # empty-scene background
    bg = np.median(np.stack(samples[-120:]), axis=0).astype(np.uint8)
    # heatmap: seconds spent per location (each sample = 1/sample_fps s)
    secs = heat / sample_fps
    norm = np.clip(secs / max(30.0, np.percentile(secs[secs > 0], 99) if (secs > 0).any() else 30.0), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (norm > 0.02)[..., None]
    left = np.where(mask, cv2.addWeighted(bg, 0.45, color, 0.55, 0), bg)
    cv2.putText(left, f"{video.stem}  ({dur:.0f}s)  heat = time vehicles stood here", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(left, f"{video.stem}  ({dur:.0f}s)  heat = time vehicles stood here", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    while len(thumbs) < n_thumbs:
        thumbs.append(np.zeros_like(thumbs[0]) if thumbs else np.zeros((h // 2, w // 2, 3), np.uint8))
    grid = np.vstack([np.hstack(thumbs[i : i + 2]) for i in range(0, n_thumbs, 2)])
    grid = cv2.resize(grid, (int(grid.shape[1] * left.shape[0] * 2 / grid.shape[0]), left.shape[0] * 2))
    left_big = cv2.resize(left, (left.shape[1] * 2, left.shape[0] * 2))
    return np.hstack([left_big, grid])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos/islab")
    ap.add_argument("--videos", nargs="*", help="Video stems (default: all)")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--thumbs", type=int, default=8)
    ap.add_argument("--out", default="outputs/overview")
    args = ap.parse_args()

    cfg = load_config()
    detector = VehicleDetector(cfg["model"]["path"], conf=0.3, vehicle_classes=cfg["vehicle_classes"])
    videos = sorted(Path(args.dir).glob("*.mp4"))
    if args.videos:
        videos = [v for v in videos if v.stem in args.videos]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for v in videos:
        img = overview(v, detector, args.sample_fps, args.thumbs)
        out = Path(args.out) / f"{v.stem}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
