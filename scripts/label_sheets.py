"""Contact sheets for ground-truth labelling: raw frames at fixed time steps,
cropped to the restricted zones, with only the zone outline and the timestamp
drawn (no detector / tracker output, so the labels stay independent).

    python scripts/label_sheets.py --step 5                       # all videos, every 5 s
    python scripts/label_sheets.py --videos ISLab-07 --start 25 --end 45 --step 1
Output: outputs/labeling/<video>_<start>-<end>s_pNN.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from edgepark.zones import load_zones, zones_path_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos/islab")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0 = end of video")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile", type=int, default=380)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--out", default="outputs/labeling")
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                    help="Explicit crop (pixels) instead of the zones' bounding box, e.g. to zoom on small vehicles")
    ap.add_argument("--tag", default="", help="Extra text in the output file name")
    args = ap.parse_args()

    videos = sorted(Path(args.dir).glob("*.mp4"))
    if args.videos:
        videos = [v for v in videos if v.stem in args.videos]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = n / fps
        zones = load_zones(zones_path_for(v), (w, h))
        pts = np.vstack([z.polygon for z in zones])
        x1, y1 = np.maximum(pts.min(0) - args.margin, 0).astype(int)
        x2, y2 = np.minimum(pts.max(0) + args.margin, [w, h]).astype(int)
        if args.crop:
            x1, y1, x2, y2 = args.crop
        end = args.end or dur
        times = np.arange(args.start, min(end, dur - 0.05) + 1e-6, args.step)
        tiles = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, f = cap.read()
            if not ok:
                continue
            for z in zones:
                cv2.polylines(f, [z.polygon.astype(np.int32)], True, (0, 255, 255), 1)
            crop = f[y1:y2, x1:x2]
            th = int(args.tile * crop.shape[0] / crop.shape[1])
            tile = cv2.resize(crop, (args.tile, th), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(tile, (0, 0), (92, 24), (0, 0, 0), -1)
            cv2.putText(tile, f"{t:6.1f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(tile)
        per = args.cols * args.rows
        for p in range(0, len(tiles), per):
            chunk = tiles[p:p + per]
            th = max(t.shape[0] for t in chunk)
            chunk = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in chunk]
            while len(chunk) % args.cols:
                chunk.append(np.zeros_like(chunk[0]))
            sheet = np.vstack([np.hstack(chunk[i:i + args.cols]) for i in range(0, len(chunk), args.cols)])
            name = f"{v.stem}{args.tag}_{int(args.start)}-{int(end)}s_p{p // per + 1:02d}.jpg"
            cv2.imwrite(str(Path(args.out) / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{v.stem}: {len(tiles)} frames, {dur:.0f}s, crop {x2 - x1}x{y2 - y1}", flush=True)


if __name__ == "__main__":
    main()
