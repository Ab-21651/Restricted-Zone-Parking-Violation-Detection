"""Build a contact sheet (one frame per video) and print basic video metadata.

    python scripts/preview_videos.py --dir data/videos/islab --out outputs/contact_sheet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def grab_frame(path: Path, at: float) -> tuple[np.ndarray | None, dict]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * at))
    ok, frame = cap.read()
    cap.release()
    meta = {"frames": n, "fps": fps, "size": f"{w}x{h}", "sec": n / fps if fps else 0}
    return (frame if ok else None), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos/islab")
    ap.add_argument("--out", default="outputs/contact_sheet.jpg")
    ap.add_argument("--at", type=float, default=0.5, help="Relative position in each video (0-1)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=400, help="Tile width in pixels")
    args = ap.parse_args()

    videos = sorted(Path(args.dir).glob("*.mp4"))
    tiles = []
    for v in videos:
        frame, meta = grab_frame(v, args.at)
        print(f"{v.name}: {meta['size']} {meta['fps']:.2f} fps {meta['frames']} frames {meta['sec']:.0f}s")
        if frame is None:
            frame = np.zeros((300, 400, 3), np.uint8)
        th = int(frame.shape[0] * args.tile / frame.shape[1])
        tile = cv2.resize(frame, (args.tile, th))
        tile = cv2.resize(tile, (args.tile, int(args.tile * 0.75)))  # uniform 4:3 tiles
        cv2.rectangle(tile, (0, 0), (150, 28), (0, 0, 0), -1)
        cv2.putText(tile, v.stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)

    blank = np.zeros_like(tiles[0])
    while len(tiles) % args.cols:
        tiles.append(blank)
    rows = [np.hstack(tiles[i : i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
