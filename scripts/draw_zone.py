"""Draw restricted-zone polygons on a video frame and save them to YAML.

    python scripts/draw_zone.py --video data/videos/islab/ISLab-07.mp4
    python scripts/draw_zone.py --video data/videos/islab/ISLab-08.mp4 --copy-from ISLab-07

Controls (in the window):
    left click   add a point to the current polygon
    right click  undo last point
    n            finish current polygon and start a new one
    a / d        step backward / forward 5 s through the video (pick a clear frame)
    s            save and quit        c  clear all        q / Esc  quit without saving

Saved to configs/zones/<video stem>.yaml (pixel coordinates + frame size).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from edgepark.zones import Zone, load_zones, save_zones, zones_path_for, ZONES_DIR

WIN = "draw zone  |  click: add  right: undo  n: new  a/d: seek  s: save  q: quit"


def read_frame(cap: cv2.VideoCapture, sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", default="restricted", help="Base name for zones")
    ap.add_argument("--at", type=float, default=0.0, help="Start at this second")
    ap.add_argument("--copy-from", default=None, help="Start from another video's zones (same scene)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    sec = args.at
    frame = read_frame(cap, sec)
    if frame is None:
        raise SystemExit(f"Cannot read {args.video}")
    h, w = frame.shape[:2]

    polygons: list[list[tuple[int, int]]] = []
    existing = zones_path_for(args.copy_from) if args.copy_from else zones_path_for(args.video)
    if existing.exists():
        polygons = [[tuple(map(int, p)) for p in z.polygon] for z in load_zones(existing, (w, h))]
        print(f"Loaded {len(polygons)} zone(s) from {existing}")
    current: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, w * 2 if w < 800 else w, h * 2 if w < 800 else h)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        view = frame.copy()
        overlay = view.copy()
        for poly in polygons:
            cv2.fillPoly(overlay, [np.array(poly, np.int32)], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, view, 0.75, 0, dst=view)
        for poly in polygons:
            cv2.polylines(view, [np.array(poly, np.int32)], True, (0, 200, 0), 2)
        if current:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 220, 255), 2)
            for p in current:
                cv2.circle(view, p, 4, (0, 220, 255), -1)
        cv2.putText(view, f"t={sec:.0f}s  zones={len(polygons)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WIN, view)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord("n") and len(current) >= 3:
            polygons.append(current.copy())
            current.clear()
        elif key == ord("c"):
            polygons.clear()
            current.clear()
        elif key in (ord("a"), ord("d")):
            sec = max(0.0, sec + (5 if key == ord("d") else -5))
            f = read_frame(cap, sec)
            if f is not None:
                frame = f
        elif key == ord("s"):
            if len(current) >= 3:
                polygons.append(current.copy())
            if not polygons:
                print("Nothing to save.")
                continue
            zones = [
                Zone(name=args.name if len(polygons) == 1 else f"{args.name}-{i + 1}", polygon=np.array(p, np.float32))
                for i, p in enumerate(polygons)
            ]
            out = zones_path_for(args.video)
            save_zones(out, zones, (w, h), video=args.video.replace("\\", "/").split("/")[-1])
            print(f"Saved {len(zones)} zone(s) to {out}")
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ZONES_DIR.mkdir(parents=True, exist_ok=True)
    main()
