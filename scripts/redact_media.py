"""Redact images / videos before sharing them publicly (report, slides, demo video).

People are blurred; with --vehicles, vehicles are pixelated as well so number
plates cannot be read (the proposal's ethics commitment for shared material).

    python scripts/redact_media.py outputs/figures/snap.jpg --vehicles
    python scripts/redact_media.py outputs/pipeline_07.mp4 --vehicles --out outputs/public/pipeline_07.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from edgepark.detector import resolve_model_path
from edgepark.privacy import PrivacyFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--vehicles", action="store_true", help="Also pixelate vehicles (hides number plates)")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--out", default=None, help="Output file (single input) or folder (default outputs/public)")
    args = ap.parse_args()

    pf = PrivacyFilter(resolve_model_path(args.model), conf=0.2)
    out_dir = Path("outputs/public")
    for inp in map(Path, args.inputs):
        dest = Path(args.out) if args.out and len(args.inputs) == 1 else (Path(args.out) if args.out else out_dir) / inp.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if inp.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(inp))
            red, n = pf.apply(img, redact_vehicles=args.vehicles)
            cv2.imwrite(str(dest), red)
            print(f"{inp} -> {dest}  ({n} regions hidden)")
            continue
        cap = cv2.VideoCapture(str(inp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer, total, frames = None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            red, n = pf.apply(frame, redact_vehicles=args.vehicles)
            if writer is None:
                h, w = red.shape[:2]
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(red)
            total += n
            frames += 1
        cap.release()
        if writer:
            writer.release()
        print(f"{inp} -> {dest}  ({frames} frames, {total} regions hidden)")


if __name__ == "__main__":
    main()
