"""Diagnose one video: replay the cached tracks through the full system and print,
every few seconds, which vehicles are in a zone, their status and dwell time.

    python scripts/trace_video.py ISLab-13 --T 60 --every 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_alerts import load_cache, zone_of  # noqa: E402
from edgepark.config import load_config  # noqa: E402
from edgepark.dwell import DwellConfig, ViolationMonitor  # noqa: E402
from edgepark.video import source_info  # noqa: E402
from edgepark.zones import load_zones, zones_path_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--T", type=float, default=60)
    ap.add_argument("--every", type=float, default=5)
    args = ap.parse_args()
    cfg = load_config()
    _, frames = load_cache(Path(f"outputs/tracks/{args.video}.jsonl"), cfg["vehicle_classes"])
    info = source_info(f"data/videos/islab/{args.video}.mp4")
    zones = load_zones(zones_path_for(args.video), (info["width"], info["height"]))
    dcfg = DwellConfig.from_dict(cfg.get("dwell"))
    dcfg.threshold_s = args.T
    mon = ViolationMonitor(zones, dcfg)
    nxt = 0.0
    for t, dets in frames:
        for ev in mon.update(t, dets):
            print(f"  EVENT {ev['event']} t={t:.1f} #{ev['track_id']} {ev['class']} {ev['zone']} dwell={ev['dwell_s']}")
        if t >= nxt:
            nxt += args.every
            near = []
            for d in dets:
                z = zone_of(zones, d)
                x, y = (int(v) for v in d.bottom_center)
                st = mon.states.get(d.track_id)
                tag = f"{st.status}:{st.dwell(t):.0f}s" if st else "-"
                near.append(f"#{d.track_id} {d.cls_name} ({x},{y}) h{int(d.height)} {z or 'out'} {tag}")
            print(f"t={t:6.1f}  " + (" | ".join(near) if near else "no vehicles"))


if __name__ == "__main__":
    main()
