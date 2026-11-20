"""Step 2: detection + tracking + restricted zone + dwell time -> violation events.

    python scripts/run_pipeline.py --source data/videos/islab/ISLab-07.mp4 --show
    python scripts/run_pipeline.py --source data/videos/islab/ISLab-07.mp4 --save outputs/ISLab-07.mp4

Zones are read from configs/zones/<video stem>.yaml (draw them with
scripts/draw_zone.py) unless --zones is given. Events are written as JSON lines
to outputs/events/<video stem>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2

from edgepark.config import PROJECT_ROOT, load_config
from edgepark.detector import VehicleDetector
from edgepark.draw import draw_hud, draw_tracks, draw_zones
from edgepark.dwell import DwellConfig, ViolationMonitor
from edgepark.video import FpsMeter, iter_frames, source_info
from edgepark.zones import load_zones, zones_path_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--zones", default=None, help="Zone YAML (default: configs/zones/<video stem>.yaml)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="Override dwell threshold (s)")
    ap.add_argument("--process-fps", type=float, default=None, help="Override frames analysed per second")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default=None, help="Write annotated video here")
    ap.add_argument("--events", default=None, help="Events JSONL (default: outputs/events/<stem>.jsonl)")
    ap.add_argument("--max-seconds", type=float, default=0, help="Stop after this much video time")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source or cfg["source"]["uri"]
    stem = Path(str(source)).stem
    info = source_info(source)

    zones_file = Path(args.zones) if args.zones else zones_path_for(source)
    if not zones_file.exists():
        raise SystemExit(f"No zones file {zones_file}. Draw one: python scripts/draw_zone.py --video {source}")
    zones = load_zones(zones_file, (info["width"], info["height"]))

    mcfg, tcfg = cfg["model"], cfg.get("tracking", {})
    detector = VehicleDetector(
        args.model or mcfg["path"],
        imgsz=mcfg.get("imgsz", 640),
        iou=mcfg.get("iou", 0.5),
        vehicle_classes=cfg["vehicle_classes"],
    )
    detector.warmup()

    dcfg = DwellConfig.from_dict(cfg.get("dwell"))
    if args.threshold is not None:
        dcfg.threshold_s = args.threshold
    monitor = ViolationMonitor(zones, dcfg)
    process_fps = args.process_fps if args.process_fps is not None else cfg.get("pipeline", {}).get("process_fps", 10)

    events_path = Path(args.events) if args.events else PROJECT_ROOT / "outputs" / "events" / f"{stem}.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_file = open(events_path, "w", encoding="utf-8")

    print(f"Source {source}  {info['width']}x{info['height']} @ {info['fps']:.2f} fps  ->  analysing ~{process_fps} fps")
    print(f"Zones {zones_file.name}: {[z.name for z in zones]}  threshold {dcfg.threshold_s}s")

    writer = None
    latencies: list[float] = []
    fps_meter = FpsMeter()
    n_events = {"violation": 0, "cleared": 0}
    t = 0.0

    for idx, t, frame in iter_frames(source, process_fps):
        detections, latency_ms = detector.track(frame, tcfg.get("tracker", "configs/bytetrack.yaml"), tcfg.get("conf", 0.1))
        latencies.append(latency_ms)
        events = monitor.update(t, detections)
        fps = fps_meter.tick()

        for ev in events:
            ev.update({"source": stem, "frame": idx})
            events_file.write(json.dumps(ev) + "\n")
            n_events[ev["event"]] += 1
            print(f"[{t:7.1f}s] {ev['event'].upper():9s} event {ev['event_id']} track #{ev['track_id']} "
                  f"{ev['class']} zone={ev['zone']} dwell={ev['dwell_s']}s")

        if args.show or args.save:
            active = {st.zone for st in monitor.states.values() if st.status == "violation"}
            draw_zones(frame, zones, active)
            draw_tracks(frame, detections, monitor.states, t)
            draw_hud(frame, [f"t {t:.1f}s  infer {latency_ms:.0f} ms  {fps:.1f} FPS",
                             f"violations {n_events['violation']}"])
        if args.save:
            if writer is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), process_fps or info["fps"], (w, h))
            writer.write(frame)
        if args.show:
            cv2.imshow(f"edgepark - {stem}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_seconds and t >= args.max_seconds:
            break

    for ev in monitor.finish(t):
        ev.update({"source": stem, "frame": None})
        events_file.write(json.dumps(ev) + "\n")
        n_events["cleared"] += 1
        print(f"[{t:7.1f}s] CLEARED   event {ev['event_id']} (end of stream) dwell={ev['dwell_s']}s")
    events_file.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if latencies:
        print(f"\nFrames analysed: {len(latencies)}  |  detect+track mean {statistics.mean(latencies):.1f} ms, "
              f"median {statistics.median(latencies):.1f} ms")
    print(f"Violations: {n_events['violation']}  ->  {events_path}")


if __name__ == "__main__":
    main()
