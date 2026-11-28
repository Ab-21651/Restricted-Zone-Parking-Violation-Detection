"""Step 6b: alert-quality evaluation against the ground truth (annotations/islab_events.csv).

Replays the cached detections + tracks (scripts/cache_tracks.py) through five
alerting variants, so they differ only in the alert logic:

  A raw           an alert for every analysed frame with a vehicle in a zone
  B track_only    one alert per tracker ID when it is first seen in a zone
  C dwell_zone    zone-level timer without tracking: alert when a zone stays occupied >= T
  D track_dwell   per-vehicle tracking + dwell time, without duplicate merging / ID re-association
  E full          the deployed system (tracking + dwell + duplicate merging + re-association)

Scoring (per video, per threshold T):
  * an alert is matched to a ground-truth event of the same zone and vehicle type
    (car-like vs motorcycle) whose vehicle is present at the alert time (+- tolerance);
  * the first alert on a GT violation (event lasting >= T) is a true positive, later ones
    are duplicates; alerts on stops shorter than T, or with no GT stop at all, are false
    alarms; alerts on GT events marked ambiguous are ignored;
  * GT violations with no alert are misses.
  precision = TP / (TP + FP + duplicates), recall = TP / GT violations.

    python scripts/eval_alerts.py                       # T = 10, 30, 60 s, all variants
Writes outputs/eval/alerts_summary.csv and outputs/eval/alerts_detail.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from edgepark.config import load_config
from edgepark.detector import Detection
from edgepark.dwell import DwellConfig, ViolationMonitor
from edgepark.zones import load_zones, zones_path_for

VARIANTS = ["base", "raw", "track_only", "dwell_zone", "track_dwell", "full"]
VARIANT_LABELS = {
    "base": "Base frame-wise ROI (box centre)",
    "raw": "A raw per-frame", "track_only": "B tracking only", "dwell_zone": "C dwell, no tracking",
    "track_dwell": "D tracking + dwell (no merging)", "full": "E full system",
}


def group(cls_name: str) -> str:
    return "motorcycle" if cls_name == "motorcycle" else "car"


def load_gt(path: Path) -> dict[str, list[dict]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    gt = defaultdict(list)
    for r in csv.DictReader(lines):
        for k in ("stop_s", "leave_s", "precision_s"):
            r[k] = float(r[k])
        r["ambiguous"] = r["ambiguous"] == "1"
        r["group"] = group(r["vehicle"])
        gt[r["video"]].append(r)
    return gt


def load_cache(path: Path, names: dict[int, str]) -> tuple[dict, list[tuple[float, list[Detection]]]]:
    frames = []
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        for line in f:
            r = json.loads(line)
            dets = [Detection(xyxy=tuple(d[:4]), cls_id=int(d[4]), cls_name=names.get(int(d[4]), str(d[4])),
                              conf=float(d[5]), track_id=d[6]) for d in r["d"]]
            frames.append((r["t"], dets))
    return meta, frames


def zone_of(zones, det: Detection) -> str | None:
    for z in zones:
        if z.contains(det.bottom_center):
            return z.name
    return None


# ------------------------------------------------------------------ variants
def alerts_base(frames, zones, T, dcfg):
    """Replication of the base paper's post-processing ROI rule (da Luz et al.):
    a detection counts when the CENTRE of its box lies inside the region; every
    analysed frame is evaluated on its own (no identity, no time)."""
    out = []
    for t, dets in frames:
        for d in dets:
            cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2.0, (d.xyxy[1] + d.xyxy[3]) / 2.0
            for z in zones:
                if z.contains((cx, cy)):
                    out.append({"t": t, "zone": z.name, "group": group(d.cls_name)})
                    break
    return out


def alerts_raw(frames, zones, T, dcfg):
    out = []
    for t, dets in frames:
        for d in dets:
            z = zone_of(zones, d)
            if z:
                out.append({"t": t, "zone": z, "group": group(d.cls_name)})
    return out


def alerts_track_only(frames, zones, T, dcfg):
    out, seen = [], set()
    for t, dets in frames:
        for d in dets:
            z = zone_of(zones, d)
            if z and d.track_id not in seen:
                seen.add(d.track_id)
                out.append({"t": t, "zone": z, "group": group(d.cls_name)})
    return out


def alerts_dwell_zone(frames, zones, T, dcfg):
    """Zone occupancy timer, no identities: occupied while any vehicle is inside
    (gaps up to grace_s bridged); one alert per occupancy period lasting >= T."""
    out = []
    state = {z.name: {"since": None, "last": None, "alerted": False} for z in zones}
    for t, dets in frames:
        inside = defaultdict(list)
        for d in dets:
            z = zone_of(zones, d)
            if z:
                inside[z].append(d)
        for name, s in state.items():
            if inside[name]:
                if s["since"] is None or t - s["last"] > dcfg.grace_s:
                    s["since"], s["alerted"] = t, False
                s["last"] = t
                if not s["alerted"] and t - s["since"] >= T:
                    s["alerted"] = True
                    big = max(inside[name], key=lambda d: d.height)
                    out.append({"t": t, "zone": name, "group": group(big.cls_name)})
    return out


def alerts_monitor(frames, zones, T, dcfg, merge: bool):
    cfg = DwellConfig(**{**dcfg.__dict__, "threshold_s": T})
    if not merge:  # disable ID re-association and duplicate-box merging
        cfg.reassoc_iou, cfg.dup_iou, cfg.dup_point_ratio = 2.0, 2.0, 0.0
    mon, out = ViolationMonitor(zones, cfg), []
    for t, dets in frames:
        for ev in mon.update(t, dets):
            if ev["event"] == "violation":
                out.append({"t": t, "zone": ev["zone"], "group": group(ev["class"]), "track": ev["track_id"]})
    return out


def run_variant(name, frames, zones, T, dcfg):
    if name == "base":
        return alerts_base(frames, zones, T, dcfg)
    if name == "raw":
        return alerts_raw(frames, zones, T, dcfg)
    if name == "track_only":
        return alerts_track_only(frames, zones, T, dcfg)
    if name == "dwell_zone":
        return alerts_dwell_zone(frames, zones, T, dcfg)
    return alerts_monitor(frames, zones, T, dcfg, merge=(name == "full"))


# ------------------------------------------------------------------- scoring
def score(alerts, gts, T, tol):
    res = {"tp": 0, "fp": 0, "dup": 0, "ignored": 0, "fn": 0, "gt": 0, "delays": []}
    hit = set()
    for a in sorted(alerts, key=lambda a: a["t"]):
        cands = [g for g in gts if g["zone"] == a["zone"] and g["group"] == a["group"]
                 and g["stop_s"] - g["precision_s"] - tol <= a["t"] <= g["leave_s"] + g["precision_s"] + tol]
        if not cands:
            res["fp"] += 1
            continue
        # prefer an unmatched violation, then any violation, then an ambiguous / short stop;
        # among equals, the vehicle whose expected alert time (stop + T) is closest - this
        # separates two vehicles of the same type parked in the same zone at the same time
        cands.sort(key=lambda g: (g["ambiguous"], g["leave_s"] - g["stop_s"] < T, id(g) in hit,
                                  abs(a["t"] - (g["stop_s"] + T))))
        g = cands[0]
        if g["ambiguous"]:
            res["ignored"] += 1
        elif g["leave_s"] - g["stop_s"] < T:
            res["fp"] += 1
        elif id(g) in hit:
            res["dup"] += 1
        else:
            hit.add(id(g))
            res["tp"] += 1
            res["delays"].append(a["t"] - (g["stop_s"] + T))
    viol = [g for g in gts if not g["ambiguous"] and g["leave_s"] - g["stop_s"] >= T]
    res["gt"] = len(viol)
    res["fn"] = sum(1 for g in viol if id(g) not in hit)
    return res


def summarize(r):
    alerts = r["tp"] + r["fp"] + r["dup"]
    p = r["tp"] / alerts if alerts else 0.0
    rec = r["tp"] / r["gt"] if r["gt"] else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    d = r["delays"]
    return {"alerts": alerts, "TP": r["tp"], "FP": r["fp"], "duplicates": r["dup"], "missed": r["fn"],
            "gt_violations": r["gt"], "ignored": r["ignored"], "precision": round(p, 3), "recall": round(rec, 3),
            "F1": round(f1, 3), "mean_delay_s": round(sum(d) / len(d), 1) if d else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="annotations/islab_events.csv")
    ap.add_argument("--tracks", default="outputs/tracks")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 30, 60])
    ap.add_argument("--tol", type=float, default=5.0, help="time tolerance (s) around GT presence")
    ap.add_argument("--grace", type=float, default=None, help="override dwell.grace_s (dev-set tuning)")
    ap.add_argument("--out", default="outputs/eval", help="output folder")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = DwellConfig.from_dict(cfg.get("dwell"))
    if args.grace is not None:
        dcfg.grace_s = args.grace
    gt = load_gt(Path(args.gt))
    caches = sorted(Path(args.tracks).glob("ISLab-*.jsonl"))
    if not caches:
        raise SystemExit("No cached tracks - run scripts/cache_tracks.py first")
    split_of = {}
    for g in (g for v in gt.values() for g in v):
        split_of[g["video"]] = g["split"]
    dev = {"ISLab-01", "ISLab-04", "ISLab-07", "ISLab-08", "ISLab-15"}

    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "dup": 0, "ignored": 0, "fn": 0, "gt": 0, "delays": []})
    detail = []
    for cache in caches:
        video = cache.stem
        meta, frames = load_cache(cache, cfg["vehicle_classes"])
        w, h = 640, 480
        from edgepark.video import source_info

        info = source_info(f"data/videos/islab/{video}.mp4")
        zones = load_zones(zones_path_for(video), (info["width"] or w, info["height"] or h))
        split = "dev" if video in dev else "test"
        for T in args.thresholds:
            for v in VARIANTS:
                alerts = run_variant(v, frames, zones, T, dcfg)
                r = score(alerts, gt.get(video, []), T, args.tol)
                detail.append({"video": video, "split": split, "T": T, "variant": v, **summarize(r)})
                for key in ((split, T, v), ("all", T, v)):
                    for k in ("tp", "fp", "dup", "ignored", "fn", "gt"):
                        totals[key][k] += r[k]
                    totals[key]["delays"] += r["delays"]
        print(f"{video}: {len(frames)} frames evaluated", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alerts_detail.json").write_text(json.dumps({"model": meta["model"], "fps": meta["fps"], "rows": detail}, indent=2))
    rows = []
    for (split, T, v), r in sorted(totals.items(), key=lambda x: (x[0][0], x[0][1], VARIANTS.index(x[0][2]))):
        rows.append({"split": split, "T": int(T), "variant": VARIANT_LABELS[v], **summarize(r)})
    with open(out / "alerts_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\nModel {meta['model']} at {meta['fps']} fps; tolerance {args.tol}s; grace {dcfg.grace_s}s")
    print(f"{'split':5s} {'T':>3s}  {'variant':34s} {'alerts':>6s} {'TP':>3s} {'FP':>5s} {'dup':>5s} {'miss':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'delay':>6s}")
    for r in rows:
        print(f"{r['split']:5s} {r['T']:>3d}  {r['variant']:34s} {r['alerts']:>6d} {r['TP']:>3d} {r['FP']:>5d} "
              f"{r['duplicates']:>5d} {r['missed']:>4d} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['F1']:>6.3f} "
              f"{r['mean_delay_s'] if r['mean_delay_s'] is not None else '-':>6}")


if __name__ == "__main__":
    main()
