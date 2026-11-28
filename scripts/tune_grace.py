"""Tune dwell.grace_s (how long a vehicle may disappear before its timer is dropped)
on the DEV videos only, with the full system at T = 60 s. Test videos are not used.

    python scripts/tune_grace.py
Writes outputs/eval/grace_sweep.json; set the chosen value in configs/default.yaml.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run from the project root
from eval_alerts import load_cache, load_gt, run_variant, score, summarize
from edgepark.config import load_config
from edgepark.dwell import DwellConfig
from edgepark.video import source_info
from edgepark.zones import load_zones, zones_path_for

DEV = ["ISLab-01", "ISLab-04", "ISLab-07", "ISLab-08", "ISLab-15"]
GRACES = [5.0, 10.0, 15.0, 20.0]
T = 60.0


def main() -> None:
    cfg = load_config()
    gt = load_gt(Path("annotations/islab_events.csv"))
    data = {}
    for v in DEV:
        _, frames = load_cache(Path(f"outputs/tracks/{v}.jsonl"), cfg["vehicle_classes"])
        info = source_info(f"data/videos/islab/{v}.mp4")
        data[v] = (frames, load_zones(zones_path_for(v), (info["width"], info["height"])))
    rows = []
    for g in GRACES:
        dcfg = DwellConfig.from_dict(cfg.get("dwell"))
        dcfg.grace_s = g
        tot = defaultdict(int)
        tot["delays"] = []
        for v, (frames, zones) in data.items():
            r = score(run_variant("full", frames, zones, T, dcfg), gt.get(v, []), T, 5.0)
            for k in ("tp", "fp", "dup", "ignored", "fn", "gt"):
                tot[k] += r[k]
            tot["delays"] += r["delays"]
        s = summarize(tot)
        rows.append({"grace_s": g, **s})
        print(f"grace {g:4.0f}s: alerts {s['alerts']} TP {s['TP']} FP {s['FP']} dup {s['duplicates']} "
              f"missed {s['missed']}  P {s['precision']} R {s['recall']} F1 {s['F1']}")
    best = max(rows, key=lambda r: (r["F1"], -r["grace_s"]))
    out = {"split": "dev", "videos": DEV, "T": T, "variant": "full", "rows": rows, "chosen": best["grace_s"]}
    Path("outputs/eval").mkdir(parents=True, exist_ok=True)
    Path("outputs/eval/grace_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"chosen grace_s = {best['grace_s']} (dev F1 {best['F1']})")


if __name__ == "__main__":
    main()
