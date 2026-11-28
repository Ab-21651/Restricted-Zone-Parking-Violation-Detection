"""Calibrate the edge container's CPU limit against a real Raspberry Pi 5.

Runs scripts/benchmark.py (val mode = the published Ultralytics protocol) inside
the edgepark-node image at several --cpus limits, then interpolates the limit at
which inference time equals the published Raspberry Pi 5 figure.

Reference (Ultralytics "Raspberry Pi" guide, Raspberry Pi 5, FP32, imgsz 640,
coco128, inference only): YOLO26n PyTorch = 299.09 ms/im.

    python scripts/calibrate_cpus.py                         # default sweep
    python scripts/calibrate_cpus.py --cpus 2 2.2 2.4 2.6    # finer sweep
Writes outputs/bench/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PI5_REFERENCE_MS = {"yolo26n.pt": 299.09}


def run_in_container(cpus: float, model: str, image: str, memory: str) -> dict:
    vol = lambda p, ro=False: ["-v", f"{PROJECT / p}:/app/{p}" + (":ro" if ro else "")]
    cmd = (
        ["docker", "run", "--rm", f"--cpus={cpus}", f"--memory={memory}"]
        + vol("models") + vol("data", ro=True) + vol("outputs") + vol("scripts", ro=True)
        + [image, "python", "scripts/benchmark.py", "--mode", "val", "--model", model, "--label", f"calib-cpus-{cpus}"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"benchmark failed at cpus={cpus}:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
    return json.loads(lines[-1])


def interpolate(points: list[tuple[float, float]], target: float) -> float | None:
    """CPU limit giving ``target`` ms, interpolating in log-log space
    (latency ~ a * cpus^-b) between the two bracketing measurements."""
    pts = sorted(points)
    for (c1, l1), (c2, l2) in zip(pts, pts[1:]):
        if min(l1, l2) <= target <= max(l1, l2) and l1 != l2:
            f = (math.log(target) - math.log(l1)) / (math.log(l2) - math.log(l1))
            return round(math.exp(math.log(c1) + f * (math.log(c2) - math.log(c1))), 2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpus", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--target-ms", type=float, default=None, help="Default: published Pi 5 value for the model")
    ap.add_argument("--image", default="edgepark-node:latest")
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--repeats", type=int, default=1, help="Runs per setting; the median is used")
    args = ap.parse_args()

    target = args.target_ms or PI5_REFERENCE_MS.get(args.model)
    if target is None:
        raise SystemExit(f"No Pi 5 reference for {args.model}; pass --target-ms")

    sweep = []
    for c in args.cpus:
        runs = [run_in_container(c, args.model, args.image, args.memory) for _ in range(args.repeats)]
        ms = sorted(r["inference_ms"] for r in runs)
        med = statistics.median(ms)
        print(f"cpus={c:<5} inference median {med:7.1f} ms/im  (runs: {', '.join(f'{m:.0f}' for m in ms)})  "
              f"mAP50-95 {runs[0]['map50_95']}", flush=True)
        sweep.append({"cpus": c, "inference_ms": round(med, 1), "runs_ms": ms, "map50_95": runs[0]["map50_95"]})

    best = interpolate([(s["cpus"], s["inference_ms"]) for s in sweep], target)
    summary = {
        "model": args.model,
        "reference": "Ultralytics Raspberry Pi guide, Raspberry Pi 5, FP32, imgsz 640, coco128",
        "pi5_inference_ms": target,
        "repeats": args.repeats,
        "sweep": sweep,
        "calibrated_cpus": best,
    }
    out = PROJECT / "outputs" / "bench" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if best:
        print(f"\nPi 5 target {target} ms  ->  calibrated limit: --cpus {best}   (saved {out})")
    else:
        print(f"\nTarget {target} ms is outside the measured range - extend --cpus. (saved {out})")


if __name__ == "__main__":
    main()
