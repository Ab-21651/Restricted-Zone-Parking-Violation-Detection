"""Results figures (400 DPI PNG + vector PDF) from the saved benchmark / evaluation files.

    python scripts/make_figures.py
Inputs : outputs/eval/detrac_summary.csv, outputs/bench/pi_sim.json, outputs/bench/calibration.json
Outputs: outputs/figures/*.png|pdf
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/figures")
MODELS = ["yolov8n", "yolo11n", "yolo26n"]
LABELS = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo26n": "YOLO26n"}
COLORS = {"yolov8n": "#4C72B0", "yolo11n": "#DD8452", "yolo26n": "#55A868"}
FMT_COLORS = {"pytorch": "#4C72B0", "onnx": "#DD8452", "ncnn": "#55A868"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
    "savefig.dpi": 400, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {OUT / name}.png/.pdf")


def detrac_accuracy() -> None:
    f = Path("outputs/eval/detrac_summary.csv")
    if not f.exists():
        return
    rows = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        rows[(r["label"], r["subset"])] = r  # last run wins
    subsets = ["all", "sunny", "cloudy", "rainy", "night"]
    models = [m for m in MODELS if (m, "all") in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [float(rows[(m, s)]["mAP50_95"]) for s in subsets if (m, s) in rows]
        xs = [j + (i - (len(models) - 1) / 2) * w for j in range(len(vals))]
        ax.bar(xs, vals, w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(subsets)), ["Overall", "Sunny", "Cloudy", "Rainy", "Night"])
    ax.set_ylabel("mAP@0.5:0.95 (vehicle)")
    ax.set_ylim(0, 0.8)
    ax.set_title("UA-DETRAC vehicle detection accuracy")
    ax.legend(ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_detrac_map_by_weather")


def pi_sim_speed() -> None:
    f = Path("outputs/bench/pi_sim.json")
    if not f.exists():
        return
    rows = json.loads(f.read_text())
    fmts = [x for x in ["pytorch", "onnx", "ncnn"] if any(r["format"] == x for r in rows)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.8 / max(len(fmts), 1)
    for i, fmt in enumerate(fmts):
        xs, ys = [], []
        for j, m in enumerate(MODELS):
            r = next((r for r in rows if r["model"] == m and r["format"] == fmt), None)
            if r:
                xs.append(j + (i - (len(fmts) - 1) / 2) * w)
                ys.append(r["inference_ms"])
        ax.bar(xs, ys, w, label={"pytorch": "PyTorch", "onnx": "ONNX", "ncnn": "NCNN"}[fmt], color=FMT_COLORS[fmt])
    ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
    ax.set_ylabel("Inference time (ms / image)")
    ax.set_title("Speed on simulated Raspberry Pi 5")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "fig_pi_sim_speed_by_format")

    # simulated vs published Pi 5 (YOLO26n)
    pts = [r for r in rows if r["model"] == "yolo26n" and r.get("pi5_published_ms")]
    if pts:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        lim = max(max(r["inference_ms"], r["pi5_published_ms"]) for r in pts) * 1.15
        ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8, label="perfect prediction")
        for r in pts:
            ax.scatter(r["pi5_published_ms"], r["inference_ms"], color=FMT_COLORS[r["format"]], s=30, zorder=3)
            ax.annotate(f"{r['format'].upper()} ({r['error_pct']:+.0f}%)", (r["pi5_published_ms"], r["inference_ms"]),
                        textcoords="offset points", xytext=(5, -10), fontsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Published Raspberry Pi 5 (ms / image)")
        ax.set_ylabel("Calibrated container (ms / image)")
        ax.set_title("Does the simulation predict the Pi 5? (YOLO26n)")
        ax.legend(frameon=False, loc="upper left")
        save(fig, "fig_sim_vs_real_pi5")


def calibration_curve() -> None:
    f = Path("outputs/bench/calibration.json")
    if not f.exists():
        return
    c = json.loads(f.read_text())
    pts = sorted((s["cpus"], s["inference_ms"]) for s in c["sweep"])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#4C72B0", label="container (median of 3)")
    ax.axhline(c["pi5_inference_ms"], color="#C44E52", ls="--", lw=1, label=f"Raspberry Pi 5 ({c['pi5_inference_ms']:.0f} ms)")
    if c.get("calibrated_cpus"):
        ax.axvline(c["calibrated_cpus"], color="grey", ls=":", lw=1)
        ax.annotate(f"{c['calibrated_cpus']} CPUs", (c["calibrated_cpus"], c["pi5_inference_ms"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_xlabel("Docker CPU limit (--cpus)")
    ax.set_ylabel("YOLO26n inference (ms / image)")
    ax.set_title("CPU-limit calibration against Raspberry Pi 5")
    ax.legend(frameon=False)
    save(fig, "fig_calibration_curve")


def alert_variants() -> None:
    f = Path("outputs/eval/alerts_summary.csv")
    if not f.exists():
        return
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    test60 = [r for r in rows if r["split"] == "test" and r["T"] == "60"]
    if test60:
        labels = [r["variant"].split(" ", 1)[1] for r in test60]
        short = [r["variant"].split(" ")[0] for r in test60]
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        x = range(len(test60))
        ax.bar([i - 0.2 for i in x], [float(r["precision"]) for r in test60], 0.4, label="Precision", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], [float(r["recall"]) for r in test60], 0.4, label="Recall", color="#DD8452")
        for i, r in enumerate(test60):
            ax.text(i, 1.03, f"{r['alerts']} alerts", ha="center", fontsize=6.5, rotation=0)
        ax.set_xticks(list(x), short)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (test videos, T = 60 s)")
        ax.set_title("Alerting variants: A raw ... E full system")
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.text(0.0, -0.16, "; ".join(f"{s} = {l}" for s, l in zip(short, labels)), fontsize=6, wrap=True)
        save(fig, "fig_alert_variants")

    full = [r for r in rows if r["split"] == "test" and r["variant"].startswith("E")]
    if full:
        full.sort(key=lambda r: int(r["T"]))
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        ts = [int(r["T"]) for r in full]
        for key, c in (("precision", "#4C72B0"), ("recall", "#DD8452"), ("F1", "#55A868")):
            ax.plot(ts, [float(r[key]) for r in full], "o-", label=key.capitalize() if key != "F1" else "F1", color=c)
        ax.set_xticks(ts)
        ax.set_xlabel("Dwell threshold T (s)")
        ax.set_ylabel("Score (test videos)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Full system: sensitivity to the dwell threshold")
        ax.legend(frameon=False)
        save(fig, "fig_threshold_sensitivity")


if __name__ == "__main__":
    detrac_accuracy()
    pi_sim_speed()
    calibration_curve()
    alert_variants()

