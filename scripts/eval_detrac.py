"""Step 5: vehicle-detection accuracy on UA-DETRAC (sampled test frames).

Every vehicle annotated in DETRAC (car, van, bus, others) is one "vehicle" class;
the detector's COCO vehicle classes (car, motorcycle, bus, truck) are merged
into the same class (class-agnostic NMS, as in the pipeline). Detections whose
centre lies in a DETRAC "ignored region" are discarded, following the DETRAC
protocol. Metrics use Ultralytics' own AP routine (COCO-style, 101-point):
precision / recall / F1 at the best-F1 confidence, mAP@0.5 and mAP@0.5:0.95.

    python scripts/eval_detrac.py --model yolov8n.pt
    python scripts/eval_detrac.py --model models/yolo26n_ncnn_model --label yolo26n-ncnn

Writes outputs/eval/detrac_<label>.json and appends a row to outputs/eval/detrac_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

ANN_DIRS = [Path("data/detrac/annotations/DETRAC-Test-Annotations-XML"),
            Path("data/detrac/annotations/DETRAC-Train-Annotations-XML")]
IOUV = np.linspace(0.5, 0.95, 10)
COCO_VEHICLES = [2, 3, 5, 7]


def load_annotations(seq: str) -> tuple[dict[int, np.ndarray], np.ndarray, str]:
    """frame number -> (N,4) xyxy ground truth; ignored regions (M,4); weather."""
    for d in ANN_DIRS:
        f = d / f"{seq}.xml"
        if f.exists():
            root = ET.parse(f).getroot()
            break
    else:
        raise FileNotFoundError(seq)
    weather = root.find("sequence_attribute").get("sence_weather")
    ign = [[float(b.get("left")), float(b.get("top")),
            float(b.get("left")) + float(b.get("width")), float(b.get("top")) + float(b.get("height"))]
           for b in root.find("ignored_region").iter("box")]
    gts = {}
    for fr in root.iter("frame"):
        boxes = []
        for b in fr.iter("box"):
            x, y, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            boxes.append([x, y, x + w, y + h])
        gts[int(fr.get("num"))] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return gts, np.array(ign, dtype=np.float32).reshape(-1, 4), weather


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n_pred, 10) true-positive matrix, one-to-one matching per IoU threshold
    (same procedure as Ultralytics' DetectionValidator.match_predictions)."""
    tp = np.zeros((len(pred), len(IOUV)), dtype=bool)
    iou = iou_matrix(gt, pred)
    for i, thr in enumerate(IOUV):
        g, p = np.nonzero(iou >= thr)
        if not len(g):
            continue
        m = np.stack([g, p, iou[g, p]], 1)
        m = m[m[:, 2].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1].astype(int), i] = True
    return tp


def metrics(tp: list, conf: list, n_gt: int) -> dict:
    from ultralytics.utils.metrics import ap_per_class

    if not tp:
        return {"P": 0, "R": 0, "F1": 0, "mAP50": 0, "mAP50_95": 0, "n_gt": n_gt, "n_det": 0}
    tp_a, conf_a = np.concatenate(tp), np.concatenate(conf)
    out = ap_per_class(tp_a, conf_a, np.zeros(len(conf_a)), np.zeros(n_gt))
    p, r, f1, ap = out[2], out[3], out[4], out[5]
    return {"P": round(float(p[0]), 4), "R": round(float(r[0]), 4), "F1": round(float(f1[0]), 4),
            "mAP50": round(float(ap[0, 0]), 4), "mAP50_95": round(float(ap[0].mean()), 4),
            "n_gt": int(n_gt), "n_det": int(len(conf_a))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--images", default="data/detrac/images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low, as for mAP")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    args = ap.parse_args()

    from ultralytics import YOLO

    from edgepark.detector import resolve_model_path

    path = resolve_model_path(args.model) if args.model.endswith(".pt") else args.model
    label = args.label or Path(path).stem.replace("_ncnn_model", "-ncnn")
    model = YOLO(path, task="detect")

    groups: dict[str, dict] = defaultdict(lambda: {"tp": [], "conf": [], "n_gt": 0})
    infer_ms: list[float] = []
    n_img = 0
    t0 = time.time()
    for seq_dir in sorted(p for p in Path(args.images).iterdir() if p.is_dir()):
        gts, ign, weather = load_annotations(seq_dir.name)
        for img in sorted(seq_dir.glob("img*.jpg")):
            num = int(img.stem[3:])
            if num not in gts:
                continue
            r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, iou=args.iou, classes=COCO_VEHICLES,
                              agnostic_nms=True, max_det=300, verbose=False, device="cpu")[0]
            infer_ms.append(r.speed["inference"])
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            if len(ign) and len(boxes):  # drop detections centred in ignored regions
                cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
                inside = ((cx[:, None] >= ign[:, 0]) & (cx[:, None] <= ign[:, 2]) &
                          (cy[:, None] >= ign[:, 1]) & (cy[:, None] <= ign[:, 3])).any(1)
                boxes, confs = boxes[~inside], confs[~inside]
            tp = match(gts[num], boxes)
            for key in ("all", weather):
                groups[key]["tp"].append(tp)
                groups[key]["conf"].append(confs)
                groups[key]["n_gt"] += len(gts[num])
            n_img += 1
        print(f"  {seq_dir.name} ({weather}) done, {n_img} images so far", flush=True)

    result = {
        "label": label, "model": str(path), "imgsz": args.imgsz, "images": n_img,
        "seconds": round(time.time() - t0, 1),
        "infer_ms_mean": round(statistics.mean(infer_ms), 1) if infer_ms else None,
        "metrics": {k: metrics(v["tp"], v["conf"], v["n_gt"]) for k, v in groups.items()},
    }
    out = Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"detrac_{label}.json").write_text(json.dumps(result, indent=2))
    summary = out / "detrac_summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "subset", "images_gt", "P", "R", "F1", "mAP50", "mAP50_95", "infer_ms_host"])
        for k, m in result["metrics"].items():
            w.writerow([label, k, m["n_gt"], m["P"], m["R"], m["F1"], m["mAP50"], m["mAP50_95"], result["infer_ms_mean"]])
    a = result["metrics"]["all"]
    print(f"{label}: {n_img} images | P {a['P']} R {a['R']} F1 {a['F1']} | mAP50 {a['mAP50']} mAP50-95 {a['mAP50_95']}")
    for k, m in result["metrics"].items():
        if k != "all":
            print(f"   {k:7s} mAP50 {m['mAP50']}  mAP50-95 {m['mAP50_95']}  (gt boxes {m['n_gt']})")


if __name__ == "__main__":
    main()
