# Edge-Based Real-Time Restricted Zone Parking Violation Detection

Detects vehicles with a COCO-pretrained YOLO model (YOLOv8n / YOLO11n), checks
whether they stay inside a restricted-zone polygon longer than a dwell threshold,
and raises alerts over MQTT to a monitoring dashboard. All inference runs on the
(simulated) edge device; the edge device is emulated with a resource-limited
container calibrated against published Raspberry Pi 5 benchmarks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Step 1 — baseline detection

Put a test video in `data/videos/` (e.g. `sample.mp4`), then:

```powershell
python scripts/run_detect.py --source data/videos/sample.mp4 --show
python scripts/run_detect.py --source data/videos/sample.mp4 --save outputs/detect.mp4
```

The script prints mean / median / p95 inference latency and FPS.

## Project layout

```
configs/        runtime configuration (model, classes, source)
src/edgepark/   library code (detector, video sources, drawing)
scripts/        runnable entry points
models/         downloaded / exported weights (git-ignored)
data/videos/    test footage (git-ignored)
outputs/        generated results (git-ignored)
```

## Roadmap

1. Baseline vehicle detection (PyTorch, laptop)
2. Restricted zone + tracking + dwell-time violation logic
3. MQTT alerts + monitoring dashboard
4. Docker simulation of the edge device (CPU/RAM limits, RTSP camera stream)
5. Model optimisation (ONNX / NCNN / TFLite / INT8) + benchmarking
6. Evaluation (detection mAP, alert false/duplicate rate)
