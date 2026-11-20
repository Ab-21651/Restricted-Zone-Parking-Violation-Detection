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
Add `--max-frames 300 --snapshot outputs/snap.jpg` for a quick run that saves one annotated frame.

Dataset: ISLab PVD videos go in `data/videos/islab/` (`ISLab-01.mp4` … `ISLab-16.mp4`).
`python scripts/preview_videos.py` builds `outputs/contact_sheet.jpg` with one frame per video.

## Step 2 — restricted zone + tracking + dwell time

1. Draw the restricted zone(s) for a video (saved to `configs/zones/<video>.yaml`):

   ```powershell
   python scripts/draw_zone.py --video data/videos/islab/ISLab-07.mp4
   python scripts/draw_zone.py --video data/videos/islab/ISLab-08.mp4 --copy-from ISLab-07   # same scene
   ```

   Left click adds a point, right click undoes, `n` starts another polygon, `a`/`d` seek ±5 s, `s` saves.

2. Run the pipeline:

   ```powershell
   python scripts/run_pipeline.py --source data/videos/islab/ISLab-07.mp4 --show
   python scripts/run_pipeline.py --source data/videos/islab/ISLab-07.mp4 --save outputs/pipeline_07.mp4
   ```

How it works: YOLO + ByteTrack give every vehicle a persistent id. A vehicle whose
ground point (bottom-centre of its box) is inside a zone and which has moved less
than 15 % of its box height over 2 s counts as *stopped*; after `dwell.threshold_s`
(30 s) of stopped time one `violation` event is raised, and a `cleared` event when
it leaves or disappears for more than 5 s. Timing uses video timestamps. Duplicate
boxes and tracker id switches are merged so one vehicle gives one alert.
Events go to `outputs/events/<video>.jsonl`; all parameters are in `configs/default.yaml`.

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
