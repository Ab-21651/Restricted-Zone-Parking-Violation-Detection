# Edge-Based Real-Time Restricted Zone Parking Violation Detection

Detects vehicles with a COCO-pretrained YOLO model (YOLOv8n / YOLO11n / YOLO26n compared;
YOLO26n in ONNX format deployed), checks whether they stay inside a restricted-zone
polygon longer than a dwell threshold (60 s), and raises alerts over MQTT to a monitoring
dashboard. All inference runs on the (simulated) edge device; the edge device is emulated
with a resource-limited Docker container calibrated against published Raspberry Pi 5
benchmarks.

**Results:** see [docs/RESULTS.md](docs/RESULTS.md) (generated from `outputs/`), figures in
`outputs/figures/`, zone rules in [docs/zone_policy.md](docs/zone_policy.md), ground truth in
[annotations/islab_events.csv](annotations/islab_events.csv).

Team: Abdullah Attique (23i-0073), Minahil Rizwan (23i-0114), Talha Akram (23i-0139).

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
than 15 % of its box height over 2 s counts as *stopped*; only real movement resets
the timer (too little history - e.g. after a frame gap on a slow edge CPU - does not).
After `dwell.threshold_s` (60 s, the AVSS 2007 / i-LIDS parked-vehicle rule) of stopped
time one `violation` event is raised, and a `cleared` event when it leaves or disappears
for longer than `dwell.grace_s`. Timing uses video timestamps (wall clock on live streams).
Duplicate boxes (overlap or almost the same ground point) and tracker id switches are
merged so one vehicle gives one alert. Events go to `outputs/events/<video>.jsonl`;
all parameters are in `configs/default.yaml`. Zones for all 16 videos follow
`docs/zone_policy.md`.

## Step 3 — MQTT alerts + monitoring dashboard

```powershell
docker compose up -d mosquitto                         # MQTT broker (port 1883)
python scripts/run_dashboard.py                        # http://127.0.0.1:8000
python scripts/run_pipeline.py --source data/videos/islab/ISLab-07.mp4 --mqtt
```

The edge node publishes to `edgepark/<node_id>/events` (violation / cleared, QoS 1,
violation events carry a JPEG snapshot) and `edgepark/<node_id>/status` (heartbeat
every 5 s, retained; Last-Will marks the node offline if it drops). Messages are
buffered while the broker is unreachable and events are always logged locally.
The dashboard receives them over MQTT and pushes them to the browser live
(Server-Sent Events). Each event carries the capture time of its frame, so the
dashboard logs capture-to-dashboard latency to `outputs/dashboard/received.jsonl`.

## Step 4 — Raspberry Pi 5 simulation in Docker

```powershell
docker build -t edgepark-node .                  # CPU-only image (once)
docker compose up -d                             # camera + edge node + broker + dashboard
docker compose logs -f edge-node
docker stats                                     # edge node stays within its CPU limit
$env:CAM="ISLab-04"; docker compose up -d        # another video (CAM = video / zone file name)
```

* `camera`: MediaMTX + ffmpeg loop an ISLab video as `rtsp://camera:8554/cam` (30 fps, no re-encode).
* `edge-node`: limited to `EDGE_CPUS` (default **0.57**) and 4 GB, 4 inference threads like the
  Pi 5's 4 cores. On a live stream it always analyses the newest frame and drops the backlog.
* Calibration (`scripts/calibrate_cpus.py`, result in `outputs/bench/calibration.json`): the
  published Ultralytics Raspberry Pi 5 figure for YOLO26n PyTorch (FP32, 640, coco128) is
  299 ms/image; the container measures 302 ms at 0.57 CPUs (3 runs: 297–304 ms).
  **Run all speed measurements on the charger in Windows "Best performance" mode** — on
  battery-saver the same limit was ~3x slower and very noisy.
* `scripts/benchmark.py --mode val|frames` measures any model / format; results go to
  `outputs/bench/results.jsonl`.

If Docker Desktop fails to start with *"remove ...\dockerInference / engine.sock:
The file cannot be accessed by the system"*, run `.\tools\fix-docker.ps1`
(stale socket files after an unclean shutdown).

## Step 5 — model comparison and optimisation

```powershell
python scripts/export_models.py                  # v8n / 11n / 26n -> ONNX + NCNN (models/)
python scripts/fetch_detrac.py                   # UA-DETRAC sample (needs Kaggle login), 842 frames
python scripts/eval_detrac.py --model yolo26n.pt # vehicle mAP overall + per weather
python scripts/bench_pi_sim.py                   # every model x format in the 0.57-CPU container
```

Detection accuracy is measured on 842 UA-DETRAC test frames (8 sequences: sunny, cloudy,
rainy, night; ignored regions respected). Speed uses the published Ultralytics Raspberry Pi
protocol inside the calibrated container, so YOLO26n can be compared format by format with
the real Pi 5 (does the simulation predict real-device speed?). Deployed choice:
**YOLO26n, ONNX** (most accurate model, fastest format in the container).

## Step 6 — alert evaluation (ground truth, ablation, threshold sensitivity)

```powershell
python scripts/label_sheets.py --step 5          # contact sheets used for labelling
python scripts/cache_tracks.py                   # detect + track all 16 videos once at 3 fps
python scripts/tune_grace.py                     # tune grace_s on the DEV videos only
python scripts/eval_alerts.py                    # 5 variants x T = 10/30/60 s, dev and test
python scripts/make_figures.py; python scripts/make_results.py
```

Ground truth (`annotations/islab_events.csv`): when each vehicle stopped in / left a zone,
labelled from raw frames (provenance in the file header). Dev videos 01, 04, 07, 08, 15 were
used for development and tuning; all other videos are the test set. Variants: A raw per-frame
alerts, B tracking only, C dwell without tracking, D tracking + dwell without duplicate merging,
E full system.

## Live demo

Double-click `run_demo.bat` (starts Docker, all containers and the dashboard), `restart_demo.bat`
(replay), `show_results.bat` (evaluation table) and `stop_demo.bat`. Step-by-step script for the
class demonstration: [docs/DEMO.md](docs/DEMO.md). The IEEE report source is in [report/](report/).

## Privacy

Alert snapshots are filtered on the edge node before they are sent: people are detected and
blurred (`privacy.blur_persons`). For material shared publicly (report, slides, demo video),
`python scripts/redact_media.py <files> --vehicles` also pixelates vehicles so number plates
cannot be read.

## Reproduce everything

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e .
python scripts/export_models.py
python scripts/eval_detrac.py --model yolov8n.pt; python scripts/eval_detrac.py --model yolo11n.pt; python scripts/eval_detrac.py --model yolo26n.pt
docker build -t edgepark-node .
python scripts/calibrate_cpus.py --cpus 0.5 0.6 0.75 1.0 --repeats 3
python scripts/bench_pi_sim.py
python scripts/cache_tracks.py; python scripts/tune_grace.py; python scripts/eval_alerts.py
python scripts/make_figures.py; python scripts/make_results.py
```

## AI assistance (disclosure)

This project was built with substantial help from an AI coding assistant (Claude, Anthropic),
working step by step under the team's direction. For transparency:

**Done by the AI assistant**
* Wrote most of the code in this repository: `src/edgepark/` (detector wrapper, tracking/dwell
  logic, zones, MQTT alerts, privacy filter, dashboard), all scripts in `scripts/`, the
  Dockerfile, `docker-compose.yml`, MediaMTX / Mosquitto configs and `tools/fix-docker.ps1`.
* Drafted the zone rules (`docs/zone_policy.md`, approved by the team) and drew the zone
  polygons for all 16 videos from those rules.
* Produced the ground-truth event labels (`annotations/islab_events.csv`) by inspecting video
  frames, at the team's request (provenance note in the file header).
* Ran the experiments (calibration, benchmarks, DETRAC evaluation, alert evaluation), diagnosed
  failures, and generated the figures and `docs/RESULTS.md` from the result files.

**Done by the team**
* Project idea, proposal and research questions; choice of the simulation-only approach.
* Dataset selection and download (ISLab PVD, UA-DETRAC); hardware/Docker setup on the laptop.
* All design decisions taken during the work (zone rules approval, counting the parked
  motorbike as a violation, keeping the roundabout scene, 60 s threshold, dev/test split).
* Reviewing, testing, committing and presenting the work.

All numeric results come from running the code in this repository on the datasets above;
none were written by hand.

## Project layout

```
configs/          runtime configuration, ByteTrack settings, zones/ (one file per video)
src/edgepark/     library: detector, video (incl. live latest-frame reader), zones, dwell logic,
                  alerts (MQTT), privacy, dashboard (FastAPI + static page)
scripts/          entry points (pipeline, dashboard, benchmarks, evaluation, figures)
docker/           Mosquitto and MediaMTX (camera) configuration
annotations/      ground truth for the ISLab videos
docs/             zone policy, RESULTS.md, DEMO.md (live demo script)
report/           IEEE conference report (LaTeX source, figures, references)
tools/            fix-docker.ps1
models/           weights and exports (git-ignored)
data/             videos, DETRAC sample, datasets (git-ignored)
outputs/          generated files; bench/, eval/ and figures/ are versioned, the rest is ignored
```
