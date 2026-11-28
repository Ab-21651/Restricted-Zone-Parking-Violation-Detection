# Live demonstration — step by step

The demo shows the whole system running: a simulated CCTV camera streams a video, a
Raspberry-Pi-5-sized edge node detects and tracks vehicles, raises a violation after
60 s of parking in the restricted zone, sends it over MQTT, and the dashboard shows it
with a snapshot (people blurred). Total time: about 5–6 minutes.

## One-click version

Double-click these files in the project folder:

| File | What it does |
|---|---|
| `run_demo.bat` | starts Docker Desktop if needed (and repairs it if it hangs), starts all four containers, opens the dashboard, then shows the edge node's live log |
| `restart_demo.bat` | replays the video from the start with a fresh edge node (old alerts disappear from the dashboard) |
| `show_results.bat` | re-runs the alert evaluation and prints the results table, opens `docs/RESULTS.md` |
| `stop_demo.bat` | stops everything |

The sections below explain the same steps command by command.

## Before class (do this at home the day before, and again 10 min before)

1. **Plug in the charger** and set Windows power mode to **Best performance**
   (Settings → System → Power & battery). On battery the node runs ~2x slower
   (about 1.9 fps instead of ~4 fps); alerts still work, the numbers just look worse.
2. Start **Docker Desktop** and wait for *Engine running*.
   If it shows *"An unexpected error occurred ... dockerInference"*: click **Quit**, then
   in PowerShell in the project folder run `.\tools\fix-docker.ps1` and wait a minute.
3. Do a full test run (steps 1–4 below) once, then `docker compose down`.
4. Close other heavy programs (games, many browser tabs, Teams video).
5. No internet is needed during the demo: images, model (`models/yolo26n.onnx`)
   and videos are all on the laptop.

## The demo

Open PowerShell in the project folder:

```powershell
cd D:\7th-Semester\edge\project\Restricted-Zone-Parking-Violation-Detection
```

### 1. Start the system (say: "four containers, like four devices")

```powershell
docker compose up -d
```

* `camera` — MediaMTX loops video ISLab-07 as a CCTV stream (`rtsp://camera:8554/cam`)
* `edge-node` — YOLO26n (ONNX) + ByteTrack + dwell timer, **limited to 0.57 CPU and 4 GB**
  = the speed of a Raspberry Pi 5 (calibrated: 302 ms vs published 299 ms)
* `mosquitto` — MQTT broker
* `dashboard` — monitoring web page

### 2. Open the dashboard

Open **http://localhost:8000** in the browser. Show:
* *Edge nodes* card: `edge-01 online`, processing fps, ms per frame, vehicles in view, zone.
* It is empty of violations at first — nothing is reported just because a car appears.

### 3. Show it really is Pi-sized (while waiting for the first alert)

```powershell
docker stats --no-stream
```

`edgepark-edge` stays near its 0.57-CPU limit and uses ~170 MB of its 4 GB.

Optional — show the raw MQTT heartbeat the node sends every 5 s:

```powershell
docker exec edgepark-mosquitto mosquitto_sub -t "edgepark/+/status" -v -C 1
```

Optional — follow the node's log:

```powershell
docker compose logs -f edge-node
```

(press Ctrl+C to stop following; the system keeps running)

### 4. The alerts (about 60–80 s after start)

* **~65 s**: *motorcycle in fire-lane* — the motorbike parked by the wall since the start.
* **~70–95 s**: *car in fire-lane* — the SUV that drives in and parks.
* Each card: snapshot with the box, zone, "stopped 60 s at alert", capture → dashboard
  latency. Point out: cars that only drive through the lane get **no** alert — that is
  what the tracking + dwell rule fixes (frame-wise ROI gave 1,776 alerts on the test
  videos, our system 5, all correct).
* When the video loops (~130 s) the vehicles "leave" and a **cleared** event closes the case.

### 5. Show the experiments (results are reproducible)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/eval_alerts.py
```

Runs in a few seconds from the cached tracks and prints the full table (base rule vs.
variants A–E, thresholds 10/30/60 s, dev and test videos). Then open
`docs/RESULTS.md` and the figures in `outputs/figures/`.

### 6. Restart from the beginning / another video

```powershell
docker compose restart camera edge-node          # replay ISLab-07 from the start
$env:CAM="ISLab-04"; docker compose up -d        # another scene (SUV parks at ~68 s, alert ~2 min)
```

### 7. Stop

```powershell
docker compose down
```

## Backup if something fails live

* `outputs/pipeline_07.mp4` — annotated recording of the same scene (boxes, zone, dwell
  timers, violation). Play it and explain the same steps.
* Record your own screen during the test run at home (Win + Alt + R in the Xbox Game Bar)
  so you have a video of the dashboard receiving the alerts.

## If something goes wrong

| Problem | Fix |
|---|---|
| Docker "unexpected error ... dockerInference" | Quit, `.\tools\fix-docker.ps1`, wait 1 min |
| Dashboard does not open | `docker compose ps` — all four must be *Up*; `docker compose up -d` again |
| No alert after 2 minutes | `docker compose logs --tail 20 edge-node`; restart with `docker compose restart camera edge-node` |
| Very slow (< 1 fps) | charger + Best performance; close other programs |
