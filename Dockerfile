# syntax=docker/dockerfile:1
# Edge node image: detector + tracker + dwell logic + MQTT publisher (and the dashboard).
# CPU-only PyTorch, as on a Raspberry Pi. Resource limits are applied at run time
# (docker compose / docker run --cpus --memory), not here.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_AUTOINSTALL=false \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

# OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy dependencies first so code changes do not re-download them.
# The pip cache mount survives failed builds on a flaky connection.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# Ultralytics: keep datasets / weights / runs inside the mounted project folders, no telemetry
RUN yolo settings datasets_dir=/app/data/datasets weights_dir=/app/models runs_dir=/app/outputs/runs sync=False

COPY configs ./configs
COPY scripts ./scripts

# A Raspberry Pi 5 has 4 cores, so inference uses 4 threads. Without this PyTorch
# picks 2 here (WSL2 exposes the 4 vCPUs as 2 cores x 2 hyper-threads).
# Applied by edgepark.detector / scripts/benchmark.py via torch.set_num_threads().
ENV EDGEPARK_THREADS=4

CMD ["python", "scripts/run_pipeline.py", "--help"]
