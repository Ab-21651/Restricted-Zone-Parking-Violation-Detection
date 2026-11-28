"""Monitoring dashboard.

Subscribes to ``<prefix>/+/events`` and ``<prefix>/+/status`` on the MQTT broker,
keeps the current state in memory and pushes updates to browsers with
Server-Sent Events (``/api/stream``). Every received event is also appended to
``outputs/dashboard/received.jsonl`` together with its measured
capture-to-dashboard latency, for evaluation.

    python scripts/run_dashboard.py            # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from ..config import PROJECT_ROOT, load_config

STATIC = Path(__file__).parent / "static"
LOG_PATH = Path(os.environ.get("DASHBOARD_LOG", PROJECT_ROOT / "outputs" / "dashboard" / "received.jsonl"))


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.nodes: dict[str, dict] = {}
        self.active: dict[str, dict] = {}  # key -> violation event
        self.history: deque[dict] = deque(maxlen=300)
        self.snapshots: dict[str, bytes] = {}
        self.latencies_ms: deque[float] = deque(maxlen=500)
        self.broker_connected = False

    @staticmethod
    def key(ev: dict) -> str:
        return f"{ev.get('node')}:{ev.get('source')}:{ev.get('event_id')}"

    def on_event(self, ev: dict, received_ts: float) -> dict:
        lat = None
        if ev.get("capture_ts"):
            lat = round((received_ts - ev["capture_ts"]) * 1000.0, 1)
        snap = ev.pop("snapshot_jpg", None)
        ev["received_ts"] = received_ts
        ev["latency_ms"] = lat
        k = self.key(ev)
        ev["key"] = k
        with self.lock:
            if snap:
                self.snapshots[k] = base64.b64decode(snap)
                ev["has_snapshot"] = True
                if len(self.snapshots) > 200:
                    self.snapshots.pop(next(iter(self.snapshots)))
            if ev["event"] == "violation":
                self.active[k] = ev
                if lat is not None:
                    self.latencies_ms.append(lat)
            elif ev["event"] == "cleared":
                prev = self.active.pop(k, None)
                if prev:
                    ev["has_snapshot"] = prev.get("has_snapshot", False)
            self.history.appendleft(ev)
        return ev

    def on_status(self, st: dict, received_ts: float) -> dict:
        st["received_ts"] = received_ts
        with self.lock:
            self.nodes[st.get("node", "?")] = st
        return st

    def snapshot(self) -> dict:
        with self.lock:
            lat = list(self.latencies_ms)
            return {
                "broker_connected": self.broker_connected,
                "nodes": self.nodes,
                "active": list(self.active.values()),
                "history": list(self.history)[:100],
                "latency": {
                    "n": len(lat),
                    "mean_ms": round(sum(lat) / len(lat), 1) if lat else None,
                    "max_ms": max(lat) if lat else None,
                },
                "server_ts": time.time(),
            }


state = DashboardState()
subscribers: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _fanout(msg: dict) -> None:
    for q in list(subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # slow browser: drop, it resyncs from /api/state


def _broadcast(msg: dict) -> None:
    if _loop is not None:
        _loop.call_soon_threadsafe(_fanout, msg)


def _make_mqtt_client(cfg: dict) -> mqtt.Client:
    qcfg = cfg.get("mqtt", {})
    prefix = qcfg.get("topic_prefix", "edgepark")
    host = os.environ.get("MQTT_HOST", qcfg.get("host", "localhost"))
    port = int(os.environ.get("MQTT_PORT", qcfg.get("port", 1883)))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8")

    def on_connect(client, userdata, flags, reason_code, properties):
        state.broker_connected = not reason_code.is_failure
        client.subscribe([(f"{prefix}/+/events", 1), (f"{prefix}/+/status", 0)])
        print(f"[dashboard] MQTT connected {host}:{port}, subscribed to {prefix}/+/events|status")
        _broadcast({"type": "broker", "connected": state.broker_connected})

    def on_disconnect(client, userdata, flags, reason_code, properties):
        state.broker_connected = False
        _broadcast({"type": "broker", "connected": False})

    def on_message(client, userdata, msg):
        received = time.time()
        try:
            data = json.loads(msg.payload)
        except ValueError:
            return
        if msg.topic.endswith("/events"):
            ev = state.on_event(data, received)
            log.write(json.dumps({k: v for k, v in ev.items() if k != "snapshot_jpg"}) + "\n")
            log.flush()
            _broadcast({"type": "event", "data": ev})
        elif msg.topic.endswith("/status"):
            _broadcast({"type": "status", "data": state.on_status(data, received)})

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"dashboard-{os.getpid()}")
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect_async(host, port, keepalive=30)
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    client = _make_mqtt_client(load_config(os.environ.get("EDGEPARK_CONFIG")))
    client.loop_start()
    yield
    client.loop_stop()
    client.disconnect()


app = FastAPI(title="EdgePark dashboard", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(state.snapshot())


@app.get("/api/snapshot/{key}")
def api_snapshot(key: str) -> Response:
    data = state.snapshots.get(key)
    if data is None:
        raise HTTPException(404, "no snapshot")
    return Response(data, media_type="image/jpeg")


@app.get("/api/stream")
async def api_stream(request: Request) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    subscribers.add(q)

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
