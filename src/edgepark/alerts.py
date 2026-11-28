"""MQTT alert publishing from the edge node.

Topics (``<prefix>/<node_id>/...``):
  * ``events``  - violation / cleared events, QoS 1 (delivered at least once).
                  Violation events carry a small JPEG snapshot (base64).
  * ``status``  - heartbeat every few seconds (FPS, latency, active violations),
                  retained, so a dashboard that connects later sees the last state.
                  A Last-Will message marks the node ``offline`` if it drops.

Publishing never blocks the detection loop: paho runs its network loop in a
background thread and buffers messages while the broker is unreachable, so the
edge node keeps working (and keeps its local JSONL log) during network outages.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt


class AlertPublisher:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        node_id: str = "edge-01",
        prefix: str = "edgepark",
        heartbeat_s: float = 5.0,
        snapshot_width: int = 480,
    ) -> None:
        self.host = os.environ.get("MQTT_HOST", host)
        self.port = int(os.environ.get("MQTT_PORT", port))
        self.node_id = node_id
        # Event ids restart at 1 when the node restarts; the run id (start time)
        # lets the dashboard tell a new run's events from the previous run's.
        self.run_id = int(time.time())
        self.events_topic = f"{prefix}/{node_id}/events"
        self.status_topic = f"{prefix}/{node_id}/status"
        self.heartbeat_s = heartbeat_s
        self.snapshot_width = snapshot_width
        self._last_heartbeat = 0.0
        self.connected = False
        self.sent = 0

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{node_id}-{socket.gethostname()}-{os.getpid()}",
        )
        self.client.max_queued_messages_set(1000)  # buffer while offline
        self.client.will_set(self.status_topic, json.dumps({"node": node_id, "state": "offline"}), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "AlertPublisher":
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()
        return self

    def wait_connected(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while not self.connected and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.connected

    def close(self) -> None:
        self._closing = True
        self._publish(self.status_topic, {"node": self.node_id, "state": "stopped", "ts": time.time()}, qos=1, retain=True)
        time.sleep(0.2)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected = not reason_code.is_failure
        print(f"[mqtt] connected to {self.host}:{self.port} ({reason_code})")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected = False
        if not getattr(self, "_closing", False):
            print(f"[mqtt] disconnected ({reason_code}); will retry")

    # ----------------------------------------------------------------- publish
    def _publish(self, topic: str, payload: dict, qos: int = 1, retain: bool = False) -> None:
        self.client.publish(topic, json.dumps(payload), qos=qos, retain=retain)
        self.sent += 1

    def publish_event(self, event: dict, capture_ts: float, snapshot: np.ndarray | None = None) -> None:
        """``capture_ts`` = wall-clock time the triggering frame was captured,
        used by the dashboard to measure capture-to-alert latency."""
        payload = dict(event, node=self.node_id, run=self.run_id, capture_ts=capture_ts, publish_ts=time.time())
        if snapshot is not None:
            h, w = snapshot.shape[:2]
            if w > self.snapshot_width:
                snapshot = cv2.resize(snapshot, (self.snapshot_width, int(h * self.snapshot_width / w)))
            ok, buf = cv2.imencode(".jpg", snapshot, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                payload["snapshot_jpg"] = base64.b64encode(buf.tobytes()).decode("ascii")
        self._publish(self.events_topic, payload, qos=1)

    def heartbeat(self, stats: dict, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last_heartbeat >= self.heartbeat_s:
            self._last_heartbeat = now
            self._publish(self.status_topic, dict(stats, node=self.node_id, run=self.run_id, state="online", ts=now), qos=0, retain=True)
