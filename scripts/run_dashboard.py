"""Start the monitoring dashboard (needs the MQTT broker: docker compose up -d mosquitto).

    python scripts/run_dashboard.py               # http://127.0.0.1:8000
    python scripts/run_dashboard.py --port 8080
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from edgepark.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    if args.config:
        os.environ["EDGEPARK_CONFIG"] = args.config
    dcfg = load_config(args.config).get("dashboard", {})
    uvicorn.run(
        "edgepark.dashboard.app:app",
        host=args.host or dcfg.get("host", "127.0.0.1"),
        port=args.port or dcfg.get("port", 8000),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
