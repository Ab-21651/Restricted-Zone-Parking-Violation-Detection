"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to configs/default.yaml)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # YAML keys may come back as ints already; normalise to int -> str
    cfg["vehicle_classes"] = {int(k): str(v) for k, v in cfg.get("vehicle_classes", {}).items()}
    return cfg
