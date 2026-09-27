"""Configuration loading.

All thresholds live in ``configs/config.yaml``. Paths in the file are relative to the
project root and are resolved to absolute ``Path`` objects here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every entry under ``paths`` against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    root = Path(root) if root else (path.resolve().parents[1] if path.exists() else PROJECT_ROOT)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["root"] = root
    cfg["paths"] = {k: (root / v) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg
