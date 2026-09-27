"""Ingestion: read the raw source files with explicit types and map them to the data contract.

In production the same functions read Parquet from the data lake (e.g. S3). Here they read the
delivered CSVs. Nothing is cleaned at this step: raw values are preserved exactly, so every later
change is traceable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TELEMETRY_DTYPES = {
    "machine_id": "string", "pressure_bar": "float64", "temp_celsius": "float64", "flow_lpm": "float64",
    "vibration_x_g": "float64", "vibration_y_g": "float64", "pump_rpm": "float64", "is_anomaly": "Int8",
    "failure_mode": "string", "rul_hours": "float64", "is_sensor_dropout": "Int8",
    "shift": "string", "day_of_week": "Int8",
}


def _find(raw_dir: Path, name: str) -> Path:
    """Look in data/raw first, then data/ (the Stage 1 layout), so both folder layouts work."""
    for candidate in (raw_dir / name, raw_dir.parent / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found in {raw_dir} or {raw_dir.parent}")


def read_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=TELEMETRY_DTYPES, parse_dates=["timestamp"])
    return df.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)


def read_failures(path: Path, renames: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["failure_timestamp", "degradation_start_timestamp"],
                     dtype={"failure_event_id": "string", "machine_id": "string", "failure_mode": "string"})
    return df.rename(columns=renames or {})


def read_maintenance(path: Path, renames: dict[str, str] | None = None) -> pd.DataFrame:
    # keep_default_na=False: the literal string "None" in component_replaced means "no component",
    # which pandas would otherwise parse as missing (EDA §10).
    df = pd.read_csv(path, parse_dates=["action_timestamp"], keep_default_na=False, na_values=[""],
                     dtype={"maintenance_id": "string", "machine_id": "string", "action_type": "string",
                            "component_replaced": "string", "technician_id": "string"})
    return df.rename(columns=renames or {}).sort_values("action_timestamp").reset_index(drop=True)


def ingest(cfg: dict, write: bool = True) -> dict[str, pd.DataFrame]:
    """Read all three sources, apply contract renames, and optionally write them as Parquet ("bronze")."""
    raw_dir = cfg["paths"]["raw_dir"]
    renames = cfg.get("contract_renames", {})
    tables = {
        "telemetry": read_telemetry(_find(raw_dir, cfg["files"]["telemetry"])),
        "failures": read_failures(_find(raw_dir, cfg["files"]["failures"]), renames.get("failures")),
        "maintenance": read_maintenance(_find(raw_dir, cfg["files"]["maintenance"]), renames.get("maintenance")),
    }
    if write:
        for name, df in tables.items():
            df.to_parquet(cfg["paths"]["interim_dir"] / f"bronze_{name}.parquet", index=False)
    return tables
