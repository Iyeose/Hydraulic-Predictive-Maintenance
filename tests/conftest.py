"""Small synthetic fixtures. Tests never need the 74 MB real file, so they run in seconds in CI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.config import load_config


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    c = load_config()          # real config values, …
    for k in c["paths"]:       # … but every output goes to a temp folder
        c["paths"][k] = root / k
        c["paths"][k].mkdir(parents=True, exist_ok=True)
    return c


def make_telemetry(machines=("HPU_01", "HPU_02"), start="2024-01-01", days=20, seed=0) -> pd.DataFrame:
    """Healthy-looking 1-minute telemetry with realistic regime set points."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 1440, freq="1min")
    hour, dow = ts.hour, ts.dayofweek
    weekend = dow >= 5
    day = (~weekend) & (hour >= 8) & (hour < 20)
    level = np.where(weekend, 0.0, np.where(day, 2.0, 1.0))
    frames = []
    for k, m in enumerate(machines):
        n = len(ts)
        vib = np.maximum(0.05, rng.gamma(2.0, 0.05, n))
        frames.append(pd.DataFrame({
            "timestamp": ts, "machine_id": m,
            "pressure_bar": 112 + 13 * level + rng.normal(0, 1.7, n),
            "temp_celsius": 50.4 + 1.3 * level + rng.normal(0, 0.5, n),
            "flow_lpm": 76 + 10 * level + rng.normal(0, 1.6, n),
            "vibration_x_g": vib, "vibration_y_g": vib,
            "pump_rpm": 1458 + 32 * level + k + rng.integers(-3, 4, n),
            "is_anomaly": 0, "failure_mode": pd.NA, "rul_hours": 500.0, "is_sensor_dropout": 0,
            "shift": np.where(weekend, "Weekend", np.where(day, "Day", "Night")),
            "day_of_week": dow,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["failure_mode"] = df["failure_mode"].astype("string")
    return df


def make_failures() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": ["F_001"], "machine_id": ["HPU_01"],
        "failure_timestamp": pd.to_datetime(["2024-01-18"]),
        "degradation_start_timestamp": pd.to_datetime(["2024-01-16"]),
        "failure_mode": ["pump_wear"], "repair_cost_usd": [20000.0], "downtime_hours": [5.5],
    })


def make_maintenance() -> pd.DataFrame:
    return pd.DataFrame({
        "action_id": ["M_001", "M_002"], "machine_id": ["HPU_01", "HPU_02"],
        "action_timestamp": pd.to_datetime(["2023-12-01 10:00", "2023-12-05 08:00"]),
        "action_type": ["Reactive", "Preventive"], "component_replaced": ["Pump", "None"],
        "technician_id": ["T_1", "T_2"], "cost_usd": [5000.0, 800.0],
    })


@pytest.fixture
def telemetry():
    return make_telemetry()


@pytest.fixture
def failures():
    return make_failures()


@pytest.fixture
def maintenance():
    return make_maintenance()
