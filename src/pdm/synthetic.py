"""Synthetic raw data with the same schema and the main EDA characteristics as the real delivery.

**For development and CI only.** The real files are private and never committed (see
``.gitignore``). This generator lets the full pipeline (Stage 2 → Stage 3) run end to end on a
fresh clone or in GitHub Actions, so that code changes can be smoke-tested without the data lake.
Model metrics obtained on synthetic data say nothing about the real HPUs and must never be quoted
as results.

What is reproduced (with the EDA section that documents it on the real data):

* 10 HPUs × 60 days of 1-minute telemetry, 9 failure events on 9 machines, HPU_10 healthy (§6.2)
* Day / Night / Weekend set points, the midnight dip, and a fleet-wide standby day on 15 Jan (§5, §7)
* machine-specific pump-rpm set points (the identity leak, §9.2)
* vibration Y duplicating X most of the time, and floor-clipped at 0.05 g (§4.6, DQ-10)
* a slow fleet-wide oil-temperature drift (DQ-11)
* single-minute glitches (pressure > 350 bar, negative temperature) and short sensor dropouts (§4.1–4.3)
* one sensor fingerprint per failure mode, ramping up through the labelled degradation window (§8)
* the flawed supplied ``rul_hours`` column (500 h sentinel, zeros during repair, §6.3)

Usage::

    python -m pdm.synthetic --out data/raw          # writes the three CSVs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = "2024-01-01"
DAYS = 60
MACHINES = [f"HPU_{i:02d}" for i in range(1, 11)]
STANDBY_DAY = "2024-01-15"

# Mirrors the real failure_labels.csv layout (dates as in the delivered file)
FAILURES = pd.DataFrame({
    "failure_event_id": [f"F_{i:03d}" for i in range(1, 10)],
    "machine_id": MACHINES[:9],
    "failure_timestamp": pd.to_datetime(["2024-02-15", "2024-02-22", "2024-02-08", "2024-02-28", "2024-01-31",
                                         "2024-02-18", "2024-02-25", "2024-02-12", "2024-01-26"]),
    "failure_mode": ["pump_wear", "valve_leakage", "contamination", "pump_wear", "valve_leakage",
                     "cylinder_drift", "valve_leakage", "pump_wear", "contamination"],
    "degradation_start_timestamp": pd.to_datetime(["2024-02-01", "2024-02-12", "2024-02-03", "2024-02-16", "2024-01-21",
                                                   "2024-02-10", "2024-02-15", "2024-01-29", "2024-01-21"]),
    "repair_cost_usd": [25094, 8711, 24507, 29482, 14065, 23021, 11598, 30336, 16482],
    "downtime_hours": [13.2, 9.8, 6.3, 22.4, 8.2, 13.1, 8.7, 14.5, 10.5],
})


def _severity(ts: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, power: float) -> np.ndarray:
    """0 before ``start``, rising as ((t-start)/(end-start))**power to 1 at ``end``, 0 after."""
    frac = ((ts - start) / (end - start)).to_numpy(dtype=float)
    s = np.clip(frac, 0, 1) ** power
    s[(ts < start) | (ts >= end)] = 0.0
    return s


def make_raw(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=DAYS * 1440, freq="1min")
    n = len(ts)
    hour, dow = ts.hour.to_numpy(), ts.dayofweek.to_numpy()
    weekend = dow >= 5
    day = (~weekend) & (hour >= 8) & (hour < 20)
    shift = np.where(weekend, "Weekend", np.where(day, "Day", "Night"))
    level = np.where(weekend, 0.0, np.where(day, 2.0, 1.0)) - 0.6 * (hour == 0)       # midnight dip
    standby = (ts.normalize() == pd.Timestamp(STANDBY_DAY))
    drift = np.linspace(0, 0.8, n)                                                     # DQ-11, °C

    frames = []
    for k, mid in enumerate(MACHINES):
        p0, t0, f0 = rng.normal(112, 4), rng.normal(50.5, 1.0), rng.normal(76, 3)
        rpm0 = 1440 + 9 * k                                                             # identity leak
        # slow AR(1) wander so that healthy hours are not i.i.d. noise
        wander = np.cumsum(rng.normal(0, 0.02, n)); wander -= pd.Series(wander).rolling(720, min_periods=1).mean().to_numpy()
        pressure = p0 + 13 * level + 1.5 * wander + rng.normal(0, 1.7, n)
        temp = t0 + 1.3 * level + drift + 0.3 * wander + rng.normal(0, 0.5, n)
        flow = f0 + 10 * level + 1.0 * wander + rng.normal(0, 1.6, n)
        vib = rng.gamma(2.0, 0.06, n)
        rpm = rpm0 + 30 * level + rng.integers(-3, 4, n)
        xy = np.zeros(n)
        # ~2 % of minutes: small benign X−Y differences (Y is otherwise a copy of X)
        blip = rng.random(n) < 0.02
        xy[blip] = rng.normal(0, 0.006, blip.sum())

        state = np.array(["healthy"] * n, dtype=object)
        rul = np.full(n, 500.0)
        ev = FAILURES[FAILURES["machine_id"] == mid]
        for _, e in ev.iterrows():
            a = rng.uniform(0.8, 1.25)                                                  # event-to-event amplitude
            s = _severity(ts, e.degradation_start_timestamp, e.failure_timestamp, power=rng.uniform(1.3, 2.0))
            m = e.failure_mode
            if m == "pump_wear":
                pressure -= a * 26 * s; flow -= a * 7 * s; temp += a * 1.2 * s; vib *= 1 + 0.4 * a * s
            elif m == "valve_leakage":
                temp += a * 5.0 * s; flow -= a * 14 * s; pressure -= a * 5 * s
            elif m == "contamination":
                pressure += a * 9 * s; flow -= a * 4 * s
                burst = rng.random(n) < 0.08 * s
                vib[burst] += rng.uniform(1.0, 2.5, burst.sum())
                late = s > 0.55
                xy[late] += rng.normal(0.03, 0.015, late.sum()) * a
            elif m == "cylinder_drift":
                pressure -= a * 12 * s; flow -= a * 3 * s
                xy += a * 0.06 * s + rng.normal(0, 0.01, n) * (s > 0)
            win = (ts >= e.degradation_start_timestamp) & (ts < e.failure_timestamp)
            state[win] = m
            hrs = ((e.failure_timestamp - ts) / pd.Timedelta("1h")).to_numpy()
            rul = np.where(win, np.minimum(hrs, 336.0), rul)
            down = (ts >= e.failure_timestamp) & (ts < e.failure_timestamp + pd.Timedelta(hours=e.downtime_hours))
            state[down] = "post_failure"; rul[down] = 0.0
            after = ts >= e.failure_timestamp + pd.Timedelta(hours=e.downtime_hours)
            rul[after & (rul == 500.0)] = 0.0                                            # the supplied zeros after repair (§6.3)
            rul[ts >= e.failure_timestamp + pd.Timedelta(days=14)] = 500.0

        # fleet-wide standby day: pump runs slowly, everything drops (§4.4)
        pressure[standby] = rng.normal(50, 2, standby.sum()); flow[standby] = rng.normal(35, 1.5, standby.sum())
        temp[standby] = rng.normal(44, 0.5, standby.sum()); rpm[standby] = 1000

        vib = np.maximum(0.05, vib)                                                     # floor clip (DQ-10)
        vib_y = np.maximum(0.05, vib - xy)
        # glitches and dropouts (§4.1–4.3)
        gp = rng.choice(n, 25, replace=False); pressure[gp] = rng.uniform(360, 420, 25)
        gt = rng.choice(n, 17, replace=False); temp[gt] = rng.uniform(-40, -5, 17)
        dropout = np.zeros(n, dtype=int)
        for _ in range(12):
            s0 = rng.integers(0, n - 30); L = int(rng.integers(2, 25))
            dropout[s0:s0 + L] = 1
        df = pd.DataFrame({
            "timestamp": ts, "machine_id": mid,
            "pressure_bar": pressure.round(2), "temp_celsius": temp.round(2), "flow_lpm": flow.round(2),
            "vibration_x_g": vib.round(3), "vibration_y_g": vib_y.round(3), "pump_rpm": rpm.astype(float),
            "is_anomaly": np.isin(state, ["healthy"], invert=True).astype(int),
            "failure_mode": np.where(np.isin(state, ["healthy", "post_failure"]), None, state),
            "rul_hours": rul.round(1), "is_sensor_dropout": dropout, "shift": shift, "day_of_week": dow,
        })
        df.loc[dropout == 1, ["pressure_bar", "temp_celsius", "flow_lpm", "vibration_x_g", "vibration_y_g", "pump_rpm"]] = np.nan
        frames.append(df)
    telemetry = pd.concat(frames, ignore_index=True)

    n_m = 51
    maintenance = pd.DataFrame({
        "maintenance_id": [f"M_{i:03d}" for i in range(1, n_m + 1)],
        "machine_id": rng.choice(MACHINES, n_m),
        "action_timestamp": pd.Timestamp("2023-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 364 * 24, n_m)), unit="h"),
        "action_type": rng.choice(["Preventive", "Reactive", "Predictive", "Inspection"], n_m, p=[0.45, 0.25, 0.1, 0.2]),
        "component_replaced": rng.choice(["Pump", "Valve", "Filter", "Seal", "Oil", "None"], n_m),
        "technician_id": rng.choice([f"T_{i}" for i in range(1, 7)], n_m),
        "cost_usd": rng.integers(300, 9000, n_m).astype(float),
    })
    return {"telemetry": telemetry, "failures": FAILURES.copy(), "maintenance": maintenance}


FILES = ("sensor_telemetry.csv", "failure_labels.csv", "maintenance_log.csv")


def write_raw(out_dir: str | Path, seed: int = 7, force: bool = False) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    existing = [f for f in FILES if (out / f).exists()]
    if existing and not force:     # never silently overwrite the real delivery
        raise FileExistsError(f"{existing} already exist in {out}; use --force to overwrite them with synthetic data")
    t = make_raw(seed)
    t["telemetry"].to_csv(out / "sensor_telemetry.csv", index=False)
    t["failures"].to_csv(out / "failure_labels.csv", index=False)
    t["maintenance"].to_csv(out / "maintenance_log.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write SYNTHETIC raw files (dev/CI only, not real data)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="overwrite existing raw files")
    a = ap.parse_args(argv)
    write_raw(a.out, a.seed, a.force)
    print(f"Synthetic raw files written to {a.out} (development data, not the real HPU delivery)")


if __name__ == "__main__":
    main()
