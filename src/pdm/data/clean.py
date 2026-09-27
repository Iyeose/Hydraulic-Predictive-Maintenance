"""Cleaning ("silver" layer): turn validated telemetry into a trustworthy minute-level series.

Rules, all driven by ``config.yaml`` and traced back to the EDA:

1. Reindex every machine onto a complete 1-minute grid, so missing minutes become explicit rows.
2. Physically impossible values are set to NaN, never clipped (DQ-02/03).
3. Short gaps are forward-filled for at most ``max_interpolation_gap_min`` minutes, and every
   filled value is flagged (DQ-01). Forward fill is used rather than linear interpolation
   because it is **causal**: linear interpolation would use a reading that is up to 15 min in
   the future, which a real-time inference service would not have.
4. Every minute is tagged with an operating regime (Day / Night / Weekend / Standby) and a
   midnight-dip flag (DQ-04, EDA §7).
5. Label columns (``is_anomaly``, ``failure_mode``, ``rul_hours``) are removed. Targets are
   rebuilt from the failure table in ``features/targets.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reindex_to_grid(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Give every machine a complete, regular time index. New rows are flagged ``was_missing_row``."""
    parts = []
    for mid, g in df.groupby("machine_id", sort=True):
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq, name="timestamp")
        g = g.set_index("timestamp").reindex(full)
        g["was_missing_row"] = g["machine_id"].isna()
        g["machine_id"] = mid
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def invalidate_out_of_range(df: pd.DataFrame, ranges: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.Series]:
    """Set impossible values to NaN. Returns the frame and the per-row count of invalidated sensors."""
    df = df.copy()
    n_invalid = pd.Series(0, index=df.index, dtype="int8")
    for col, (lo, hi) in ranges.items():
        bad = (df[col] < lo) | (df[col] > hi)
        df.loc[bad, col] = np.nan
        n_invalid += bad.astype("int8")
    return df, n_invalid


def fill_short_gaps(df: pd.DataFrame, sensors: list[str], limit: int) -> tuple[pd.DataFrame, pd.Series]:
    """Causal forward-fill, per machine, for at most ``limit`` consecutive minutes."""
    df = df.copy()
    was_nan = df[sensors].isna()
    df[sensors] = df.groupby("machine_id", sort=False)[sensors].ffill(limit=limit)
    imputed = (was_nan & df[sensors].notna()).any(axis=1)
    return df, imputed


def tag_regimes(df: pd.DataFrame, cfg_regimes: dict) -> pd.DataFrame:
    """Add ``day_type`` (Weekday/Weekend), ``regime`` (Day/Night/Weekend/Standby) and ``is_midnight_dip``."""
    df = df.copy()
    ts = df["timestamp"]
    hour = ts.dt.hour
    weekend = ts.dt.dayofweek.isin(cfg_regimes["weekend_days"])
    h0, h1 = cfg_regimes["day_shift_hours"]
    standby = df["pump_rpm"] < cfg_regimes["standby_rpm_below"]      # NaN rpm → not standby
    df["hour"] = hour.astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["day_type"] = np.where(weekend, "Weekend", "Weekday")
    df["is_standby"] = standby.fillna(False).astype(bool)
    df["regime"] = np.select([df["is_standby"], weekend, hour.between(h0, h1 - 1)],
                             ["Standby", "Weekend", "Day"], default="Night")
    df["is_midnight_dip"] = hour.isin(cfg_regimes["midnight_dip_hours"])
    return df


def clean_telemetry(tel: pd.DataFrame, cfg: dict, write: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run all cleaning rules. Returns the silver frame and a log of what each rule changed."""
    sensors = cfg["sensors"]
    c = cfg["cleaning"]
    log = {"rows_in": len(tel)}

    # Keep the source shift label only to cross-check our own regime tagging, then drop it
    source_shift = tel.set_index(["machine_id", "timestamp"])["shift"] if "shift" in tel else None
    df = tel.drop(columns=[x for x in cfg["leakage_columns"] + ["shift"] if x in tel.columns])

    df = reindex_to_grid(df, cfg["validation"]["sampling_interval"])
    log["rows_added_by_reindex"] = int(df["was_missing_row"].sum())
    log["missing_values_before"] = int(df[sensors].isna().sum().sum())

    df, n_invalid = invalidate_out_of_range(df, c["valid_ranges"])
    df["n_invalid_sensors"] = n_invalid
    log["values_invalidated"] = {s: int(v) for s, v in
                                 ((s, ((tel[s] < lo) | (tel[s] > hi)).sum()) for s, (lo, hi) in c["valid_ranges"].items())}

    df, imputed = fill_short_gaps(df, sensors, c["max_interpolation_gap_min"])
    df["was_imputed"] = imputed
    df["is_sensor_dropout"] = df["is_sensor_dropout"].fillna(1).astype("int8")
    log["rows_imputed"] = int(imputed.sum())
    log["missing_values_after"] = int(df[sensors].isna().sum().sum())

    df = tag_regimes(df, cfg["regimes"])
    log["regime_minutes"] = df["regime"].value_counts().to_dict()
    log["standby_days"] = sorted({str(d.date()) for d in df.loc[df["is_standby"], "timestamp"].dt.normalize().unique()})
    if source_shift is not None:
        chk = df.set_index(["machine_id", "timestamp"])[["regime"]].join(source_shift.rename("shift"), how="inner")
        chk = chk[chk["regime"] != "Standby"]
        log["regime_vs_source_shift_agreement"] = float((chk["regime"] == chk["shift"]).mean())

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    if write:
        df.to_parquet(cfg["paths"]["interim_dir"] / "silver_telemetry.parquet", index=False)
    log["rows_out"] = len(df)
    return df, log
