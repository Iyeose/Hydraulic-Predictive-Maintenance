import numpy as np
import pandas as pd

from conftest import make_telemetry
from pdm.data.clean import clean_telemetry


def test_impossible_values_are_removed_then_filled_and_flagged(cfg):
    tel = make_telemetry(machines=("HPU_01",), days=1)
    tel.loc[100, "pressure_bar"] = 560.0
    tel.loc[200, "temp_celsius"] = -8.0
    out, log = clean_telemetry(tel, cfg, write=False)
    assert out.loc[100, "pressure_bar"] == tel.loc[99, "pressure_bar"]      # forward-filled from the previous minute
    assert out.loc[100, "was_imputed"] and out.loc[100, "n_invalid_sensors"] == 1
    assert out["pressure_bar"].max() < 350 and out["temp_celsius"].min() > 10
    assert log["values_invalidated"]["pressure_bar"] == 1


def test_only_the_first_n_minutes_of_a_long_gap_are_filled(cfg):
    tel = make_telemetry(machines=("HPU_01",), days=1)
    gap = range(500, 520)                                    # 20-minute dropout
    tel.loc[gap, cfg["sensors"]] = np.nan
    tel.loc[gap, "is_sensor_dropout"] = 1
    out, _ = clean_telemetry(tel, cfg, write=False)
    limit = cfg["cleaning"]["max_interpolation_gap_min"]
    assert out.loc[500:500 + limit - 1, "pressure_bar"].notna().all()
    assert out.loc[500 + limit:519, "pressure_bar"].isna().all()


def test_missing_timestamps_become_explicit_rows(cfg):
    tel = make_telemetry(machines=("HPU_01",), days=1).drop(index=[10, 11, 12])
    out, log = clean_telemetry(tel, cfg, write=False)
    assert len(out) == 1440 and log["rows_added_by_reindex"] == 3
    assert out.loc[10:12, "was_missing_row"].all()


def test_regime_tagging_matches_shift_boundaries_and_detects_standby(cfg):
    tel = make_telemetry(machines=("HPU_01",), days=7)               # 2024-01-01 is a Monday
    sat = tel["timestamp"] == pd.Timestamp("2024-01-06 12:00")
    tel.loc[tel["timestamp"].dt.date == pd.Timestamp("2024-01-03").date(), "pump_rpm"] = 1000.0
    out, _ = clean_telemetry(tel, cfg, write=False)
    at = lambda t: out.loc[out["timestamp"] == pd.Timestamp(t), "regime"].iat[0]  # noqa: E731
    assert at("2024-01-01 08:00") == "Day" and at("2024-01-01 19:59") == "Day"
    assert at("2024-01-01 07:59") == "Night" and at("2024-01-01 20:00") == "Night"
    assert at("2024-01-06 12:00") == "Weekend"
    assert at("2024-01-03 12:00") == "Standby"
    assert out.loc[out["timestamp"] == pd.Timestamp("2024-01-02 00:30"), "is_midnight_dip"].iat[0]
    assert sat.any()


def test_label_columns_are_dropped_and_regimes_agree_with_source(cfg):
    out, log = clean_telemetry(make_telemetry(days=3), cfg, write=False)
    assert not set(cfg["leakage_columns"]) & set(out.columns)
    assert log["regime_vs_source_shift_agreement"] == 1.0
