import numpy as np
import pandas as pd

from conftest import make_failures
from pdm.features.targets import build_targets


def _rows(machines=("HPU_01", "HPU_02"), start="2024-01-01 01:00", end="2024-01-21 00:00"):
    T = pd.date_range(start, end, freq="1h")
    return pd.concat([pd.DataFrame({"machine_id": m, "timestamp": T}) for m in machines], ignore_index=True)


def _at(t, machine, ts):
    return t[(t["machine_id"] == machine) & (t["timestamp"] == pd.Timestamp(ts))].iloc[0]


def test_labels_rul_and_warning_before_failure(cfg):
    t = build_targets(_rows(), make_failures(), cfg)
    # failure 2024-01-18 00:00, degradation from 2024-01-16 00:00
    assert _at(t, "HPU_01", "2024-01-16 00:00")["failure_mode_label"] == "healthy"   # window 15th 23:00–24:00
    r = _at(t, "HPU_01", "2024-01-16 01:00")
    assert r["failure_mode_label"] == "pump_wear" and r["rul_hours"] == 47
    assert _at(t, "HPU_01", "2024-01-18 00:00")["rul_hours"] == 0
    cap = cfg["targets"]["rul_cap_hours"]
    assert _at(t, "HPU_01", "2024-01-02 00:00")["rul_hours"] == min(384, cap)
    assert _at(t, "HPU_01", "2024-01-11 00:00")["fail_within_h"] == 1   # 168 h before failure
    assert _at(t, "HPU_01", "2024-01-10 23:00")["fail_within_h"] == 0


def test_repair_window_is_excluded(cfg):
    t = build_targets(_rows(), make_failures(), cfg)   # downtime 5.5 h → repair until 05:30
    assert _at(t, "HPU_01", "2024-01-18 06:00")["in_repair"]            # window 05:00–06:00 overlaps repair
    assert np.isnan(_at(t, "HPU_01", "2024-01-18 06:00")["rul_hours"])
    after = _at(t, "HPU_01", "2024-01-18 07:00")
    assert not after["in_repair"] and after["failure_mode_label"] == "healthy" and after["life_cycle"] == 1


def test_right_censoring_without_observed_failure(cfg):
    rows = _rows()
    t = build_targets(rows, make_failures(), cfg)
    cap = cfg["targets"]["rul_cap_hours"]
    end = rows["timestamp"].max()
    early = _at(t, "HPU_02", end - pd.Timedelta(hours=cap + 5))
    late = _at(t, "HPU_02", end - pd.Timedelta(hours=10))
    assert early["rul_hours"] == cap and not early["rul_censored"]     # bound already ≥ cap → exactly cap
    assert np.isnan(late["rul_hours"]) and late["rul_censored"]
    assert pd.isna(late["fail_within_h"])
