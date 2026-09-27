import numpy as np
import pandas as pd

from conftest import make_telemetry
from pdm.data.clean import clean_telemetry
from pdm.features.baseline import RegimeBaseline


def _silver(cfg, **kw):
    return clean_telemetry(make_telemetry(**kw), cfg, write=False)[0]


def test_baseline_ignores_data_after_reference_period(cfg):
    s = _silver(cfg, days=20)
    b1 = RegimeBaseline(cfg["baseline"]["signals"], 14).fit(s).table_
    s2 = s.copy()
    s2.loc[s2["timestamp"] >= "2024-01-15", "pressure_bar"] *= 3
    b2 = RegimeBaseline(cfg["baseline"]["signals"], 14).fit(s2).table_
    pd.testing.assert_frame_equal(b1, b2)


def test_regime_offsets_are_removed(cfg):
    s = _silver(cfg, days=20)
    z = RegimeBaseline(cfg["baseline"]["signals"], 14).fit_transform(s)
    by_regime = z.groupby("regime")["z_pressure_bar"].mean()
    raw = s.groupby("regime")["pressure_bar"].mean()
    assert raw.max() - raw.min() > 20                      # raw set points differ by >20 bar …
    assert by_regime.abs().max() < 0.2                     # … normalised values are all ≈ 0


def test_standby_rows_get_no_health_score(cfg):
    tel = make_telemetry(days=16)
    tel.loc[tel["timestamp"].dt.date == pd.Timestamp("2024-01-15").date(), "pump_rpm"] = 1000.0
    s = clean_telemetry(tel, cfg, write=False)[0]
    z = RegimeBaseline(cfg["baseline"]["signals"], 14).fit_transform(s)
    assert z.loc[z["is_standby"], "z_pressure_bar"].isna().all()
    assert z.loc[~z["is_standby"], "z_pressure_bar"].notna().all()


def test_save_and_load_round_trip(cfg, tmp_path):
    s = _silver(cfg, days=15)
    b = RegimeBaseline(cfg["baseline"]["signals"], 14).fit(s)
    b.save(tmp_path / "b.parquet")
    b2 = RegimeBaseline.load(tmp_path / "b.parquet", cfg["baseline"]["signals"])
    np.testing.assert_allclose(b.transform(s)["z_flow_lpm"], b2.transform(s)["z_flow_lpm"])
