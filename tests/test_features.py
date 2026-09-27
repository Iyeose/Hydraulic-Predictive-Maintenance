import numpy as np
import pandas as pd

from conftest import make_maintenance, make_telemetry
from pdm.data.clean import clean_telemetry
from pdm.features.baseline import RegimeBaseline
from pdm.features.build import _rolling_slope, build_features


def _features(cfg, tel):
    s = clean_telemetry(tel, cfg, write=False)[0]
    s = RegimeBaseline(cfg["baseline"]["signals"], 14).fit_transform(s)
    return build_features(s, make_maintenance(), cfg)[0], s


def test_row_T_summarises_the_previous_hour(cfg):
    feats, s = _features(cfg, make_telemetry(days=16))
    T = pd.Timestamp("2024-01-10 13:00")
    minutes = s[(s["machine_id"] == "HPU_01") & (s["timestamp"] >= T - pd.Timedelta("1h")) & (s["timestamp"] < T)]
    row = feats[(feats["machine_id"] == "HPU_01") & (feats["timestamp"] == T)].iloc[0]
    assert np.isclose(row["pressure_bar_mean"], minutes["pressure_bar"].mean())
    assert np.isclose(row["vibration_x_g_max"], minutes["vibration_x_g"].max())


def test_no_look_ahead(cfg):
    """Changing the future must not change any feature computed before it."""
    tel = make_telemetry(days=18)
    base, _ = _features(cfg, tel)
    cutoff = pd.Timestamp("2024-01-16 12:00")
    fut = tel["timestamp"] >= cutoff
    tel2 = tel.copy()
    tel2.loc[fut, "pressure_bar"] *= 0.5
    tel2.loc[fut, "vibration_x_g"] += 5.0
    tel2.loc[fut, "temp_celsius"] += 10.0
    pert, _ = _features(cfg, tel2)
    cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    past_b = base[base["timestamp"] <= cutoff].set_index(["machine_id", "timestamp"])[cols]
    past_p = pert[pert["timestamp"] <= cutoff].set_index(["machine_id", "timestamp"])[cols]
    pd.testing.assert_frame_equal(past_b, past_p)
    # sanity: the perturbation does change features after the cutoff
    after = base["timestamp"] > cutoff + pd.Timedelta("1h")
    assert not np.allclose(base.loc[after, "pressure_bar_mean"], pert.loc[after, "pressure_bar_mean"])


def test_rolling_slope_recovers_a_linear_trend():
    y = pd.Series(3.0 + 0.5 * np.arange(50))
    assert np.allclose(_rolling_slope(y, 10, 5).dropna(), 0.5)


def test_spectral_features_locate_a_periodic_signal(cfg):
    tel = make_telemetry(machines=("HPU_01",), days=16)
    n = np.arange(len(tel))
    tel["vibration_x_g"] = 0.3 + 0.2 * np.sin(2 * np.pi * 0.1 * n)        # 0.1 cycles/min → band 1
    tel["vibration_y_g"] = tel["vibration_x_g"]
    feats, _ = _features(cfg, tel)
    shares = feats[["vib_spec_band0_share", "vib_spec_band1_share", "vib_spec_band2_share"]].dropna()
    assert (shares["vib_spec_band1_share"] > 0.9).all()


def test_every_feature_is_documented(cfg):
    tel = make_telemetry(days=16)
    s = RegimeBaseline(cfg["baseline"]["signals"], 14).fit_transform(clean_telemetry(tel, cfg, write=False)[0])
    feats, dictionary = build_features(s, make_maintenance(), cfg)
    meta = {"machine_id", "timestamp", "regime", "in_baseline_period", "hour_of_day"}
    undocumented = set(feats.columns) - set(dictionary["feature"]) - meta
    assert not undocumented, undocumented
