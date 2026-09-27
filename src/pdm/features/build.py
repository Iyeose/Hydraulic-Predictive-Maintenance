"""Feature engineering ("gold" layer): minute-level silver data → hourly, model-ready features.

Timing convention (important for leakage): a feature row stamped ``T`` summarises the minutes
``[T - 1h, T)`` and uses **only** data before ``T``. Rolling windows, lags and the FFT all look
backwards. ``tests/test_features.py`` enforces this by perturbing future data and checking that
past features do not change.

Feature groups
--------------
* ``hourly_raw``   : mean/std/min/max of raw sensors in the hour
* ``hourly_regime``: regime-normalised z-score and % deviation summaries (EDA §8.1)
* ``fleet_relative``: z-score minus the fleet median at the same hour. This removes slow drift that
  affects every machine at once, such as the ~1 °C seasonal oil-temperature rise (DQ-11)
* ``vibration``    : p95, kurtosis, crest factor, burst count
* ``cross_sensor`` : X−Y divergence, pressure-vs-flow deviation, within-hour correlations
* ``rolling``      : 6/24/72 h trailing mean, std, min, max and trend slope of key hourly signals
* ``lag``          : 1 h / 24 h lags and differences
* ``spectral``     : band-energy shares and spectral entropy of the trailing 6 h of vibration
* ``context``      : hour of day (cyclic), weekend, standby share
* ``quality``      : share of imputed / invalid / missing minutes
* ``maintenance``  : days since last maintenance, recent reactive actions (candidate only; see leak check)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Z_SIGNALS = ["pressure_bar", "temp_celsius", "flow_lpm", "vibration_x_g", "vibration_y_g"]
RAW = ["pressure_bar", "temp_celsius", "flow_lpm", "vibration_x_g", "vibration_y_g", "pump_rpm"]


def hourly_aggregate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    """Aggregate minute data into hourly rows stamped at the *end* of the hour."""
    fc = cfg["features"]
    reg: dict[str, tuple[str, str]] = {}
    d = df.copy()
    d["T"] = d["timestamp"].dt.floor("1h") + pd.Timedelta("1h")

    # Helper columns so that moments/correlations can be aggregated with fast built-in reducers
    vx = d["vibration_x_g"]
    d["_vx2"], d["_vx3"], d["_vx4"] = vx ** 2, vx ** 3, vx ** 4
    d["_burst"] = (vx > fc["vibration_burst_g"]).astype(float).where(vx.notna())
    d["_xy"] = d["vibration_x_g"] - d["vibration_y_g"]
    d["_xy_abs"] = d["_xy"].abs()
    d["_xy_nonzero"] = (d["_xy_abs"] > 1e-9).astype(float).where(d["_xy"].notna())
    d["_xy_div"] = (d["_xy_abs"] > fc["xy_divergence_g"]).astype(float).where(d["_xy"].notna())
    d["_pf"] = d["pressure_bar"] * d["flow_lpm"]
    d["_tf"] = d["temp_celsius"] * d["flow_lpm"]
    d["_p2"], d["_f2"], d["_t2"] = d["pressure_bar"] ** 2, d["flow_lpm"] ** 2, d["temp_celsius"] ** 2
    d["_p_minus_f"] = d["pct_pressure_bar"] - d["pct_flow_lpm"]
    d["_nan"] = d[RAW].isna().any(axis=1).astype(float)

    agg = {}
    for s in RAW:
        for f in ("mean", "std", "min", "max"):
            agg[f"{s}_{f}"] = (s, f)
            reg[f"{s}_{f}"] = ("hourly_raw", f"hourly {f} of raw {s}")
    for s in Z_SIGNALS:
        for f in ("mean", "min", "max"):
            agg[f"z_{s}_{f}"] = (f"z_{s}", f)
            reg[f"z_{s}_{f}"] = ("hourly_regime", f"hourly {f} of {s} z-score vs own same-hour healthy baseline")
        agg[f"pct_{s}_mean"] = (f"pct_{s}", "mean")
        reg[f"pct_{s}_mean"] = ("hourly_regime", f"hourly mean % deviation of {s} from baseline")
    agg.update({
        "vib_x_p95": ("vibration_x_g", lambda x: x.quantile(0.95)),
        "_vx2": ("_vx2", "mean"), "_vx3": ("_vx3", "mean"), "_vx4": ("_vx4", "mean"),
        "vib_x_burst_count": ("_burst", "sum"),
        "xy_delta_mean": ("_xy", "mean"), "xy_delta_absmax": ("_xy_abs", "max"),
        "xy_nonzero_share": ("_xy_nonzero", "mean"),
        "xy_divergence_share": ("_xy_div", "mean"),
        "pct_pressure_minus_flow_mean": ("_p_minus_f", "mean"),
        "_pf": ("_pf", "mean"), "_tf": ("_tf", "mean"), "_p2": ("_p2", "mean"), "_f2": ("_f2", "mean"), "_t2": ("_t2", "mean"),
        "imputed_share": ("was_imputed", "mean"), "invalid_count": ("n_invalid_sensors", "sum"),
        "missing_share": ("_nan", "mean"), "standby_share": ("is_standby", "mean"),
        "n_minutes": ("timestamp", "size"),
    })
    # The quantile lambda is slow on 14k groups; compute p95 separately with a vectorised groupby quantile
    p95_spec = agg.pop("vib_x_p95")
    h = d.groupby(["machine_id", "T"], sort=True).agg(**agg)
    h["vib_x_p95"] = d.groupby(["machine_id", "T"], sort=True)[p95_spec[0]].quantile(0.95)

    # Moment-based vibration shape features
    m1 = h["vibration_x_g_mean"]
    var = (h["_vx2"] - m1 ** 2).clip(lower=1e-12)
    m4c = h["_vx4"] - 4 * m1 * h["_vx3"] + 6 * m1 ** 2 * h["_vx2"] - 3 * m1 ** 4
    h["vib_x_kurtosis"] = m4c / var ** 2 - 3
    h["vib_x_crest"] = h["vibration_x_g_max"] / np.sqrt(h["_vx2"]).replace(0, np.nan)

    # Within-hour Pearson correlations from aggregated moments
    def _corr(xy, x, y, x2, y2):
        cov = h[xy] - h[x] * h[y]
        return cov / np.sqrt((h[x2] - h[x] ** 2).clip(lower=1e-12) * (h[y2] - h[y] ** 2).clip(lower=1e-12))
    h["corr_pressure_flow"] = _corr("_pf", "pressure_bar_mean", "flow_lpm_mean", "_p2", "_f2")
    h["corr_temp_flow"] = _corr("_tf", "temp_celsius_mean", "flow_lpm_mean", "_t2", "_f2")
    h = h.drop(columns=[c for c in h.columns if c.startswith("_")])

    reg.update({
        "vib_x_p95": ("vibration", "hourly 95th percentile of vibration X (g)"),
        "vib_x_burst_count": ("vibration", f"minutes in the hour with vibration X > {fc['vibration_burst_g']} g"),
        "vib_x_kurtosis": ("vibration", "excess kurtosis of vibration X within the hour (impulsiveness)"),
        "vib_x_crest": ("vibration", "crest factor = max / RMS of vibration X within the hour"),
        "xy_delta_mean": ("cross_sensor", "mean of vibration X − Y (g); ~0 when healthy (EDA §4.6, §8.4)"),
        "xy_delta_absmax": ("cross_sensor", "max |vibration X − Y| in the hour (g)"),
        "xy_nonzero_share": ("cross_sensor", "share of minutes where X and Y vibration differ"),
        "xy_divergence_share": ("cross_sensor", f"share of minutes with |X − Y| > {fc['xy_divergence_g']} g"),
        "pct_pressure_minus_flow_mean": ("cross_sensor", "pressure % deviation minus flow % deviation: <0 pump wear, >0 valve leakage"),
        "corr_pressure_flow": ("cross_sensor", "within-hour correlation of pressure and flow"),
        "corr_temp_flow": ("cross_sensor", "within-hour correlation of temperature and flow"),
        "imputed_share": ("quality", "share of minutes forward-filled by cleaning"),
        "invalid_count": ("quality", "number of physically impossible readings removed in the hour"),
        "missing_share": ("quality", "share of minutes with at least one sensor still missing"),
        "standby_share": ("context", "share of minutes in standby (rpm < threshold)"),
        "n_minutes": ("quality", "minutes available in the hour"),
    })
    return h.reset_index().rename(columns={"T": "timestamp"}), reg


def _rolling_slope(y: pd.Series, w: int, minp: int) -> pd.Series:
    """Least-squares slope per hour over a trailing window, vectorised via rolling moments."""
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index).where(y.notna())
    r = lambda s: s.rolling(w, min_periods=minp)  # noqa: E731
    mt, my = r(t).mean(), r(y).mean()
    cov = r(t * y).mean() - mt * my
    var = r(t * t).mean() - mt ** 2
    return cov / var.replace(0, np.nan)


ROLL_BASE = ["z_pressure_bar_mean", "z_temp_celsius_mean", "fz_temp_celsius_mean", "z_flow_lpm_mean", "z_vibration_x_g_mean",
             "z_vibration_x_g_max", "vib_x_kurtosis", "xy_delta_absmax", "pct_pressure_minus_flow_mean",
             "corr_pressure_flow"]
LAG_BASE = ["z_pressure_bar_mean", "z_temp_celsius_mean", "z_flow_lpm_mean", "z_vibration_x_g_mean", "xy_delta_mean"]


FLEET_BASE = ["z_pressure_bar_mean", "z_temp_celsius_mean", "z_flow_lpm_mean", "z_vibration_x_g_mean"]


def add_fleet_relative(h: pd.DataFrame, reg: dict) -> pd.DataFrame:
    """Subtract the fleet median of the same hour (a causal, cross-sectional normalisation)."""
    h = h.copy()
    for c in FLEET_BASE:
        med = h.groupby("timestamp")[c].transform("median")
        h["f" + c] = h[c] - med
        reg["f" + c] = ("fleet_relative", f"{c} minus the fleet median at the same hour")
    return h


def add_rolling_and_lags(h: pd.DataFrame, cfg: dict, reg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    parts = []
    for _, g in h.groupby("machine_id", sort=True):
        g = g.sort_values("timestamp").copy()
        new = {}
        for c in ROLL_BASE:
            for w in fc["rolling_windows_h"]:
                minp = max(2, w // 2)
                r = g[c].rolling(w, min_periods=minp)
                new[f"{c}_roll{w}h_mean"] = r.mean()
                new[f"{c}_roll{w}h_std"] = r.std()
                new[f"{c}_roll{w}h_min"] = r.min()
                new[f"{c}_roll{w}h_max"] = r.max()
                new[f"{c}_roll{w}h_slope"] = _rolling_slope(g[c], w, minp)
        for c in LAG_BASE:
            for L in fc["lags_h"]:
                new[f"{c}_lag{L}h"] = g[c].shift(L)
                new[f"{c}_diff{L}h"] = g[c] - g[c].shift(L)
        parts.append(pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1))
    for c in ROLL_BASE:
        for w in fc["rolling_windows_h"]:
            for st in ("mean", "std", "min", "max", "slope"):
                reg[f"{c}_roll{w}h_{st}"] = ("rolling", f"trailing {w} h {st} of {c}" + (" (per hour)" if st == "slope" else ""))
    for c in LAG_BASE:
        for L in fc["lags_h"]:
            reg[f"{c}_lag{L}h"] = ("lag", f"{c} {L} h earlier")
            reg[f"{c}_diff{L}h"] = ("lag", f"change in {c} over {L} h")
    return pd.concat(parts, ignore_index=True)


def spectral_features(silver: pd.DataFrame, hourly_index: pd.DataFrame, cfg: dict, reg: dict) -> pd.DataFrame:
    """FFT band-energy shares of the trailing ``spectral_window_min`` minutes of vibration X.

    Note: 1-minute sampling cannot resolve mechanical vibration frequencies (that needs kHz
    raw data). These features describe how *bursty* the amplitude series is over hours.
    """
    fc = cfg["features"]
    W = int(fc["spectral_window_min"])
    bands = fc["spectral_bands"]
    freqs = np.fft.rfftfreq(W, d=1.0)          # cycles per minute, 0 … 0.5
    out = []
    for mid, g in silver.groupby("machine_id", sort=True):
        ts = g["timestamp"].to_numpy()
        x = g["vibration_x_g"].to_numpy(dtype=float)
        targets = hourly_index.loc[hourly_index["machine_id"] == mid, "timestamp"].to_numpy()
        end = np.searchsorted(ts, targets, side="left")          # minutes strictly before T
        ok = end >= W
        res = pd.DataFrame({"machine_id": mid, "timestamp": targets})
        cols = {f"vib_spec_band{i}_share": np.full(len(targets), np.nan) for i in range(len(bands))}
        cols["vib_spec_entropy"] = np.full(len(targets), np.nan)
        cols["vib_spec_log_power"] = np.full(len(targets), np.nan)
        if ok.any():
            win = sliding_window_view(x, W)[end[ok] - W]          # (n, W), rows end at minute T-1
            med = np.nanmedian(win, axis=1, keepdims=True)
            win = np.where(np.isnan(win), med, win)
            win = win - win.mean(axis=1, keepdims=True)
            p = np.abs(np.fft.rfft(win, axis=1)) ** 2
            p[:, 0] = 0.0
            tot = p.sum(axis=1)
            safe = np.where(tot > 0, tot, np.nan)
            for i, (lo, hi) in enumerate(bands):
                m = (freqs > lo) & (freqs <= hi)
                cols[f"vib_spec_band{i}_share"][ok] = p[:, m].sum(axis=1) / safe
            pn = p / safe[:, None]
            cols["vib_spec_entropy"][ok] = -np.nansum(pn * np.log(pn + 1e-12), axis=1) / np.log(p.shape[1])
            cols["vib_spec_log_power"][ok] = np.log10(tot / W + 1e-12)
        out.append(res.assign(**cols))
    for i, (lo, hi) in enumerate(bands):
        reg[f"vib_spec_band{i}_share"] = ("spectral", f"share of trailing {W}-min vibration power in {lo}–{hi} cycles/min")
    reg["vib_spec_entropy"] = ("spectral", "normalised spectral entropy of trailing vibration (1 = noise-like)")
    reg["vib_spec_log_power"] = ("spectral", "log10 total variance of trailing vibration")
    return pd.concat(out, ignore_index=True)


def context_features(h: pd.DataFrame, reg: dict) -> pd.DataFrame:
    h = h.copy()
    start = h["timestamp"] - pd.Timedelta("1h")                 # the hour the row describes
    hr = start.dt.hour
    h["hour_sin"], h["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    h["is_weekend"] = start.dt.dayofweek.isin([5, 6]).astype(int)
    h["hour_of_day"] = hr
    reg.update({"hour_sin": ("context", "hour of day (cyclic sine)"), "hour_cos": ("context", "hour of day (cyclic cosine)"),
                "is_weekend": ("context", "1 if the hour is on Sat/Sun"), "hour_of_day": ("context", "hour of day 0–23")})
    return h


def maintenance_features(h: pd.DataFrame, mnt: pd.DataFrame, reg: dict) -> pd.DataFrame:
    """Days since last maintenance / reactive action and count of reactive actions in the prior 90 days."""
    h = h.sort_values("timestamp").copy()
    m = mnt.sort_values("action_timestamp")[["machine_id", "action_timestamp", "action_type"]]
    last_any = pd.merge_asof(h[["machine_id", "timestamp"]], m.rename(columns={"action_timestamp": "last_any"}),
                             left_on="timestamp", right_on="last_any", by="machine_id", direction="backward")
    r = m[m["action_type"] == "Reactive"]
    last_re = pd.merge_asof(h[["machine_id", "timestamp"]], r.rename(columns={"action_timestamp": "last_re"}),
                            left_on="timestamp", right_on="last_re", by="machine_id", direction="backward")
    h["days_since_maintenance"] = ((h["timestamp"] - last_any["last_any"].values) / pd.Timedelta("1D")).to_numpy()
    h["days_since_reactive"] = ((h["timestamp"] - last_re["last_re"].values) / pd.Timedelta("1D")).to_numpy()
    cnt = []
    for mid, g in h.groupby("machine_id", sort=False):
        rt = np.sort(r.loc[r["machine_id"] == mid, "action_timestamp"].to_numpy())
        T = g["timestamp"].to_numpy()
        cnt.append(pd.Series(np.searchsorted(rt, T) - np.searchsorted(rt, T - np.timedelta64(90, "D")), index=g.index))
    h["reactive_actions_90d"] = pd.concat(cnt)
    reg.update({"days_since_maintenance": ("maintenance", "days since the last maintenance action of any type"),
                "days_since_reactive": ("maintenance", "days since the last reactive (breakdown) repair"),
                "reactive_actions_90d": ("maintenance", "reactive actions in the previous 90 days")})
    return h


def build_features(silver_z: pd.DataFrame, maintenance: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature build. Returns (hourly feature table, feature dictionary)."""
    h, reg = hourly_aggregate(silver_z, cfg)
    h = add_fleet_relative(h, reg)
    h = add_rolling_and_lags(h, cfg, reg)
    spec = spectral_features(silver_z, h[["machine_id", "timestamp"]], cfg, reg)
    h = h.merge(spec, on=["machine_id", "timestamp"], how="left")
    h = context_features(h, reg)
    h = maintenance_features(h, maintenance, reg)
    # Baseline-period flag (z-scores are in-sample there) and regime label of the hour
    ref_end = silver_z.groupby("machine_id")["timestamp"].min() + pd.Timedelta(days=cfg["baseline"]["reference_days"])
    h["in_baseline_period"] = h["timestamp"] <= h["machine_id"].map(ref_end)
    regime = (silver_z.assign(T=silver_z["timestamp"].dt.floor("1h") + pd.Timedelta("1h"))
                      .groupby(["machine_id", "T"])["regime"].agg(lambda s: s.mode().iat[0]).rename("regime"))
    h = h.merge(regime.reset_index().rename(columns={"T": "timestamp"}), on=["machine_id", "timestamp"], how="left")
    h = h.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    dictionary = (pd.DataFrame([(k, g, d) for k, (g, d) in reg.items()], columns=["feature", "group", "description"])
                    .query("feature in @h.columns").reset_index(drop=True))
    return h, dictionary
