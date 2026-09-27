"""Target construction, rebuilt from the failure-event table (EDA §6.3, DQ-07).

For every feature row (machine, time T), where the row summarises the minutes [T - 1h, T):

* ``failure_mode_label``: the mode of the *next* failure if T lies inside its degradation
  window, otherwise ``"healthy"``.
* ``rul_hours``: ``min(next_failure - T, cap)``. This is the piecewise-linear capped RUL used
  in the C-MAPSS literature. When no failure is observed after T (for example HPU_10, or a
  machine after its repair), RUL is right-censored. We only know RUL ≥ time to end of data. If
  that bound already exceeds the cap, the capped RUL is known exactly (= cap). Otherwise it is
  NaN and ``rul_censored`` is True.
* ``fail_within_h``: binary early-warning target. It uses the same censoring logic.
* ``in_repair``: the machine is down after a failure (``downtime_hours``). These rows are
  excluded from training and evaluation.
* ``life_cycle``: counts failures so far. Each repair starts a new life cycle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets(rows: pd.DataFrame, failures: pd.DataFrame, cfg: dict,
                  data_end: pd.Series | None = None, grain: str = "1h") -> pd.DataFrame:
    """``rows`` needs columns ``machine_id`` and ``timestamp`` (T = end of the aggregation window).

    ``data_end`` maps machine_id to the end of its observed data (for censoring). It defaults
    to the last T per machine.
    """
    cap = float(cfg["targets"]["rul_cap_hours"])
    horizon = float(cfg["targets"]["warning_horizon_hours"])
    g = pd.Timedelta(grain)
    if data_end is None:
        data_end = rows.groupby("machine_id")["timestamp"].max()

    out = []
    for mid, r in rows.groupby("machine_id", sort=False):
        T = r["timestamp"].to_numpy().astype("datetime64[ns]")
        ev = failures[failures["machine_id"] == mid].sort_values("failure_timestamp")
        f_ts = ev["failure_timestamp"].to_numpy().astype("datetime64[ns]")
        # index of the next failure at or after T (T == failure means the window ends at the failure)
        nxt = np.searchsorted(f_ts, T, side="left")
        has_next = nxt < len(f_ts)
        nxt_c = np.minimum(nxt, max(len(f_ts) - 1, 0))

        res = pd.DataFrame(index=r.index)
        res["life_cycle"] = nxt  # number of failures strictly before T
        if len(ev):
            res["next_event_id"] = np.where(has_next, ev["event_id"].to_numpy()[nxt_c], None)
            htf = np.where(has_next, (f_ts[nxt_c] - T) / np.timedelta64(1, "h"), np.nan)
            deg = ev["degradation_start_timestamp"].to_numpy().astype("datetime64[ns]")[nxt_c]
            in_window = has_next & ((T - g.to_timedelta64()) >= deg)
            res["failure_mode_label"] = np.where(in_window, ev["failure_mode"].to_numpy()[nxt_c], "healthy")
            # repair: the window [T-1h, T) overlaps [previous failure, previous failure + downtime)
            prev = nxt - 1
            prev_c = np.maximum(prev, 0)
            downtime = pd.to_timedelta(ev["downtime_hours"].to_numpy()[prev_c], unit="h").to_numpy().astype("timedelta64[ns]")
            repair_end = f_ts[prev_c] + downtime
            res["in_repair"] = (prev >= 0) & ((T - g.to_timedelta64()) < repair_end)
        else:
            res["next_event_id"] = None
            htf = np.full(len(r), np.nan)
            res["failure_mode_label"] = "healthy"
            res["in_repair"] = False

        bound = (pd.Timestamp(data_end[mid]).to_datetime64().astype("datetime64[ns]") - T) / np.timedelta64(1, "h")   # censoring bound
        res["hours_to_failure"] = htf
        rul = np.where(has_next, np.minimum(htf, cap), np.where(bound >= cap, cap, np.nan))
        res["rul_hours"] = rul
        res["rul_censored"] = np.isnan(rul)
        warn = np.where(has_next, (htf <= horizon).astype(float), np.where(bound >= horizon, 0.0, np.nan))
        res["fail_within_h"] = warn
        # repair rows carry no valid target
        res.loc[res["in_repair"], ["rul_hours", "fail_within_h"]] = np.nan
        res.loc[res["in_repair"], "failure_mode_label"] = "repair"
        out.append(res)

    tgt = rows[["machine_id", "timestamp"]].join(pd.concat(out))
    tgt["is_degrading"] = ~tgt["failure_mode_label"].isin(["healthy", "repair"])
    tgt["fail_within_h"] = tgt["fail_within_h"].astype("Float32")
    return tgt
