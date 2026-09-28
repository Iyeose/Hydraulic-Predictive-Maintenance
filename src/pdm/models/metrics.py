"""Evaluation metrics for the three tasks.

* Failure mode: macro-F1 and per-class confusion matrix, plus event-level mode identification.
* RUL: MAE, RMSE and the asymmetric NASA (PHM08) score, on all rows with a known RUL and on the
  "critical" rows where the true RUL is below the cap (the last 14 days before a failure).
* Early warning: hourly precision / recall, the realised false-alarm rate, and the warning
  **lead time** per failure event at a fixed false-alarm rate.

Alarm logic is shared by the rule baseline and the ML models: a score above the threshold only
becomes an alarm after ``persistence`` consecutive hours, and never during standby or repair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)


# --- RUL ------------------------------------------------------------------------------------------
def nasa_score(y_true, y_pred, unit_hours: float = 24.0, a_early: float = 13.0, a_late: float = 10.0) -> float:
    """Mean asymmetric PHM08 score (lower is better; 0 = perfect).

    d = (predicted − true) / unit_hours. A late prediction (predicted RUL too long, d > 0) costs
    exp(d / a_late) − 1, an early one exp(−d / a_early) − 1, so being late is penalised more.
    The mean (not the sum) is reported so that folds of different length are comparable.
    """
    d = (np.asarray(y_pred, float) - np.asarray(y_true, float)) / unit_hours
    s = np.where(d > 0, np.exp(d / a_late) - 1, np.exp(-d / a_early) - 1)
    return float(np.mean(s)) if len(s) else float("nan")


def regression_metrics(y_true, y_pred, cap: float, nasa: dict) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {}
    for name, m in (("all", np.ones_like(y_true, bool)), ("critical", y_true < cap)):
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        out[f"mae_{name}"] = float(mean_absolute_error(yt, yp))
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out[f"nasa_{name}"] = nasa_score(yt, yp, **nasa)
        out[f"n_{name}"] = int(m.sum())
    return out


# --- failure mode ---------------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, classes: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true, object), np.asarray(y_pred, object)
    present = [c for c in classes if c in set(y_true) | set(y_pred)]
    f1 = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    deg = y_true != "healthy"
    return {
        "macro_f1": float(np.mean(f1)) if len(f1) else float("nan"),
        "f1_per_class": dict(zip(present, map(float, f1))),
        "accuracy": float(np.mean(y_true == y_pred)),
        # "something is wrong": any failure mode predicted during a labelled degradation window
        "detection_recall": float(np.mean(y_pred[deg] != "healthy")) if deg.any() else float("nan"),
        "healthy_false_positive_rate": float(np.mean(y_pred[~deg] != "healthy")) if (~deg).any() else float("nan"),
    }


def confusion_frame(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(np.asarray(y_true, object), np.asarray(y_pred, object), labels=classes)
    return pd.DataFrame(cm, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted"))


def event_mode_identification(pred: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per failure event: the most frequent non-healthy prediction inside its degradation window.

    ``pred`` needs machine_id, timestamp, y_pred. This is how an engineer reads the model: over the
    days before a failure, which mode did it mostly point to?
    """
    rows = []
    for _, e in events.iterrows():
        w = pred[(pred["machine_id"] == e.machine_id) & (pred["timestamp"] > e.degradation_start_timestamp)
                 & (pred["timestamp"] <= e.failure_timestamp)]
        flagged = w.loc[w["y_pred"] != "healthy", "y_pred"]
        top = flagged.value_counts().index[0] if len(flagged) else "healthy"
        rows.append({"event_id": e.event_id, "machine_id": e.machine_id, "true_mode": e.failure_mode, "predicted_mode": top,
                     "share_hours_flagged": float(len(flagged) / max(len(w), 1)),
                     "share_flagged_as_true_mode": float((flagged == e.failure_mode).mean()) if len(flagged) else 0.0,
                     "correct": top == e.failure_mode})
    return pd.DataFrame(rows)


# --- alarms and early warning ---------------------------------------------------------------------
def sustained(flag: pd.Series | np.ndarray, groups: pd.Series | np.ndarray, persistence: int) -> np.ndarray:
    """True where ``flag`` has been True for ``persistence`` consecutive rows within each group.

    Rows must be sorted by group and time on a complete hourly grid (the Stage 2 dataset is).
    """
    f = pd.Series(np.asarray(flag, bool).astype(np.int8))
    if persistence <= 1:
        return f.to_numpy().astype(bool)
    g = pd.Series(np.asarray(groups))
    return (f.groupby(g.values).rolling(persistence, min_periods=persistence).min()
             .reset_index(level=0, drop=True).sort_index().fillna(0).to_numpy().astype(bool))


def alarms_from_scores(scores, groups, active, threshold: float, persistence: int) -> np.ndarray:
    """Alarm = score ≥ threshold on an active hour, sustained for ``persistence`` hours."""
    above = (np.asarray(scores, float) >= threshold) & np.asarray(active, bool)
    return sustained(above, groups, persistence)


def false_alarm_rate(alarm, y, scored) -> float:
    """Share of scored negative hours (fail_within_h == 0) that are in alarm."""
    neg = np.asarray(scored, bool) & (np.asarray(y, float) == 0)
    return float(np.asarray(alarm, bool)[neg].mean()) if neg.any() else float("nan")


def threshold_for_far(scores, y, groups, active, scored, target: float, persistence: int) -> float:
    """Lowest threshold whose false-alarm rate is ≤ ``target``.

    FAR is non-increasing in the threshold (a higher threshold gives a subset of alarms, also after
    the persistence filter), so a binary search over the observed scores finds it exactly.
    """
    s = np.asarray(scores, float)
    cand = np.unique(s[np.asarray(active, bool) & ~np.isnan(s)])
    if len(cand) == 0:
        return float("inf")
    far = lambda t: false_alarm_rate(alarms_from_scores(s, groups, active, t, persistence), y, scored)  # noqa: E731
    lo, hi = 0, len(cand) - 1
    if far(cand[hi]) > target:          # even the highest score alarms too often
        return float(np.nextafter(cand[hi], np.inf))
    while lo < hi:
        mid = (lo + hi) // 2
        if far(cand[mid]) <= target:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])


def lead_times(frame: pd.DataFrame, events: pd.DataFrame, horizon_h: float) -> pd.DataFrame:
    """Warning lead time per event: failure time − first alarm in the event's warning window.

    The window starts at the earlier of the labelled degradation start and ``failure − horizon``.
    Alarms before that are false alarms (counted in the false-alarm rate), not early warnings.
    ``frame`` needs machine_id, timestamp, alarm. Events on machines absent from ``frame`` are skipped.
    """
    rows = []
    for _, e in events[events["machine_id"].isin(frame["machine_id"].unique())].iterrows():
        start = min(e.degradation_start_timestamp, e.failure_timestamp - pd.Timedelta(hours=horizon_h))
        w = frame[(frame["machine_id"] == e.machine_id) & (frame["timestamp"] > start)
                  & (frame["timestamp"] <= e.failure_timestamp) & frame["alarm"]]
        lead = (e.failure_timestamp - w["timestamp"].min()) / pd.Timedelta("1h") if len(w) else 0.0
        rows.append({"event_id": e.event_id, "machine_id": e.machine_id, "failure_mode": e.failure_mode,
                     "window_h": (e.failure_timestamp - start) / pd.Timedelta("1h"), "lead_time_h": float(lead),
                     "detected": bool(len(w))})
    return pd.DataFrame(rows)


def alarm_episodes(alarm, groups) -> int:
    """Number of distinct alarm episodes (rising edges) across machines."""
    a = pd.Series(np.asarray(alarm, bool).astype(int))
    prev = a.groupby(np.asarray(groups)).shift(1).fillna(0)
    return int(((a == 1) & (prev == 0)).sum())


def warning_metrics(frame: pd.DataFrame, events: pd.DataFrame, horizon_h: float) -> tuple[dict, pd.DataFrame]:
    """Operating-point metrics. ``frame``: machine_id, timestamp, y (fail_within_h), scored (bool), alarm, score."""
    sc = frame["scored"].to_numpy(bool)
    y = frame["y"].to_numpy(float)[sc]
    a = frame["alarm"].to_numpy(bool)[sc]
    neg_frame = frame[frame["scored"] & (frame["y"] == 0)]
    lt = lead_times(frame, events, horizon_h)
    out = {
        "precision": float(precision_score(y, a, zero_division=0)),
        "recall": float(recall_score(y, a, zero_division=0)),
        "false_alarm_rate": false_alarm_rate(frame["alarm"], frame["y"], frame["scored"]),
        "false_alarm_episodes_per_1000h": 1000 * alarm_episodes(neg_frame["alarm"], neg_frame["machine_id"]) / max(len(neg_frame), 1),
        "events_detected": int(lt["detected"].sum()) if len(lt) else 0,
        "events_total": int(len(lt)),
        "lead_time_median_h": float(lt["lead_time_h"].median()) if len(lt) else float("nan"),
        "lead_time_mean_h": float(lt["lead_time_h"].mean()) if len(lt) else float("nan"),
    }
    s = frame["score"].to_numpy(float)[sc]
    if len(np.unique(y)) == 2 and len(np.unique(s)) > 2:           # ranking metrics only for real scores
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
        if np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
            out["brier"] = float(brier_score_loss(y, s))
    return out, lt
