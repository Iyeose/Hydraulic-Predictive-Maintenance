"""Leave-one-machine-out (LOMO) evaluation for the three tasks.

Fold *k* holds out one machine completely (Stage 2 ``folds.csv``). Every number reported in
Stage 3 is an out-of-fold number: the model that scored a machine never saw any hour of it.

The early-warning task needs two extra decisions, a probability calibration and an alarm
threshold. Both are made **inside** each outer fold with an inner LOMO over the nine training
machines, so the held-out machine influences neither of them::

    for each test machine m:
        for each training machine j ≠ m:   fit on the other 8, score j        → inner out-of-fold scores
        fit calibrator + choose threshold on the inner scores (target false-alarm rate)
        fit on all 9 training machines, score m, apply calibrator and threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .baselines import RuleBaseline
from .data import ModelData, classification_rows, early_degradation, fold_split, rul_rows, warning_rows, warning_train_rows
from .estimators import make_classifier, make_regressor, positive_proba
from .metrics import (alarms_from_scores, classification_metrics, confusion_frame, event_mode_identification,
                      regression_metrics, threshold_for_far, warning_metrics)


@dataclass
class TaskResult:
    task: str
    model: str
    oof: pd.DataFrame
    per_fold: pd.DataFrame
    summary: dict
    tables: dict = field(default_factory=dict)      # extra tables logged as artifacts
    final: dict = field(default_factory=dict)       # e.g. calibrator / threshold for the deployed model


def _rule_baseline(cfg: dict) -> RuleBaseline:
    rb = cfg["modelling"]["rule_baseline"]
    return RuleBaseline(rb["rules"], rb["window_h"], rb["persistence_h"])


def _params(cfg: dict, name: str) -> dict:
    return dict(cfg["modelling"]["models"].get(name, {}))


# --- failure-mode classification ------------------------------------------------------------------
def run_classification(data: ModelData, cfg: dict, model: str) -> TaskResult:
    ds, X_cols, classes, seed = data.ds, data.features, cfg["modelling"]["classes"], cfg["random_state"]
    mask = classification_rows(ds)
    rule_mode = _rule_baseline(cfg).predict(ds)["mode"] if model == "rules" else None
    parts, fold_rows = [], []
    for _, f in data.folds.iterrows():
        tr, te = fold_split(ds, mask, f.test_machine)
        if model == "rules":
            y_pred, proba = rule_mode.iloc[te].to_numpy(), None
        else:
            est = make_classifier(model, _params(cfg, model), seed).fit(ds[X_cols].iloc[tr], ds["failure_mode_label"].iloc[tr])
            p = est.predict_proba(ds[X_cols].iloc[te])
            proba = pd.DataFrame(0.0, index=range(len(te)), columns=classes)
            proba[list(est.classes_)] = p
            y_pred = np.asarray(classes, object)[proba.to_numpy().argmax(axis=1)]
        part = ds.iloc[te][["machine_id", "timestamp", "fold"]].assign(y_true=ds["failure_mode_label"].iloc[te].to_numpy(), y_pred=y_pred)
        if proba is not None:
            part = pd.concat([part.reset_index(drop=True), proba.add_prefix("p_")], axis=1)
        parts.append(part)
        m = classification_metrics(part["y_true"], part["y_pred"], classes)
        fold_rows.append({"fold": f.fold, "test_machine": f.test_machine, "test_modes": f.test_failure_modes,
                          "unseen_mode": f.unseen_mode_in_train if isinstance(f.unseen_mode_in_train, str) else "",
                          "macro_f1": m["macro_f1"], "detection_recall": m["detection_recall"],
                          "healthy_fpr": m["healthy_false_positive_rate"]})
    oof = pd.concat(parts, ignore_index=True)
    per_fold = pd.DataFrame(fold_rows)

    unseen_folds = per_fold.loc[per_fold["unseen_mode"] != "", "fold"].tolist()
    seen = oof[~oof["fold"].isin(unseen_folds)]
    m_all = classification_metrics(oof["y_true"], oof["y_pred"], classes)
    m_seen = classification_metrics(seen["y_true"], seen["y_pred"], classes)
    ev = event_mode_identification(oof, data.events)
    summary = {
        # headline: pooled out-of-fold macro-F1 over folds whose failure mode was seen in training
        "macro_f1_seen": m_seen["macro_f1"],
        "macro_f1_all": m_all["macro_f1"],
        "accuracy_seen": m_seen["accuracy"],
        "detection_recall": m_all["detection_recall"],
        "healthy_false_positive_rate": m_all["healthy_false_positive_rate"],
        "macro_f1_fold_mean": float(per_fold["macro_f1"].mean()),
        "events_mode_correct": int(ev["correct"].sum()), "events_total": int(len(ev)),
        **{f"f1_seen_{k}": v for k, v in m_seen["f1_per_class"].items()},
    }
    for fo in unseen_folds:                         # reported separately, never hidden in an average
        u = oof[oof["fold"] == fo]
        summary[f"unseen_fold{fo}_detection_recall"] = classification_metrics(u["y_true"], u["y_pred"], classes)["detection_recall"]
    tables = {"confusion_seen": confusion_frame(seen["y_true"], seen["y_pred"], classes),
              "confusion_all": confusion_frame(oof["y_true"], oof["y_pred"], classes),
              "per_fold": per_fold, "events": ev}
    return TaskResult("failure_mode", model, oof, per_fold, summary, tables)


# --- RUL regression -------------------------------------------------------------------------------
def run_rul(data: ModelData, cfg: dict, model: str) -> TaskResult:
    ds, X_cols, seed = data.ds, data.features, cfg["random_state"]
    cap, nasa = float(cfg["targets"]["rul_cap_hours"]), cfg["modelling"]["rul"]["nasa_score"]
    mask = rul_rows(ds)
    parts, fold_rows = [], []
    for _, f in data.folds.iterrows():
        tr, te = fold_split(ds, mask, f.test_machine)
        est = make_regressor(model, _params(cfg, model), seed).fit(ds[X_cols].iloc[tr], ds["rul_hours"].iloc[tr])
        y_pred = np.clip(est.predict(ds[X_cols].iloc[te]), 0, cap)
        part = ds.iloc[te][["machine_id", "timestamp", "fold"]].assign(y_true=ds["rul_hours"].iloc[te].to_numpy(), y_pred=y_pred)
        parts.append(part)
        fold_rows.append({"fold": f.fold, "test_machine": f.test_machine, "test_modes": f.test_failure_modes,
                          **regression_metrics(part["y_true"], part["y_pred"], cap, nasa)})
    oof = pd.concat(parts, ignore_index=True)
    per_fold = pd.DataFrame(fold_rows)
    summary = regression_metrics(oof["y_true"], oof["y_pred"], cap, nasa)
    return TaskResult("rul", model, oof, per_fold, summary, {"per_fold": per_fold})


# --- early warning --------------------------------------------------------------------------------
def fit_calibrator(raw, y, method: str):
    raw, y = np.asarray(raw, float), np.asarray(y, float)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    if method == "sigmoid":
        lr = LogisticRegression().fit(raw.reshape(-1, 1), y)
        return _SigmoidCalibrator(lr)
    raise ValueError(method)


class _SigmoidCalibrator:
    def __init__(self, lr):
        self.lr = lr

    def predict(self, raw):
        return self.lr.predict_proba(np.asarray(raw, float).reshape(-1, 1))[:, 1]


def _score_machines(ds, X_cols, train_mask, score_idx, model, cfg):
    est = make_classifier(model, _params(cfg, model), cfg["random_state"], balanced=False)
    est.fit(ds[X_cols][train_mask.to_numpy()], ds["fail_within_h"][train_mask].astype(int))
    return est, positive_proba(est, ds[X_cols].iloc[score_idx])


def run_warning(data: ModelData, cfg: dict, model: str, target_far: float | None = None) -> TaskResult:
    ds, X_cols = data.ds, data.features
    ew = cfg["modelling"]["early_warning"]
    target_far = ew["target_false_alarm_rate"] if target_far is None else target_far
    horizon, persistence = float(cfg["targets"]["warning_horizon_hours"]), int(ew["persistence_h"])
    scored, active, machine = warning_rows(ds), ds["eligible"], ds["machine_id"]
    tmask = warning_train_rows(ds)                   # fit + calibration rows (no early-degradation hours)
    early = early_degradation(ds).to_numpy()

    raw = np.full(len(ds), np.nan)       # outer out-of-fold raw scores, every row of every machine
    prob = np.full(len(ds), np.nan)
    alarm = np.zeros(len(ds), bool)
    fold_info = []
    if model == "rules":
        rb = _rule_baseline(cfg).predict(ds)
        raw[:] = prob[:] = rb["alarm"].astype(float).to_numpy()
        alarm = rb["alarm"].to_numpy()
        fold_info = [{"fold": f.fold, "test_machine": f.test_machine, "threshold_raw": np.nan, "threshold_prob": np.nan,
                      "inner_far": np.nan} for _, f in data.folds.iterrows()]
    else:
        for _, f in data.folds.iterrows():
            m = f.test_machine
            train_machines = [x for x in data.machines if x != m]
            inner = np.full(len(ds), np.nan)
            for j in train_machines:                                        # inner LOMO
                idx_j = np.flatnonzero(machine.eq(j))
                _, inner[idx_j] = _score_machines(ds, X_cols, tmask & machine.ne(j) & machine.ne(m), idx_j, model, cfg)
            in_tr = machine.isin(train_machines).to_numpy()
            cal_rows = in_tr & tmask.to_numpy()
            cal = fit_calibrator(inner[cal_rows], ds["fail_within_h"].to_numpy(float)[cal_rows], ew["calibration"])
            sub = np.flatnonzero(in_tr)
            thr = threshold_for_far(inner[sub], ds["fail_within_h"].to_numpy(float)[sub], machine.to_numpy()[sub],
                                    active.to_numpy()[sub], scored.to_numpy()[sub], target_far, persistence, early[sub])
            idx_m = np.flatnonzero(machine.eq(m))
            _, raw[idx_m] = _score_machines(ds, X_cols, tmask & machine.ne(m), idx_m, model, cfg)
            prob[idx_m] = cal.predict(raw[idx_m])
            alarm[idx_m] = alarms_from_scores(raw[idx_m], machine.to_numpy()[idx_m], active.to_numpy()[idx_m], thr, persistence)
            inner_alarm = alarms_from_scores(inner[sub], machine.to_numpy()[sub], active.to_numpy()[sub], thr, persistence)
            neg = scored.to_numpy()[sub] & (ds["fail_within_h"].to_numpy(float)[sub] == 0) & ~early[sub]
            fold_info.append({"fold": f.fold, "test_machine": m, "threshold_raw": thr,
                              "threshold_prob": float(cal.predict([thr])[0]) if np.isfinite(thr) else np.nan,
                              "inner_far": float(inner_alarm[neg].mean())})

    frame = ds[["machine_id", "timestamp", "fold"]].assign(y=ds["fail_within_h"].astype(float).to_numpy(), scored=scored.to_numpy(),
                                                          early=early, alarm=alarm, score=prob, raw_score=raw)
    summary, lt = warning_metrics(frame, data.events, horizon)
    fold_rows = []
    for fi in fold_info:
        fr = frame[frame["machine_id"] == fi["test_machine"]]
        mfold, _ = warning_metrics(fr, data.events, horizon)
        fold_rows.append({**fi, "far": mfold["false_alarm_rate"], "recall": mfold["recall"], "precision": mfold["precision"],
                          "lead_time_h": mfold["lead_time_median_h"] if mfold["events_total"] else np.nan})
    per_fold = pd.DataFrame(fold_rows)
    summary["target_false_alarm_rate"] = target_far
    lt_by_mode = lt.groupby("failure_mode")["lead_time_h"].median() if len(lt) else pd.Series(dtype=float)
    summary.update({f"lead_time_median_h_{k}": float(v) for k, v in lt_by_mode.items()})

    final = {}
    if model != "rules":
        # Deployed operating point: calibrator and threshold from the outer out-of-fold scores of all machines
        sc = tmask.to_numpy()
        final["calibrator"] = fit_calibrator(raw[sc], ds["fail_within_h"].to_numpy(float)[sc], ew["calibration"])
        final["threshold_raw"] = threshold_for_far(raw, ds["fail_within_h"].to_numpy(float), machine.to_numpy(), active.to_numpy(),
                                                   scored.to_numpy(), target_far, persistence, early)
        final["threshold_prob"] = float(final["calibrator"].predict([final["threshold_raw"]])[0])
        final["persistence_h"] = persistence
    oof = frame[frame["scored"]].reset_index(drop=True)
    return TaskResult("early_warning", model, oof, per_fold, summary,
                      {"per_fold": per_fold, "lead_times": lt, "scores_all_hours": frame}, final)


def far_sweep(data: ModelData, cfg: dict, result: TaskResult, targets=(0.0, 0.001, 0.002, 0.005, 0.01, 0.02)) -> pd.DataFrame:
    """Lead time vs false-alarm rate on the pooled out-of-fold scores (the trade-off curve).

    The threshold here is set on the pooled out-of-fold scores themselves, so this curve is
    descriptive; the operating-point numbers in the summary use thresholds chosen per fold.
    """
    ds = data.ds
    frame = result.tables["scores_all_hours"]
    assert (frame["timestamp"].to_numpy() == ds["timestamp"].to_numpy()).all()
    raw = np.nan_to_num(frame["raw_score"].to_numpy(float), nan=-np.inf)
    persistence = int(cfg["modelling"]["early_warning"]["persistence_h"])
    rows = []
    for t in targets:
        thr = threshold_for_far(raw, frame["y"], frame["machine_id"], ds["eligible"], frame["scored"], t, persistence, frame["early"])
        f = frame.assign(alarm=alarms_from_scores(raw, frame["machine_id"], ds["eligible"], thr, persistence))
        m, _ = warning_metrics(f, data.events, float(cfg["targets"]["warning_horizon_hours"]))
        rows.append({"target_far": t, "realised_far": m["false_alarm_rate"], "recall": m["recall"], "precision": m["precision"],
                     "events_detected": m["events_detected"], "lead_time_median_h": m["lead_time_median_h"]})
    return pd.DataFrame(rows)
