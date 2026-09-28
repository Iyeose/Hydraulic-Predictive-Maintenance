"""Stage 3 tests: metrics, alarm logic, rule baseline, leave-one-machine-out hygiene, early-warning
threshold selection, and the MLflow tracking / registry round trip. Synthetic and fast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.models import cv
from pdm.models.baselines import RuleBaseline
from pdm.models.data import ModelData, check_features, dataset_fingerprint, events_from_targets, fold_split
from pdm.models.metrics import (alarms_from_scores, false_alarm_rate, lead_times, nasa_score, sustained,
                                threshold_for_far)

FEATS = ["f_pressure", "f_temp", "f_noise"]


def make_hourly(n_hours=320, fail_at=260, window=110, seed=0) -> ModelData:
    """Four machines on an hourly grid, one failure each (two pump wear, two valve leakage)."""
    rng = np.random.default_rng(seed)
    modes = {"HPU_01": "pump_wear", "HPU_02": "valve_leakage", "HPU_03": "pump_wear", "HPU_04": "valve_leakage"}
    t = pd.date_range("2024-01-01 01:00", periods=n_hours, freq="1h")
    parts = []
    for k, (m, mode) in enumerate(modes.items()):
        h = np.arange(n_hours)
        htf = np.where(h <= fail_at, fail_at - h, np.nan).astype(float)
        deg = (h > fail_at - window) & (h <= fail_at)
        sev = np.where(deg, (h - (fail_at - window)) / window, 0.0)
        repair = (h > fail_at) & (h <= fail_at + 5)
        label = np.where(repair, "repair", np.where(deg, mode, "healthy"))
        rul = np.where(h <= fail_at, np.minimum(htf, 336.0), np.nan)
        warn = np.where(h <= fail_at, (htf <= 168).astype(float), np.nan)
        rul[repair] = np.nan; warn[repair] = np.nan
        parts.append(pd.DataFrame({
            "machine_id": m, "timestamp": t,
            "f_pressure": -4 * sev * (mode == "pump_wear") + rng.normal(0, 0.3, n_hours),
            "f_temp": 4 * sev * (mode == "valve_leakage") + rng.normal(0, 0.3, n_hours),
            "f_noise": rng.normal(0, 1, n_hours),
            # rule signals
            "z_pressure_bar_mean": -5 * sev * (mode == "pump_wear") + rng.normal(0, 0.2, n_hours),
            "z_temp_celsius_mean": 5 * sev * (mode == "valve_leakage") + rng.normal(0, 0.2, n_hours),
            "failure_mode_label": label, "rul_hours": rul, "fail_within_h": warn, "hours_to_failure": htf,
            "next_event_id": np.where(h <= fail_at, f"F_{k}", None), "is_degrading": deg,
            "in_repair": repair, "standby_share": 0.0, "eligible": ~repair & (h >= 24), "fold": k,
            "rul_censored": np.isnan(rul),
        }))
    ds = pd.concat(parts, ignore_index=True)
    folds = pd.DataFrame({"fold": range(4), "test_machine": list(modes), "test_failure_modes": list(modes.values()),
                          "unseen_mode_in_train": [np.nan] * 4})
    return ModelData(ds=ds, features=FEATS, folds=folds, events=events_from_targets(ds), dataset_hash=dataset_fingerprint(ds, FEATS))


@pytest.fixture(scope="module")
def mdata():
    return make_hourly()


@pytest.fixture
def mcfg(cfg):
    c = dict(cfg)
    c["modelling"] = {**cfg["modelling"],
                      "classes": ["healthy", "pump_wear", "valve_leakage"],
                      "rule_baseline": {"window_h": 3, "persistence_h": 3, "rules": {
                          "temp_high": {"signal": "z_temp_celsius_mean", "direction": 1, "threshold": 3.0, "mode": "valve_leakage"},
                          "pressure_low": {"signal": "z_pressure_bar_mean", "direction": -1, "threshold": 3.0, "mode": "pump_wear"}}},
                      "models": {**cfg["modelling"]["models"], "logreg": {"C": 1.0, "max_iter": 500}}}
    return c


# --- metrics ---------------------------------------------------------------------------------------
def test_nasa_score_is_zero_when_perfect_and_penalises_late_more():
    y = np.array([100.0, 50.0])
    assert nasa_score(y, y) == 0.0
    late, early = nasa_score([100.0], [148.0]), nasa_score([100.0], [52.0])
    assert late > early > 0


def test_sustained_needs_consecutive_hours_within_each_machine():
    flag = [1, 1, 0, 1, 1, 1, 1, 1]
    grp = ["A"] * 4 + ["B"] * 4
    # A: never 3 in a row; B: rows 5-7 are ≥ 3 in a row; A's trailing 1 must not chain into B
    assert sustained(flag, grp, 3).tolist() == [False] * 6 + [True, True]


def test_threshold_for_far_is_the_lowest_threshold_meeting_the_target():
    rng = np.random.default_rng(1)
    scores = rng.random(1000)
    y = (scores > 0.9).astype(float)                     # positives score high
    y[rng.choice(1000, 50, replace=False)] = 0           # plus some high-scoring negatives
    grp, act = np.repeat(["A", "B"], 500), np.ones(1000, bool)
    thr = threshold_for_far(scores, y, grp, act, act, target=0.01, persistence=1)
    far = lambda t: false_alarm_rate(alarms_from_scores(scores, grp, act, t, 1), y, act)  # noqa: E731
    assert far(thr) <= 0.01
    below = scores[scores < thr].max()
    assert far(below) > 0.01                             # any lower observed score would break the target


def test_lead_time_ignores_alarms_before_the_warning_window():
    ev = pd.DataFrame({"event_id": ["F"], "machine_id": ["A"], "failure_mode": ["pump_wear"],
                       "failure_timestamp": [pd.Timestamp("2024-01-20")], "degradation_start_timestamp": [pd.Timestamp("2024-01-15")]})
    ts = pd.date_range("2024-01-01", "2024-01-20", freq="1h")
    alarm = (ts == pd.Timestamp("2024-01-05")) | (ts >= pd.Timestamp("2024-01-18"))   # early false alarm + real one
    lt = lead_times(pd.DataFrame({"machine_id": "A", "timestamp": ts, "alarm": alarm}), ev, horizon_h=168)
    assert lt.loc[0, "lead_time_h"] == 48.0 and lt.loc[0, "window_h"] == 168.0


# --- baselines and data guards ---------------------------------------------------------------------
def test_rule_baseline_alarms_after_window_and_persistence_and_maps_mode(mdata, mcfg):
    rb = mcfg["modelling"]["rule_baseline"]
    pred = RuleBaseline(rb["rules"], rb["window_h"], rb["persistence_h"]).predict(mdata.ds)
    a = pred.loc[mdata.ds.machine_id == "HPU_01"]
    assert not a["alarm"].iloc[:150].any()                        # healthy period: no alarms
    assert a["alarm"].iloc[240:261].all()                         # late degradation: alarm
    assert set(a.loc[a["alarm"], "mode"]) == {"pump_wear"}
    b = pred.loc[mdata.ds.machine_id == "HPU_02"]
    assert set(b.loc[b["alarm"], "mode"]) == {"valve_leakage"}


def test_rule_baseline_is_silent_during_standby(mdata, mcfg):
    ds = mdata.ds.copy()
    ds["standby_share"] = 1.0
    rb = mcfg["modelling"]["rule_baseline"]
    assert not RuleBaseline(rb["rules"], rb["window_h"], rb["persistence_h"]).predict(ds)["alarm"].any()


def test_check_features_rejects_targets_and_identity_carriers(cfg):
    with pytest.raises(ValueError):
        check_features(["f_ok", "rul_hours"])
    with pytest.raises(ValueError):
        check_features(["f_ok", "pump_rpm_mean"], cfg)
    check_features(["f_ok"], cfg)


def test_fold_split_never_shares_a_machine(mdata):
    for m in mdata.machines:
        tr, te = fold_split(mdata.ds, mdata.ds["eligible"], m)
        assert set(mdata.ds.machine_id.iloc[te]) == {m} and m not in set(mdata.ds.machine_id.iloc[tr])


def test_events_reconstructed_from_targets(mdata):
    ev = mdata.events
    assert len(ev) == 4 and set(ev.failure_mode) == {"pump_wear", "valve_leakage"}
    assert (ev.failure_timestamp - ev.degradation_start_timestamp == pd.Timedelta(hours=110)).all()


# --- leave-one-machine-out runners -----------------------------------------------------------------
def test_classification_oof_covers_each_row_once_and_learns(mdata, mcfg):
    r = cv.run_classification(mdata, mcfg, "logreg")
    assert not r.oof.duplicated(["machine_id", "timestamp"]).any()
    assert len(r.oof) == int((mdata.ds.eligible & (mdata.ds.failure_mode_label != "repair")).sum())
    assert r.summary["macro_f1_seen"] > 0.6 and r.summary["events_mode_correct"] >= 3


def test_rul_predictions_are_clipped_to_the_cap(mdata, mcfg):
    r = cv.run_rul(mdata, mcfg, "ridge")
    assert r.oof["y_pred"].between(0, mcfg["targets"]["rul_cap_hours"]).all()
    assert r.summary["mae_critical"] < cv.run_rul(mdata, mcfg, "constant").summary["mae_critical"]


def test_warning_threshold_is_chosen_without_the_test_machine(mdata, mcfg, monkeypatch):
    seen = []
    orig = cv._score_machines

    def spy(ds, X_cols, train_mask, score_idx, model, cfg):
        seen.append((set(ds.loc[train_mask.to_numpy(), "machine_id"]), set(ds["machine_id"].iloc[score_idx])))
        return orig(ds, X_cols, train_mask, score_idx, model, cfg)

    monkeypatch.setattr(cv, "_score_machines", spy)
    r = cv.run_warning(mdata, mcfg, "logreg", target_far=0.01)
    assert all(not (tr & te) for tr, te in seen)                  # never trained on a machine it scores
    assert len(seen) == 4 * 3 + 4                                 # 3 inner fits + 1 outer fit per fold
    assert (r.per_fold["inner_far"] <= 0.01 + 1e-12).all()
    assert {"calibrator", "threshold_raw", "threshold_prob"} <= set(r.final)
    assert 0 <= r.oof["score"].min() and r.oof["score"].max() <= 1


# --- tracking and registry -------------------------------------------------------------------------
def test_mlflow_tracking_and_registry_round_trip(mdata, mcfg, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from pdm.models.estimators import make_classifier
    from pdm.models.registry import PdMModel, log_and_register
    from pdm.models.tracking import log_task_result, parent_run

    c = dict(mcfg)
    c["root"] = tmp_path
    c["modelling"] = {**mcfg["modelling"], "tracking": {"tracking_uri": "sqlite:///mlflow.db", "artifact_dir": "mlartifacts",
                                                         "experiment": "test"}}
    r = cv.run_classification(mdata, c, "logreg")
    ds, X = mdata.ds, mdata.features
    rows = ds["eligible"] & (ds["failure_mode_label"] != "repair")
    with parent_run(c, mdata, "synthetic", "test-parent") as parent:
        run_id = log_task_result(r, c, mdata, "synthetic")
        with mlflow.start_run(run_name="final", nested=True):
            est = make_classifier("logreg", c["modelling"]["models"]["logreg"], 0).fit(ds.loc[rows, X], ds.loc[rows, "failure_mode_label"])
            model = PdMModel("failure_mode", est, X, classes=c["modelling"]["classes"])
            out = log_and_register(model, ds.loc[rows], "test-model", "champion", tags={"passes_gate": True})
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["dataset_hash"] == mdata.dataset_hash and run.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert "macro_f1_seen" in run.data.metrics and len(client.get_metric_history(run_id, "fold_macro_f1")) == 4
    loaded = mlflow.pyfunc.load_model("models:/test-model@champion")
    x = ds.loc[rows].head(20)
    pd.testing.assert_frame_equal(loaded.predict(x).reset_index(drop=True), model.predict(None, x).reset_index(drop=True))
    assert out["version"] == "1" or str(out["version"]) == "1"


def test_synthetic_generator_never_overwrites_real_files(tmp_path):
    from pdm.synthetic import write_raw
    (tmp_path / "sensor_telemetry.csv").write_text("real data")
    with pytest.raises(FileExistsError):
        write_raw(tmp_path)
    assert (tmp_path / "sensor_telemetry.csv").read_text() == "real data"
