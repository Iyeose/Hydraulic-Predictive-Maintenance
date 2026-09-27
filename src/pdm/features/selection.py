"""Feature screening: excluded-by-config → mostly-missing → constant → identity leak → redundancy.

The **identity-leak check** is the important one. With one failure mode per machine, any feature
that reveals *which machine* a row came from also reveals the label without saying anything
about machine health (EDA §9.2). For every candidate we train a small decision tree to predict
``machine_id`` from that feature alone, using healthy hours only. Training uses hours before
``split_date`` and testing uses hours after it. Chance accuracy is 1/10. A feature that does
much better than chance is excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

META = {"machine_id", "timestamp", "regime", "in_baseline_period", "n_minutes", "fold",
        "failure_mode_label", "rul_hours", "rul_censored", "fail_within_h", "hours_to_failure",
        "in_repair", "life_cycle", "next_event_id", "is_degrading", "hour_of_day"}
GROUP_PRIORITY = ["fleet_relative", "hourly_regime", "cross_sensor", "vibration", "rolling", "lag", "spectral",
                  "context", "quality", "maintenance", "hourly_raw"]


def eligible_rows(ds: pd.DataFrame) -> pd.Series:
    """Rows usable for training/evaluation: not in repair, not in the baseline period, not standby."""
    return (~ds["in_repair"]) & (~ds["in_baseline_period"]) & (ds["standby_share"] < 0.5)


def identity_leak_scores(ds: pd.DataFrame, features: list[str], split_date: str, seed: int = 42) -> pd.Series:
    healthy = ds[(ds["failure_mode_label"] == "healthy") & (ds["standby_share"] < 0.5)]
    tr, te = healthy[healthy["timestamp"] < split_date], healthy[healthy["timestamp"] >= split_date]
    scores = {}
    for f in features:
        clf = DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=20, random_state=seed)
        clf.fit(tr[[f]].fillna(-999), tr["machine_id"])
        scores[f] = float((clf.predict(te[[f]].fillna(-999)) == te["machine_id"]).mean())
    return pd.Series(scores, name="machine_id_accuracy")


def identity_chance_level(ds: pd.DataFrame, split_date: str) -> float:
    """Accuracy of always guessing the most frequent machine in the test period (the no-information baseline)."""
    healthy = ds[(ds["failure_mode_label"] == "healthy") & (ds["standby_share"] < 0.5)]
    te = healthy[healthy["timestamp"] >= split_date]
    return float(te["machine_id"].value_counts(normalize=True).max())


def multi_feature_identity_accuracy(ds: pd.DataFrame, features: list[str], split_date: str, seed: int = 42) -> float:
    healthy = ds[(ds["failure_mode_label"] == "healthy") & (ds["standby_share"] < 0.5)]
    tr, te = healthy[healthy["timestamp"] < split_date], healthy[healthy["timestamp"] >= split_date]
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    clf.fit(tr[features], tr["machine_id"])
    return float((clf.predict(te[features]) == te["machine_id"]).mean())


def select_features(ds: pd.DataFrame, dictionary: pd.DataFrame, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    fc, lc = cfg["features"], cfg["leakage_check"]
    groups = dictionary.set_index("feature")["group"]
    cands = [c for c in ds.columns if c not in META and c in groups.index and pd.api.types.is_numeric_dtype(ds[c])]
    rep = pd.DataFrame(index=cands)
    rep["group"] = groups.reindex(cands)
    rep["status"] = "kept"
    rep["detail"] = ""

    excl = tuple(fc["exclude_from_model"])
    rep.loc[[c for c in cands if c.startswith(excl)], "status"] = "excluded_by_config"
    rep.loc[rep["group"].isin(fc.get("exclude_groups", [])), "status"] = "excluded_by_config"

    elig = ds[eligible_rows(ds)]
    rep["nan_share"] = elig[cands].isna().mean()
    live = rep["status"] == "kept"
    rep.loc[live & (rep["nan_share"] > 0.5), "status"] = "dropped_mostly_missing"
    live = rep["status"] == "kept"
    std = elig[cands].std()
    rep.loc[live & (std.reindex(rep.index).fillna(0) < 1e-9), "status"] = "dropped_constant"

    # Identity leak: scored for every candidate (including config exclusions) for transparency
    rep["machine_id_accuracy"] = identity_leak_scores(ds, cands, lc["split_date"], cfg["random_state"])
    live = rep["status"] == "kept"
    leak = live & (rep["machine_id_accuracy"] > lc["single_feature_flag_accuracy"])
    rep.loc[leak, "status"] = "dropped_identity_leak"

    # Redundancy: greedy in group-priority order, drop if |r| > threshold with an already-kept feature
    live_feats = rep.index[rep["status"] == "kept"].tolist()
    order = sorted(live_feats, key=lambda f: (GROUP_PRIORITY.index(rep.at[f, "group"]) if rep.at[f, "group"] in GROUP_PRIORITY else 99,
                                              live_feats.index(f)))
    corr = elig[order].corr().abs()
    kept: list[str] = []
    for f in order:
        hit = next((k for k in kept if corr.at[f, k] > fc["corr_prune_threshold"]), None)
        if hit:
            rep.at[f, "status"], rep.at[f, "detail"] = "dropped_correlated", f"|r| > {fc['corr_prune_threshold']} with {hit}"
        else:
            kept.append(f)
    rep.index.name = "feature"
    return kept, rep.reset_index()
