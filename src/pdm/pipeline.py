"""End-to-end Stage 2 pipeline: raw files → validated → cleaned → features + targets.

Run from the project root:

    python -m pdm.pipeline --config configs/config.yaml

Outputs (data/processed):
    dataset.parquet           hourly features + targets + fold, one row per machine-hour
    feature_list.json         the model feature set after screening
    feature_dictionary.csv    every candidate feature, its group, description and screening outcome
    folds.csv                 leave-one-machine-out fold definitions
    regime_baseline.parquet   per-machine healthy baselines (needed again at inference time)
Reports (reports/): validation_report.md/json, pipeline_summary.json
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .config import load_config
from .data.clean import clean_telemetry
from .data.ingest import ingest
from .data.validate import validate_all
from .features.baseline import RegimeBaseline
from .features.build import build_features
from .features.selection import eligible_rows, identity_chance_level, multi_feature_identity_accuracy, select_features
from .features.splits import assign_folds, leave_one_machine_out
from .features.targets import build_targets


def run(cfg: dict, verbose: bool = True) -> dict:
    t0 = time.time()
    log = lambda m: verbose and print(f"[{time.time() - t0:6.1f}s] {m}")  # noqa: E731
    P = cfg["paths"]

    tables = ingest(cfg); log("ingested raw files")
    valid, report, _ = validate_all(tables, cfg); log(f"validated — passed={report.passed}")
    silver, clean_log = clean_telemetry(valid["telemetry"], cfg); log("cleaned telemetry")

    baseline = RegimeBaseline(cfg["baseline"]["signals"], cfg["baseline"]["reference_days"],
                              cfg["baseline"]["min_samples_per_cell"]).fit(silver)
    baseline.save(P["processed_dir"] / "regime_baseline.parquet")
    silver_z = baseline.transform(silver); log("fitted regime baselines")

    feats, dictionary = build_features(silver_z, valid["maintenance"], cfg); log(f"built {feats.shape[1]} columns")
    data_end = silver.groupby("machine_id")["timestamp"].max() + pd.Timedelta(cfg["validation"]["sampling_interval"])
    targets = build_targets(feats[["machine_id", "timestamp"]], valid["failures"], cfg, data_end=data_end)
    ds = feats.merge(targets, on=["machine_id", "timestamp"], how="left")

    folds = leave_one_machine_out(ds["machine_id"].unique(), valid["failures"])
    ds["fold"] = assign_folds(ds, folds)
    ds["eligible"] = eligible_rows(ds)

    selected, screening = select_features(ds, dictionary, cfg); log(f"selected {len(selected)} features")
    dictionary = dictionary.merge(screening.drop(columns="group"), on="feature", how="left")
    lc = cfg["leakage_check"]
    id_acc_raw = multi_feature_identity_accuracy(ds, [c for c in dictionary.query("group == 'hourly_raw'")["feature"]], lc["split_date"])
    id_acc_sel = multi_feature_identity_accuracy(ds, selected, lc["split_date"])

    ds.to_parquet(P["processed_dir"] / "dataset.parquet", index=False)
    dictionary.to_csv(P["processed_dir"] / "feature_dictionary.csv", index=False)
    folds.to_csv(P["processed_dir"] / "folds.csv", index=False)
    (P["processed_dir"] / "feature_list.json").write_text(json.dumps(selected, indent=2))

    summary = {
        "validation_passed": report.passed,
        "cleaning": clean_log,
        "dataset_rows": len(ds), "eligible_rows": int(ds["eligible"].sum()),
        "candidate_features": int(len(dictionary)), "selected_features": len(selected),
        "screening": dictionary["status"].value_counts().to_dict(),
        "identity_accuracy_chance_level": round(identity_chance_level(ds, lc["split_date"]), 3),
        "identity_accuracy_raw_features": round(id_acc_raw, 3),
        "identity_accuracy_selected_features": round(id_acc_sel, 3),
        "label_counts_eligible": ds.loc[ds["eligible"], "failure_mode_label"].value_counts().to_dict(),
        "rul_known_eligible": int(ds.loc[ds["eligible"], "rul_hours"].notna().sum()),
        "runtime_s": round(time.time() - t0, 1),
    }
    (P["reports_dir"] / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("done")
    return {"dataset": ds, "features": selected, "dictionary": dictionary, "folds": folds,
            "report": report, "summary": summary, "silver": silver_z, "baseline": baseline, "tables": valid}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2 pipeline: validate, clean, build features and targets")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    res = run(load_config(args.config))
    print(json.dumps(res["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
