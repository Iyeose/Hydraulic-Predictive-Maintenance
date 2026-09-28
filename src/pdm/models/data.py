"""Loading the Stage 2 artefacts and defining which rows each task is trained and scored on.

Stage 3 never rebuilds features. It reads exactly what Stage 2 persisted, so every model is
trained on the same table, the same screened feature list and the same leave-one-machine-out folds.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.selection import META

# Target and bookkeeping columns. None of these may ever be a model input.
TARGET_COLUMNS = ["failure_mode_label", "rul_hours", "rul_censored", "fail_within_h", "hours_to_failure",
                  "in_repair", "life_cycle", "next_event_id", "is_degrading", "eligible", "fold"]


@dataclass
class ModelData:
    ds: pd.DataFrame                 # full hourly table (all rows, all columns), sorted by machine and time
    features: list[str]              # screened feature list from Stage 2
    folds: pd.DataFrame              # fold → test machine
    events: pd.DataFrame             # one row per failure event (derived from the targets)
    dataset_hash: str

    @property
    def machines(self) -> list[str]:
        return sorted(self.ds["machine_id"].unique())


def dataset_fingerprint(ds: pd.DataFrame, features: list[str]) -> str:
    """Stable short hash of the model-relevant content (rows, targets, feature values and list)."""
    h = hashlib.sha256()
    cols = ["machine_id", "timestamp", "failure_mode_label", "rul_hours", "fail_within_h", "fold", "eligible"] + features
    h.update(pd.util.hash_pandas_object(ds[cols], index=False).values.tobytes())
    h.update(json.dumps(features).encode())
    return h.hexdigest()[:12]


def events_from_targets(ds: pd.DataFrame) -> pd.DataFrame:
    """Failure events reconstructed from the target columns (no need to re-read the raw label file).

    failure time = T + hours_to_failure; degradation start = first labelled hour − 1 h (the row
    stamped T describes [T − 1 h, T)).
    """
    d = ds[ds["next_event_id"].notna() & ds["hours_to_failure"].notna()]
    ev = (d.assign(failure_timestamp=d["timestamp"] + pd.to_timedelta(d["hours_to_failure"], unit="h"))
            .groupby("next_event_id").agg(machine_id=("machine_id", "first"), failure_timestamp=("failure_timestamp", "first")))
    lab = ds[ds["is_degrading"]].groupby("next_event_id").agg(failure_mode=("failure_mode_label", "first"),
                                                              first_labelled=("timestamp", "min"))
    ev = ev.join(lab, how="inner")
    ev["degradation_start_timestamp"] = ev["first_labelled"] - pd.Timedelta("1h")
    return ev.drop(columns="first_labelled").rename_axis("event_id").reset_index().sort_values("failure_timestamp").reset_index(drop=True)


def load_model_data(cfg: dict) -> ModelData:
    P = cfg["paths"]["processed_dir"]
    ds = pd.read_parquet(P / "dataset.parquet").sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    features = json.loads((P / "feature_list.json").read_text())
    folds = pd.read_csv(P / "folds.csv")
    check_features(features)
    return ModelData(ds=ds, features=features, folds=folds, events=events_from_targets(ds),
                     dataset_hash=dataset_fingerprint(ds, features))


def check_features(features: list[str], cfg: dict | None = None) -> None:
    """Refuse to train on anything that is a label, a target, or an excluded identity carrier."""
    bad = [f for f in features if f in META or f in TARGET_COLUMNS]
    if cfg is not None:
        bad += [f for f in features if f in cfg["leakage_columns"] or f.startswith(tuple(cfg["features"]["exclude_from_model"]))]
    if bad:
        raise ValueError(f"leakage: target / excluded columns in the feature list: {sorted(set(bad))}")


# --- task row masks -------------------------------------------------------------------------------
def classification_rows(ds: pd.DataFrame) -> pd.Series:
    return ds["eligible"] & ds["failure_mode_label"].ne("repair")


def rul_rows(ds: pd.DataFrame) -> pd.Series:
    """Rows with an exactly known (possibly capped) RUL. Censored rows are never used."""
    return ds["eligible"] & ds["rul_hours"].notna()


def warning_rows(ds: pd.DataFrame) -> pd.Series:
    return ds["eligible"] & ds["fail_within_h"].notna()


def fold_split(ds: pd.DataFrame, mask: pd.Series, test_machine: str) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of train (other machines) and test (held-out machine) within ``mask``."""
    is_test = ds["machine_id"].eq(test_machine).to_numpy()
    m = mask.to_numpy()
    tr, te = np.flatnonzero(m & ~is_test), np.flatnonzero(m & is_test)
    assert not set(ds["machine_id"].iloc[tr]) & set(ds["machine_id"].iloc[te]), "machine in both train and test"
    return tr, te
