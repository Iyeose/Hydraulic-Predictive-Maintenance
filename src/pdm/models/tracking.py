"""MLflow experiment tracking.

Every Stage 3 run records enough to reproduce it exactly:

* **parameters**: task, model family, hyperparameters, number of features, operating-point settings
* **metrics**: the pooled out-of-fold summary, plus each fold's metric as a step series
  (``fold_<metric>`` with step = fold number), so per-machine variation is visible in the UI
* **tags**: git commit / branch / dirty flag, config hash, dataset fingerprint, data label
* **artifacts**: feature list, config snapshot, out-of-fold predictions, per-fold and confusion
  tables, figures

A training invocation is one parent run with one nested run per (task, model).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import yaml

from ..config import PROJECT_ROOT
from .cv import TaskResult
from .data import ModelData


def tracking_uri(cfg: dict) -> str:
    uri = cfg["modelling"]["tracking"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(Path(cfg['root']) / uri[len('sqlite:///'):]).resolve()}"
    return uri


def setup_tracking(cfg: dict) -> str:
    """Point MLflow at the project store and return the experiment id (created on first use)."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri(cfg))
    t = cfg["modelling"]["tracking"]
    exp = mlflow.get_experiment_by_name(t["experiment"])
    if exp is None:
        art = (Path(cfg["root"]) / t["artifact_dir"]).resolve()
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(t["experiment"], artifact_location=art.as_uri())
    return mlflow.set_experiment(t["experiment"]).experiment_id


def git_info(root: str | Path) -> dict[str, str]:
    def g(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    return {"git_commit": g("rev-parse", "HEAD"), "git_branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": str(bool(g("status", "--porcelain", "--untracked-files=no")))}


def config_hash(cfg: dict) -> str:
    """Hash of every setting (paths and the machine-specific root excluded)."""
    body = {k: v for k, v in cfg.items() if k not in ("root", "paths")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_tags(cfg: dict, data: ModelData, data_label: str) -> dict[str, str]:
    # The commit of the code that ran (the pdm package's repository), not of the data folder
    return {**git_info(PROJECT_ROOT), "config_hash": config_hash(cfg), "dataset_hash": data.dataset_hash,
            "data_label": data_label, "stage": "3", "validation": "leave_one_machine_out"}


def _finite(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


@contextmanager
def parent_run(cfg: dict, data: ModelData, data_label: str, run_name: str):
    import mlflow
    setup_tracking(cfg)
    with mlflow.start_run(run_name=run_name, tags=run_tags(cfg, data, data_label)) as run:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "feature_list.json").write_text(json.dumps(data.features, indent=2))
            (tmp / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k not in ("root", "paths")}, sort_keys=False))
            data.folds.to_csv(tmp / "folds.csv", index=False)
            mlflow.log_artifacts(str(tmp), "inputs")
        mlflow.log_params({"n_features": len(data.features), "n_rows": len(data.ds), "n_eligible": int(data.ds["eligible"].sum())})
        yield run


def log_task_result(result: TaskResult, cfg: dict, data: ModelData, data_label: str, figures: dict | None = None,
                    extra_params: dict | None = None) -> str:
    """Log one (task, model) evaluation as a nested run. Returns the run id."""
    import mlflow
    params = {"task": result.task, "model": result.model, "n_features": len(data.features),
              **{f"hp_{k}": v for k, v in cfg["modelling"]["models"].get(result.model, {}).items()}, **(extra_params or {})}
    if result.task == "early_warning":
        ew = cfg["modelling"]["early_warning"]
        params.update({"calibration": ew["calibration"], "persistence_h": ew["persistence_h"],
                       "target_false_alarm_rate": result.summary.get("target_false_alarm_rate")})
    with mlflow.start_run(run_name=f"{result.task}/{result.model}", nested=True,
                          tags={**run_tags(cfg, data, data_label), "task": result.task, "model_family": result.model}) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(_finite(result.summary))
        for _, row in result.per_fold.iterrows():
            for k, v in _finite(row.drop(labels=["fold"], errors="ignore").to_dict()).items():
                mlflow.log_metric(f"fold_{k}", v, step=int(row["fold"]))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result.oof.to_parquet(tmp / "oof_predictions.parquet", index=False)
            for name, t in result.tables.items():
                if name == "scores_all_hours":
                    continue
                t.to_csv(tmp / f"{name}.csv", index=isinstance(t.index, pd.Index) and t.index.name is not None)
            (tmp / "summary.json").write_text(json.dumps(result.summary, indent=2, default=str))
            mlflow.log_artifacts(str(tmp), "evaluation")
        for name, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"figures/{name}.png")
        return run.info.run_id
