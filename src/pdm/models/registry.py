"""Packaging trained models for serving and registering them in the MLflow Model Registry.

Each task is wrapped in one ``mlflow.pyfunc`` model that carries everything inference needs
besides the features themselves: the fitted estimator, the exact feature list and order, the
class labels, and for the early-warning model the probability calibrator and alarm threshold.
Stage 5/6 can then load any of them the same way::

    model = mlflow.pyfunc.load_model("models:/hpu-early-warning@champion")
    model.predict(hourly_features)        # DataFrame with the Stage 2 feature columns

The alarm **persistence** rule (N consecutive hours above the threshold) needs the machine's
recent history, so the service applies it with :func:`pdm.models.metrics.sustained` on the
``above_threshold`` column; the model itself scores one hour at a time.
"""
from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from mlflow.pyfunc import PythonModel
except ImportError:          # mlflow is optional for code that only evaluates models
    PythonModel = object

PDM_PACKAGE_DIR = Path(__file__).resolve().parents[1]


class PdMModel(PythonModel):
    """A fitted estimator plus the metadata needed to use it correctly."""

    def __init__(self, task: str, estimator, features: list[str], classes: list[str] | None = None,
                 calibrator=None, threshold_raw: float | None = None, threshold_prob: float | None = None,
                 persistence_h: int | None = None, rul_cap: float | None = None, horizon_h: float | None = None):
        self.task, self.estimator, self.features = task, estimator, list(features)
        self.classes, self.calibrator = classes, calibrator
        self.threshold_raw, self.threshold_prob, self.persistence_h = threshold_raw, threshold_prob, persistence_h
        self.rul_cap, self.horizon_h = rul_cap, horizon_h

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"{len(missing)} model features missing from input, e.g. {missing[:5]}")
        return df[self.features].astype(float)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:  # noqa: ARG002
        X = self._X(model_input)
        if self.task == "failure_mode":
            p = self.estimator.predict_proba(X)
            proba = pd.DataFrame(0.0, index=model_input.index, columns=self.classes)
            proba[list(self.estimator.classes_)] = p
            out = proba.add_prefix("p_")
            out.insert(0, "failure_mode_pred", np.asarray(self.classes, object)[proba.to_numpy().argmax(axis=1)])
            return out
        if self.task == "rul":
            return pd.DataFrame({"rul_hours_pred": np.clip(self.estimator.predict(X), 0, self.rul_cap)}, index=model_input.index)
        if self.task == "early_warning":
            from .estimators import positive_proba
            raw = positive_proba(self.estimator, X)
            return pd.DataFrame({"p_fail_within_horizon": self.calibrator.predict(raw), "raw_score": raw,
                                 "above_threshold": raw >= self.threshold_raw}, index=model_input.index)
        raise ValueError(self.task)


def _requirements() -> list[str]:
    reqs = []
    for pkg in ("mlflow", "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"):
        try:
            reqs.append(f"{pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    return reqs


def log_and_register(model: PdMModel, example: pd.DataFrame, registered_name: str | None, alias: str | None,
                     tags: dict | None = None) -> dict:
    """Log the pyfunc model in the active run; optionally register it and point ``alias`` at it."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    X = example[model.features].astype(float)
    sig = infer_signature(X, model.predict(None, X))
    info = mlflow.pyfunc.log_model(name="model", python_model=model, code_paths=[str(PDM_PACKAGE_DIR)],
                                   signature=sig, input_example=X.head(3), pip_requirements=_requirements(),
                                   registered_model_name=registered_name)
    out = {"model_uri": info.model_uri, "version": None}
    if registered_name:
        client = MlflowClient()
        version = info.registered_model_version
        for k, v in (tags or {}).items():
            client.set_model_version_tag(registered_name, version, k, str(v))
        if alias:
            client.set_registered_model_alias(registered_name, alias, version)
        out["version"] = version
    return out
