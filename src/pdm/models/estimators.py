"""Model factories. Every estimator exposes the scikit-learn API (fit / predict / predict_proba).

Class imbalance (≈ 80 % healthy hours, 192 cylinder-drift hours) is handled with balanced class
weights, so the rare modes are not ignored. Gradient-boosting models handle the NaNs left at the
start of rolling windows natively. The linear baselines impute the median and standardise.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .baselines import ConstantRegressor

CLASSIFIERS = ("logreg", "lightgbm", "xgboost")
REGRESSORS = ("constant", "ridge", "lightgbm", "xgboost")


class XGBLabelled:
    """XGBoost classifier that accepts string labels and (optionally) balanced sample weights."""

    def __init__(self, balanced: bool = True, **params):
        self.balanced, self.params = balanced, params

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.le_ = LabelEncoder().fit(y)
        yi = self.le_.transform(y)
        obj = "binary:logistic" if len(self.le_.classes_) == 2 else "multi:softprob"
        self.model_ = XGBClassifier(objective=obj, tree_method="hist", n_jobs=-1, verbosity=0, **self.params)
        self.model_.fit(X, yi, sample_weight=compute_sample_weight("balanced", yi) if self.balanced else None)
        self.classes_ = self.le_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_classifier(name: str, params: dict, seed: int, balanced: bool = True):
    """``balanced=False`` for the early-warning model: its scores are calibrated probabilities and
    reweighting would push them towards 1, which makes the alarm threshold hard to resolve."""
    cw = "balanced" if balanced else None
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(class_weight=cw, random_state=seed, **params))
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(class_weight=cw, random_state=seed, n_jobs=-1, verbose=-1, **params)
    if name == "xgboost":
        return XGBLabelled(balanced=balanced, random_state=seed, **params)
    raise ValueError(f"unknown classifier {name!r}")


def make_regressor(name: str, params: dict, seed: int):
    if name == "constant":
        return ConstantRegressor()
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(**params))
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, tree_method="hist", n_jobs=-1, verbosity=0, **params)
    raise ValueError(f"unknown regressor {name!r}")


def positive_proba(model, X) -> np.ndarray:
    """Probability of the positive class (label 1) for a binary classifier."""
    p = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    return p[:, classes.index(1)] if 1 in classes else p[:, -1]
