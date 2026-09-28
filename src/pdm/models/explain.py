"""SHAP explanations: which features drive a prediction, globally and for a single alert.

For the tree models SHAP values are exact (TreeExplainer) and in log-odds units for classifiers
(raw prediction units for RUL). The early-warning calibrator is monotone, so the features that
push the log-odds up are exactly the ones that push the calibrated probability up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tree_model(estimator):
    """The object SHAP should see (unwrap our XGBoost label wrapper)."""
    return getattr(estimator, "model_", estimator)


def shap_values(estimator, X: pd.DataFrame) -> np.ndarray:
    """SHAP values with shape (rows, features) or (rows, features, classes)."""
    import shap
    if hasattr(estimator, "steps"):          # linear pipeline: exact linear SHAP on the standardised inputs
        imp, sc, lin = (s for _, s in estimator.steps)
        Z = sc.transform(imp.transform(X))
        coef = lin.coef_                       # (classes or 1, features)
        vals = Z[:, :, None] * coef.T[None, :, :]
        return vals[:, :, 0] if vals.shape[2] == 1 else vals
    ex = shap.TreeExplainer(_tree_model(estimator))
    v = ex(X, check_additivity=False).values
    if isinstance(v, list):
        v = np.stack(v, axis=-1)
    return v


def global_importance(estimator, X: pd.DataFrame, class_names: list[str] | None = None) -> pd.DataFrame:
    """Mean |SHAP| per feature (one column per class for multiclass models), sorted by the total."""
    v = shap_values(estimator, X)
    if v.ndim == 3:
        names = class_names if class_names is not None else [str(c) for c in estimator.classes_]
        df = pd.DataFrame(np.abs(v).mean(axis=0), index=X.columns, columns=names)
    else:
        df = pd.DataFrame({"mean_abs_shap": np.abs(v).mean(axis=0)}, index=X.columns)
    df["total"] = df.sum(axis=1)
    return df.sort_values("total", ascending=False).rename_axis("feature")


def explain_rows(estimator, X: pd.DataFrame, top_k: int = 8, class_index: int | None = None,
                 dictionary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top contributing features for each row (for multiclass, towards ``class_index``).

    Returns a long table: row, rank, feature, value, shap, description. This is what an engineer
    sees next to an alert: *why* did it fire?
    """
    v = shap_values(estimator, X)
    if v.ndim == 3:
        v = v[:, :, class_index if class_index is not None else 0]
    desc = dictionary.set_index("feature")["description"] if dictionary is not None else None
    rows = []
    for i, idx in enumerate(X.index):
        order = np.argsort(-np.abs(v[i]))[:top_k]
        for r, j in enumerate(order, 1):
            f = X.columns[j]
            rows.append({"row": idx, "rank": r, "feature": f, "value": float(X.iat[i, j]), "shap": float(v[i, j]),
                         "description": desc.get(f, "") if desc is not None else ""})
    return pd.DataFrame(rows)
