"""Baselines every model must beat.

* :class:`RuleBaseline`: the transparent condition monitor from EDA §8.3, evaluated at hourly
  grain on the Stage 2 regime z-scores. It has no fitted parameters (thresholds were set by
  engineering inspection), so leave-one-machine-out does not change it. It gives an alarm
  (early-warning baseline) and a failure-mode guess from which rule fired (classification baseline).
* Constant RUL: predict the training mean, the floor for any RUL model.

Logistic / ridge regression baselines are built in :mod:`pdm.models.estimators`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sustained


class RuleBaseline:
    def __init__(self, rules: dict, window_h: int = 6, persistence_h: int = 6, standby_share_max: float = 0.5):
        self.rules, self.window_h, self.persistence_h, self.standby_share_max = rules, window_h, persistence_h, standby_share_max

    def rule_flags(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Boolean flag per rule and row. ``ds`` must be sorted by machine and time (full hourly grid)."""
        g = ds.groupby("machine_id", sort=False)
        active = (ds["standby_share"] < self.standby_share_max) & ~ds["in_repair"]
        out = pd.DataFrame(index=ds.index)
        for name, r in self.rules.items():
            roll = g[r["signal"]].rolling(self.window_h, min_periods=self.window_h)
            if r["direction"] == 0:
                sig = roll.mean().abs()
            else:
                sig = roll.median() * r["direction"]
            sig = sig.reset_index(level=0, drop=True).reindex(ds.index)
            out[name] = sustained((sig > r["threshold"]).fillna(False) & active, ds["machine_id"], self.persistence_h)
        return out

    def predict(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Returns ``alarm`` (any rule), ``mode`` (first matching rule's mode, else healthy) and the flags."""
        flags = self.rule_flags(ds)
        mode = pd.Series("healthy", index=ds.index, dtype=object)
        for name in reversed(list(self.rules)):          # earlier rules take precedence
            mode[flags[name].to_numpy()] = self.rules[name]["mode"]
        return flags.assign(alarm=flags.any(axis=1), mode=mode)


class ConstantRegressor:
    """Predicts the training mean. The 'no-information' RUL baseline."""

    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)
