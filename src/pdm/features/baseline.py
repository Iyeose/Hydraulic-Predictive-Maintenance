"""Per-machine healthy baseline, conditioned on operating regime (EDA §8.1).

"Normal" is learned for each machine × day type (Weekday/Weekend) × hour of day from that
machine's own first ``reference_days`` of data (a commissioning window), with standby
minutes excluded. Readings are then expressed as

* ``z_<signal>``: (value − baseline median) / baseline std. Used for thresholds and models.
* ``pct_<signal>``: % deviation from the baseline median. Easier for engineers to read.

This removes the Day/Night/Weekend set points, the midnight dip and machine-to-machine
offsets, including the rpm offset that identifies each machine. Standby minutes get NaN,
because there is no meaningful health assessment while the pump idles.

Leakage note: the baseline uses only each machine's *own earliest* data. A new machine needs
``reference_days`` of history before it can be scored (the cold-start requirement).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["machine_id", "day_type", "hour"]


@dataclass
class RegimeBaseline:
    signals: list[str]
    reference_days: int = 14
    min_samples: int = 20
    table_: pd.DataFrame | None = None
    reference_end_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "RegimeBaseline":
        start = df.groupby("machine_id")["timestamp"].transform("min")
        self.reference_end_ = df.groupby("machine_id")["timestamp"].min() + pd.Timedelta(days=self.reference_days)
        ref = df[(df["timestamp"] < start + pd.Timedelta(days=self.reference_days)) & ~df["is_standby"]]

        cell = ref.groupby(KEYS)[self.signals].agg(["median", "std", "count"])
        # Fallback for sparse cells: the machine × day type level (all hours pooled)
        coarse = ref.groupby(["machine_id", "day_type"])[self.signals].agg(["median", "std"])
        rows = []
        for s in self.signals:
            t = cell[s].copy()
            sparse = t["count"] < self.min_samples
            if sparse.any():
                fb = coarse[s].reindex(t.index.droplevel("hour"))
                t.loc[sparse, ["median", "std"]] = fb.loc[sparse.values].to_numpy()
            # Floor the std so tiny-variance cells cannot produce huge z-scores
            floor = t["std"].median() * 0.25
            t["std"] = t["std"].clip(lower=floor)
            t.columns = [f"{s}__{c}" for c in t.columns]
            rows.append(t)
        self.table_ = pd.concat(rows, axis=1)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.table_ is None:
            raise RuntimeError("fit() the baseline first")
        out = df.copy()
        idx = pd.MultiIndex.from_frame(out[KEYS])
        tab = self.table_.reindex(idx)
        for s in self.signals:
            med = tab[f"{s}__median"].to_numpy()
            std = tab[f"{s}__std"].to_numpy()
            z = (out[s].to_numpy() - med) / std
            pct = 100.0 * (out[s].to_numpy() / med - 1.0)
            standby = out["is_standby"].to_numpy()
            out[f"z_{s}"] = np.where(standby, np.nan, z)
            out[f"pct_{s}"] = np.where(standby, np.nan, pct)
        out["in_baseline_period"] = out["timestamp"] < out["machine_id"].map(self.reference_end_)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # Persisted with the model so inference uses exactly the same "normal"
    def save(self, path: Path) -> None:
        self.table_.reset_index().to_parquet(path, index=False)
        self.reference_end_.rename("reference_end").reset_index().to_parquet(Path(path).with_suffix(".ref.parquet"), index=False)

    @classmethod
    def load(cls, path: Path, signals: list[str]) -> "RegimeBaseline":
        b = cls(signals=signals)
        b.table_ = pd.read_parquet(path).set_index(KEYS)
        b.reference_end_ = pd.read_parquet(Path(path).with_suffix(".ref.parquet")).set_index("machine_id")["reference_end"]
        return b
