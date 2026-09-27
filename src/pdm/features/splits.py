"""Validation splits, fixed once and saved so every model in Stage 3 is scored the same way.

Why leave-one-machine-out (LOMO)? Minute and hour rows from the same machine are strongly
autocorrelated, and each machine has exactly one failure mode plus its own set points (EDA §6.2,
§9.2). A random row split would put near-identical neighbouring hours in train and test, and
would let a model identify the machine instead of the fault. LOMO asks the question that
matters in production: *does the model work on a machine it has never seen?*

Known limitation: cylinder drift occurs on one machine only (HPU_06). In the HPU_06 fold the
model has never seen that class in training. That fold measures detection ("something is
wrong"), not classification of the mode, and Stage 3 must report it separately.
"""
from __future__ import annotations

import pandas as pd


def leave_one_machine_out(machines, failures: pd.DataFrame) -> pd.DataFrame:
    machines = sorted(pd.unique(pd.Series(list(machines))))
    modes = failures.groupby("machine_id")["failure_mode"].agg(lambda s: ", ".join(sorted(set(s))))
    rows = []
    for k, test_m in enumerate(machines):
        train_modes = set(failures.loc[failures["machine_id"] != test_m, "failure_mode"])
        test_modes = set(failures.loc[failures["machine_id"] == test_m, "failure_mode"])
        rows.append({"fold": k, "test_machine": test_m, "test_failure_modes": modes.get(test_m, "none (healthy reference)"),
                     "unseen_mode_in_train": ", ".join(sorted(test_modes - train_modes)) or ""})
    return pd.DataFrame(rows)


def assign_folds(rows: pd.DataFrame, folds: pd.DataFrame) -> pd.Series:
    """Fold number of each row = the fold in which its machine is the test machine."""
    return rows["machine_id"].map(folds.set_index("test_machine")["fold"]).astype("int16")


def time_holdout_mask(rows: pd.DataFrame, start: str) -> pd.Series:
    """True for rows in the time hold-out period (secondary, forward-in-time check)."""
    return rows["timestamp"] >= pd.Timestamp(start)
