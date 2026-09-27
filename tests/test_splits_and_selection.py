import numpy as np
import pandas as pd

from pdm.features.selection import identity_leak_scores
from pdm.features.splits import assign_folds, leave_one_machine_out


def test_each_machine_is_test_exactly_once():
    fail = pd.DataFrame({"machine_id": ["A", "B", "C"], "failure_mode": ["x", "y", "y"]})
    folds = leave_one_machine_out(["A", "B", "C", "D"], fail)
    assert sorted(folds["test_machine"]) == ["A", "B", "C", "D"]
    assert folds.set_index("test_machine").at["A", "unseen_mode_in_train"] == "x"   # singleton class
    assert folds.set_index("test_machine").at["B", "unseen_mode_in_train"] == ""
    rows = pd.DataFrame({"machine_id": ["A", "D", "D"]})
    assert list(assign_folds(rows, folds)) == [0, 3, 3]


def test_identity_leak_check_flags_a_machine_fingerprint():
    rng = np.random.default_rng(0)
    T = pd.date_range("2024-01-01", "2024-02-10", freq="1h")
    rows = []
    for k in range(10):
        rows.append(pd.DataFrame({"machine_id": f"M{k}", "timestamp": T, "failure_mode_label": "healthy",
                                  "standby_share": 0.0, "fingerprint": k + rng.normal(0, 0.05, len(T)),
                                  "honest": rng.normal(0, 1, len(T))}))
    ds = pd.concat(rows, ignore_index=True)
    s = identity_leak_scores(ds, ["fingerprint", "honest"], "2024-01-21")
    assert s["fingerprint"] > 0.9 and s["honest"] < 0.2
