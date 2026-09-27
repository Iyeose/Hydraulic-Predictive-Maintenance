import pandas as pd
import pytest

from conftest import make_failures, make_maintenance, make_telemetry
from pdm.data.ingest import read_maintenance
from pdm.data.validate import BatchRejected, validate_all


def _tables(tel=None, fail=None, mnt=None):
    return {"telemetry": make_telemetry(days=2) if tel is None else tel,
            "failures": make_failures() if fail is None else fail,
            "maintenance": make_maintenance() if mnt is None else mnt}


def test_clean_batch_passes(cfg):
    out, report, quarantine = validate_all(_tables(), cfg, write=False)
    assert report.passed
    assert all(len(q) == 0 for q in quarantine.values())
    assert len(out["telemetry"]) == 2 * 2 * 1440


def test_duplicate_keys_are_quarantined(cfg):
    tel = make_telemetry(days=1)
    tel = pd.concat([tel, tel.iloc[[5]]], ignore_index=True)
    out, _, q = validate_all(_tables(tel=tel), cfg, write=False)
    assert len(q["telemetry"]) == 1
    assert "duplicate" in q["telemetry"]["quarantine_reason"].iat[0]
    assert not out["telemetry"].duplicated(["machine_id", "timestamp"]).any()


def test_bad_category_rows_are_quarantined_not_rejected(cfg):
    tel = make_telemetry(days=1)
    tel.loc[10, "shift"] = "Lunch"
    out, report, q = validate_all(_tables(tel=tel), cfg, write=False)
    assert report.passed and len(q["telemetry"]) == 1
    assert "shift" in q["telemetry"]["quarantine_reason"].iat[0]


def test_missing_sensor_column_rejects_batch(cfg):
    tel = make_telemetry(days=1).drop(columns="flow_lpm")
    with pytest.raises(BatchRejected):
        validate_all(_tables(tel=tel), cfg, write=False)


def test_out_of_range_values_are_warned_and_kept(cfg):
    tel = make_telemetry(days=1)
    tel.loc[3, "pressure_bar"] = 550.0
    out, report, _ = validate_all(_tables(tel=tel), cfg, write=False)
    res = report.to_frame()
    row = res[res["check"].str.startswith("pressure_bar within")].iloc[0]
    assert row["severity"] == "warning" and row["n_failed"] == 1
    assert len(out["telemetry"]) == len(tel)


def test_failure_events_rules(cfg):
    fail = pd.concat([make_failures()] * 3, ignore_index=True)
    fail["event_id"] = ["F_1", "F_2", "F_3"]
    fail.loc[1, "failure_mode"] = "gremlins"
    fail.loc[2, "degradation_start_timestamp"] = fail.loc[2, "failure_timestamp"] + pd.Timedelta("1D")
    out, _, q = validate_all(_tables(fail=fail), cfg, write=False)
    assert list(out["failures"]["event_id"]) == ["F_1"]
    assert len(q["failures"]) == 2


def test_unknown_machine_in_events_is_quarantined(cfg):
    mnt = make_maintenance()
    mnt.loc[1, "machine_id"] = "HPU_99"
    out, _, q = validate_all(_tables(mnt=mnt), cfg, write=False)
    assert "HPU_99" not in set(out["maintenance"]["machine_id"])
    assert len(q["maintenance"]) == 1


def test_literal_none_component_survives_csv_round_trip(tmp_path):
    path = tmp_path / "m.csv"
    make_maintenance().rename(columns={"action_id": "maintenance_id"}).to_csv(path, index=False)
    df = read_maintenance(path, {"maintenance_id": "action_id"})
    assert (df["component_replaced"] == "None").sum() == 1
