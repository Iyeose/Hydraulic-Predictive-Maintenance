"""Data contract as executable pandera schemas.

These are *hard* rules. A row that breaks one is invalid as a record (wrong key, impossible
category, bad type) and is quarantined. Physically implausible sensor *values* are soft rules:
they are handled by the cleaning step (see ``clean.py``), because the rest of the row is still
valuable.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


def telemetry_schema(cfg: dict) -> DataFrameSchema:
    v = cfg["validation"]
    sensor = lambda: Column(float, nullable=True, coerce=True)  # noqa: E731  (NaN allowed: dropouts)
    return DataFrameSchema(
        {
            "machine_id": Column(str, Check.str_matches(v["machine_id_pattern"]), nullable=False, coerce=True),
            "timestamp": Column("datetime64[ns]", nullable=False, coerce=True),
            **{s: sensor() for s in cfg["sensors"]},
            "is_sensor_dropout": Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
            "shift": Column(str, Check.isin(v["shifts"]), nullable=False, coerce=True),
            "day_of_week": Column(int, Check.in_range(0, 6), nullable=False, coerce=True),
        },
        unique=["machine_id", "timestamp"],
        strict=False,          # extra (label) columns are allowed but never used as inputs
        name="Sensor_Telemetry",
    )


def failures_schema(cfg: dict) -> DataFrameSchema:
    v = cfg["validation"]
    return DataFrameSchema(
        {
            "event_id": Column(str, unique=True, nullable=False, coerce=True),
            "machine_id": Column(str, Check.str_matches(v["machine_id_pattern"]), coerce=True),
            "failure_timestamp": Column("datetime64[ns]", nullable=False, coerce=True),
            "degradation_start_timestamp": Column("datetime64[ns]", nullable=False, coerce=True),
            "failure_mode": Column(str, Check.isin(v["failure_modes"]), coerce=True),
            "downtime_hours": Column(float, Check.gt(0), coerce=True),
            "repair_cost_usd": Column(float, Check.ge(0), nullable=True, coerce=True, required=False),
        },
        checks=[Check(lambda d: d["degradation_start_timestamp"] < d["failure_timestamp"],
                      element_wise=False, error="degradation must start before failure")],
        strict=False,
        name="Failure_Events",
    )


def maintenance_schema(cfg: dict) -> DataFrameSchema:
    v = cfg["validation"]
    return DataFrameSchema(
        {
            "action_id": Column(str, unique=True, nullable=False, coerce=True),
            "machine_id": Column(str, Check.str_matches(v["machine_id_pattern"]), coerce=True),
            "action_timestamp": Column("datetime64[ns]", nullable=False, coerce=True),
            "action_type": Column(str, Check.isin(v["action_types"]), coerce=True),
            "component_replaced": Column(str, Check.isin(v["components"]), coerce=True),
            "cost_usd": Column(float, Check.ge(0), nullable=True, coerce=True, required=False),
        },
        strict=False,
        name="Maintenance_Actions",
    )


SCHEMAS = {"telemetry": telemetry_schema, "failures": failures_schema, "maintenance": maintenance_schema}
__all__ = ["SCHEMAS", "telemetry_schema", "failures_schema", "maintenance_schema", "pa"]
