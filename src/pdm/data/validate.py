"""Validation gate: enforce the data contract, quarantine bad records, and produce a report.

Three outcomes for any check:

* **reject**: the batch is structurally unusable (for example a required column is missing),
  so ``BatchRejected`` is raised and nothing downstream runs.
* **quarantine**: individual rows break a hard rule. They are removed and written to
  ``data/quarantine`` together with the reason.
* **warn**: a soft rule such as a sensor range or a sampling gap. Rows are kept and the
  cleaning step deals with them. The count goes into the report.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa

from .schemas import SCHEMAS


class BatchRejected(RuntimeError):
    """Raised when a table cannot be validated at all (schema-level failure)."""


@dataclass
class CheckResult:
    table: str
    check: str
    severity: str          # "error" | "warning" | "info"
    n_failed: int
    action: str
    detail: str = ""


@dataclass
class ValidationReport:
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rows_in: dict = field(default_factory=dict)
    rows_out: dict = field(default_factory=dict)
    results: list[CheckResult] = field(default_factory=list)

    def add(self, *args, **kwargs):
        self.results.append(CheckResult(*args, **kwargs))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.results])

    @property
    def passed(self) -> bool:
        return not any(r.severity == "error" and r.action == "reject" for r in self.results)

    def to_markdown(self) -> str:
        lines = [f"# Data validation report\n", f"Generated (UTC): {self.created_utc}\n",
                 "| Table | Rows in | Rows out |", "|---|---:|---:|"]
        lines += [f"| {t} | {self.rows_in[t]:,} | {self.rows_out.get(t, 0):,} |" for t in self.rows_in]
        lines += ["", "| Table | Check | Severity | Failed | Action | Detail |", "|---|---|---|---:|---|---|"]
        lines += [f"| {r.table} | {r.check} | {r.severity} | {r.n_failed:,} | {r.action} | {r.detail} |" for r in self.results]
        return "\n".join(lines) + "\n"

    def save(self, reports_dir):
        (reports_dir / "validation_report.md").write_text(self.to_markdown(), encoding="utf-8")
        (reports_dir / "validation_report.json").write_text(
            json.dumps({**asdict(self), "passed": self.passed}, indent=2, default=str), encoding="utf-8")


def _apply_schema(name: str, df: pd.DataFrame, schema: pa.DataFrameSchema, report: ValidationReport):
    """Validate lazily; split into (valid rows, quarantined rows with reasons)."""
    try:
        return schema.validate(df, lazy=True), pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        # Failures without a row index are schema-level (missing column, wrong dtype that cannot be coerced)
        schema_level = fc[fc["index"].isna()]
        if len(schema_level):
            for _, r in schema_level.iterrows():
                report.add(name, f"{r['column']}: {r['check']}", "error", 0, "reject", str(r["failure_case"]))
            raise BatchRejected(f"{name}: {len(schema_level)} schema-level failure(s)") from err
        bad_idx = fc["index"].astype(int).unique()
        reasons = (fc.assign(reason=fc["column"].astype(str) + ": " + fc["check"].astype(str))
                     .groupby("index")["reason"].agg("; ".join))
        for (col, chk), g in fc.groupby(["column", "check"], dropna=False):
            report.add(name, f"{col}: {chk}", "error", int(g["index"].nunique()), "quarantine")
        quarantined = df.loc[bad_idx].assign(quarantine_reason=reasons.reindex(bad_idx).values)
        return schema.validate(df.drop(index=bad_idx), lazy=True), quarantined


def validate_telemetry(df: pd.DataFrame, cfg: dict, report: ValidationReport):
    name = "telemetry"
    report.rows_in[name] = len(df)
    required = {"machine_id", "timestamp", *cfg["sensors"]}
    missing = required - set(df.columns)
    if missing:
        report.add(name, "required columns present", "error", len(missing), "reject", ", ".join(sorted(missing)))
        raise BatchRejected(f"telemetry missing columns: {sorted(missing)}")

    # 1) Duplicate keys: keep the first reading, quarantine the rest
    dup = df.duplicated(["machine_id", "timestamp"], keep="first")
    quarantine = [df[dup].assign(quarantine_reason="duplicate (machine_id, timestamp)")]
    report.add(name, "unique (machine_id, timestamp)", "error" if dup.any() else "info", int(dup.sum()),
               "quarantine" if dup.any() else "pass")
    df = df[~dup]

    # 2) Contract schema (types, categories, key format)
    valid, bad = _apply_schema(name, df, SCHEMAS[name](cfg), report)
    if not len(bad):
        report.add(name, "contract schema", "info", 0, "pass")
    quarantine.append(bad)

    # 3) Soft checks, reported only (handled by the cleaning step)
    for s, (lo, hi) in cfg["cleaning"]["valid_ranges"].items():
        n = int(((valid[s] < lo) | (valid[s] > hi)).sum())
        report.add(name, f"{s} within [{lo}, {hi}]", "warning" if n else "info", n,
                   "set NaN in cleaning" if n else "pass")
    for s, floor in cfg["validation"].get("sensor_floor", {}).items():
        at_floor = int((valid[s] == floor).sum())
        report.add(name, f"{s} readings exactly at sensor floor {floor}", "warning" if at_floor else "info", at_floor,
                   "keep; exclude floor-driven minima from features", f"{at_floor / max(len(valid), 1):.1%} of rows")
    nan_rows = valid[cfg["sensors"]].isna().any(axis=1)
    mismatch = int((nan_rows != valid["is_sensor_dropout"].astype(bool)).sum())
    report.add(name, "missing sensors ⇔ dropout flag", "warning" if mismatch else "info", mismatch,
               "investigate" if mismatch else "pass", f"{int(nan_rows.sum()):,} dropout rows")
    step = pd.Timedelta(cfg["validation"]["sampling_interval"])
    diffs = valid.groupby("machine_id")["timestamp"].diff().dropna()
    gaps = int((diffs > step).sum())
    off_grid = int((valid["timestamp"].dt.second != 0).sum())
    report.add(name, f"regular {cfg['validation']['sampling_interval']} grid", "warning" if gaps else "info", gaps,
               "reindex to grid in cleaning" if gaps else "pass", f"{off_grid} off-grid timestamps")
    leaks = [c for c in cfg["leakage_columns"] if c in valid.columns]
    report.add(name, "label columns present in telemetry", "info", len(leaks), "excluded from features", ", ".join(leaks))

    report.rows_out[name] = len(valid)
    return valid, pd.concat([q for q in quarantine if len(q)], ignore_index=True) if any(len(q) for q in quarantine) else pd.DataFrame()


def validate_table(name: str, df: pd.DataFrame, cfg: dict, report: ValidationReport):
    report.rows_in[name] = len(df)
    valid, bad = _apply_schema(name, df, SCHEMAS[name](cfg), report)
    if not len(bad):
        report.add(name, "contract schema", "info", 0, "pass")
    report.rows_out[name] = len(valid)
    return valid, bad


def validate_all(tables: dict[str, pd.DataFrame], cfg: dict, write: bool = True):
    """Validate every table plus cross-table rules. Returns (valid tables, report)."""
    report = ValidationReport()
    out, quarantine = {}, {}
    out["telemetry"], quarantine["telemetry"] = validate_telemetry(tables["telemetry"], cfg, report)
    for name in ("failures", "maintenance"):
        out[name], quarantine[name] = validate_table(name, tables[name], cfg, report)

    # Cross-table referential integrity: events must refer to machines we have telemetry for
    known = set(out["telemetry"]["machine_id"].unique())
    for name in ("failures", "maintenance"):
        orphan = ~out[name]["machine_id"].isin(known)
        report.add(name, "machine_id exists in telemetry", "error" if orphan.any() else "info", int(orphan.sum()),
                   "quarantine" if orphan.any() else "pass")
        if orphan.any():
            quarantine[name] = pd.concat([quarantine[name], out[name][orphan].assign(quarantine_reason="unknown machine_id")])
            out[name] = out[name][~orphan]
            report.rows_out[name] = len(out[name])

    # Coverage: do the event tables overlap the telemetry period? (EDA §10.3: maintenance does not)
    t0, t1 = out["telemetry"]["timestamp"].min(), out["telemetry"]["timestamp"].max()
    for name, col in (("failures", "failure_timestamp"), ("maintenance", "action_timestamp")):
        inside = out[name][col].between(t0, t1 + pd.Timedelta("1D"))
        report.add(name, "events inside telemetry period", "warning" if (~inside).any() else "info",
                   int((~inside).sum()), "report only", f"{int(inside.sum())} of {len(inside)} inside {t0.date()}–{t1.date()}")

    if write:
        for name, q in quarantine.items():
            if len(q):
                q.to_parquet(cfg["paths"]["quarantine_dir"] / f"{name}_quarantine.parquet", index=False)
        for name, df in out.items():
            df.to_parquet(cfg["paths"]["interim_dir"] / f"validated_{name}.parquet", index=False)
        report.save(cfg["paths"]["reports_dir"])
    return out, report, quarantine
