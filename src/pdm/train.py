"""Stage 3 training CLI: evaluate baselines and models leave-one-machine-out, track everything in
MLflow, then fit the selected model per task on all machines, explain it and register it.

Run from the project root after the Stage 2 pipeline (``python -m pdm.pipeline``)::

    python -m pdm.train                       # all tasks, all candidate models
    python -m pdm.train --tasks failure_mode --models lightgbm
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Outputs: MLflow runs (``mlflow.db`` + ``mlartifacts/``), registered models, and
``reports/stage3_model_comparison.{json,md}`` plus figures in ``reports/figures/``.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .models import cv
from .models.data import check_features, classification_rows, load_model_data, rul_rows, warning_rows

TASKS = ("failure_mode", "rul", "early_warning")
CANDIDATES = {
    "failure_mode": {"baselines": ["rules", "logreg"], "models": ["lightgbm", "xgboost"]},
    "rul": {"baselines": ["constant", "ridge"], "models": ["lightgbm", "xgboost"]},
    "early_warning": {"baselines": ["rules", "logreg"], "models": ["lightgbm", "xgboost"]},
}
# Primary metric per task and whether higher is better
PRIMARY = {"failure_mode": ("macro_f1_seen", True), "rul": ("mae_critical", False), "early_warning": ("lead_time_mean_h", True)}
RUNNERS = {"failure_mode": cv.run_classification, "rul": cv.run_rul, "early_warning": cv.run_warning}


def _better(a: float, b: float, higher: bool) -> bool:
    return (a > b) if higher else (a < b)


def select_and_gate(task: str, results: dict[str, cv.TaskResult], baselines: list[str], far_budget: float | None = None) -> dict:
    """Pick the best non-baseline model on the primary metric and check it beats every baseline.

    Early warning: a model only qualifies if its realised false-alarm rate is within ``far_budget``,
    so a longer lead time can never be bought with more false alarms than the baseline accepts.
    """
    metric, higher = PRIMARY[task]
    cands = {n: r for n, r in results.items() if n.split("@")[0] not in baselines}
    if task == "early_warning" and far_budget is not None:
        ok = {n: r for n, r in cands.items() if r.summary["false_alarm_rate"] <= far_budget + 1e-12}
        cands = ok or cands
    if not cands:
        return {"selected": None}
    best = max(cands, key=lambda n: cands[n].summary[metric] * (1 if higher else -1))
    v = cands[best].summary[metric]
    beats = {b: bool(_better(v, results[b].summary[metric], higher)) for b in baselines if b in results}
    within_far = (far_budget is None) or (cands[best].summary.get("false_alarm_rate", 0) <= far_budget + 1e-12)
    return {"selected": best, "primary_metric": metric, "value": v, "higher_is_better": higher,
            "beats_baselines": beats, "within_false_alarm_budget": within_far,
            "passes_gate": bool(beats) and all(beats.values()) and within_far}


def _figures(task: str, results: dict[str, cv.TaskResult], data, cfg) -> dict:
    from .models import plots
    figs = {}
    if task == "failure_mode":
        for n, r in results.items():
            figs[f"{task}_{n}_confusion"] = plots.confusion_heatmap(
                r.tables["confusion_all"], f"{n}: out-of-fold confusion (all folds; HPU_06 fold cannot predict cylinder drift)")
        figs[f"{task}_per_fold"] = plots.per_fold_bars({n: r.per_fold for n, r in results.items()}, "macro_f1",
                                                       "Failure mode: macro-F1 per held-out machine", "macro-F1")
    elif task == "rul":
        figs[f"{task}_per_fold"] = plots.per_fold_bars({n: r.per_fold for n, r in results.items()}, "mae_critical",
                                                       "RUL: MAE in the last 14 days before failure, per held-out machine", "MAE (h)")
        machines = [m for m in ["HPU_01", "HPU_03", "HPU_06"] if m in data.machines][:3]
        figs[f"{task}_traces"] = plots.rul_traces({n: r.oof for n, r in results.items()}, machines,
                                                  float(cfg["targets"]["rul_cap_hours"]), "RUL: out-of-fold predictions")
    elif task == "early_warning":
        figs[f"{task}_lead_times"] = plots.lead_time_bars({n: r.tables["lead_times"] for n, r in results.items()},
                                                          "Early warning: lead time per failure event (out-of-fold)")
    return figs


def run(cfg: dict, tasks=TASKS, models: list[str] | None = None, data_label: str = "real", register: bool = True,
        verbose: bool = True) -> dict:
    import matplotlib.pyplot as plt
    import mlflow

    from .models import explain, plots
    from .models.registry import PdMModel, log_and_register
    from .models.tracking import log_task_result, parent_run, run_tags
    from .models.estimators import make_classifier, make_regressor

    t0 = time.time()
    log = lambda m: verbose and print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    data = load_model_data(cfg)
    check_features(data.features, cfg)
    ew = cfg["modelling"]["early_warning"]
    fig_dir = cfg["paths"]["reports_dir"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = {"data_label": data_label, "dataset_hash": data.dataset_hash, "n_features": len(data.features), "tasks": {}}

    with parent_run(cfg, data, data_label, run_name=f"stage3-train-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        report["mlflow_parent_run_id"] = parent.info.run_id
        for task in tasks:
            cand = CANDIDATES[task]
            names = cand["baselines"] + [m for m in cand["models"] if models is None or m in models]
            results: dict[str, cv.TaskResult] = {}
            for n in names:
                results[n] = RUNNERS[task](data, cfg, n)
                log(f"{task:<14} {n:<9} " + ", ".join(f"{k}={v:.3f}" for k, v in results[n].summary.items()
                                                       if k in (PRIMARY[task][0], "macro_f1_all", "rmse_critical", "nasa_critical",
                                                                "false_alarm_rate", "events_detected", "lead_time_median_h")))
            far_budget = None
            if task == "early_warning":
                rules_far = results["rules"].summary["false_alarm_rate"] if "rules" in results else None
                far_budget = max(ew["target_false_alarm_rate"], rules_far or 0.0)
                # If the rules raise fewer false alarms than our target, also compare at the rules' own rate
                if rules_far is not None and rules_far < ew["target_false_alarm_rate"]:
                    far_budget = rules_far
                    for n in [m for m in list(results) if m not in cand["baselines"]]:
                        key = f"{n}@rules_far"
                        results[key] = cv.run_warning(data, cfg, n, target_far=rules_far)
                        log(f"{task:<14} {key}: FAR={results[key].summary['false_alarm_rate']:.4f}, "
                            f"lead={results[key].summary['lead_time_mean_h']:.1f} h")
            gate = select_and_gate(task, results, cand["baselines"], far_budget)
            figs = _figures(task, results, data, cfg)

            run_ids = {}
            for n, r in results.items():
                own = {k: v for k, v in figs.items() if k.startswith(f"{task}_{n}_")}
                run_ids[n] = log_task_result(r, cfg, data, data_label, figures=own)
            for k, f in figs.items():
                mlflow.log_figure(f, f"figures/{k}.png")
                f.savefig(fig_dir / f"stage3_{k}.png", bbox_inches="tight", dpi=130)
                plt.close(f)

            sweeps = {}
            if task == "early_warning":
                for n in [m for m in results if m not in cand["baselines"] and "@" not in m] + (["logreg"] if "logreg" in results else []):
                    sweeps[n] = cv.far_sweep(data, cfg, results[n])
                f = plots.far_curve(sweeps, results["rules"].summary if "rules" in results else None,
                                    "Lead time vs false alarms (pooled out-of-fold)")
                mlflow.log_figure(f, "figures/early_warning_far_curve.png")
                f.savefig(fig_dir / "stage3_early_warning_far_curve.png", bbox_inches="tight", dpi=130)
                plt.close(f)

            entry = {"gate": gate, "models": {n: {"summary": r.summary, "mlflow_run_id": run_ids[n]} for n, r in results.items()},
                     "per_fold": {n: r.per_fold.to_dict(orient="records") for n, r in results.items()}}
            if task == "failure_mode":
                entry["events"] = {n: r.tables["events"].to_dict(orient="records") for n, r in results.items()}
                entry["confusion_all"] = {n: r.tables["confusion_all"].to_dict() for n, r in results.items()}
            if task == "early_warning":
                entry["lead_times"] = {n: r.tables["lead_times"].to_dict(orient="records") for n, r in results.items()}
                entry["far_sweep"] = {n: s.to_dict(orient="records") for n, s in sweeps.items()}
            report["tasks"][task] = entry
            log(f"{task}: selected {gate.get('selected')} (gate passed: {gate.get('passes_gate')})")

            # ---- final model: fit on every machine, explain, register -------------------------------------
            best = gate.get("selected")
            if not best:
                continue
            base = best.split("@")[0]
            ds, X = data.ds, data.features
            with mlflow.start_run(run_name=f"final/{task}/{base}", nested=True,
                                  tags={**run_tags(cfg, data, data_label), "task": task, "model_family": base, "final": "true"}):
                seed = cfg["random_state"]
                if task == "failure_mode":
                    m = classification_rows(ds)
                    est = make_classifier(base, cfg["modelling"]["models"].get(base, {}), seed).fit(ds.loc[m, X], ds.loc[m, "failure_mode_label"])
                    model = PdMModel(task, est, X, classes=cfg["modelling"]["classes"])
                elif task == "rul":
                    m = rul_rows(ds)
                    est = make_regressor(base, cfg["modelling"]["models"].get(base, {}), seed).fit(ds.loc[m, X], ds.loc[m, "rul_hours"])
                    model = PdMModel(task, est, X, rul_cap=float(cfg["targets"]["rul_cap_hours"]))
                else:
                    m = warning_rows(ds)
                    est = make_classifier(base, cfg["modelling"]["models"].get(base, {}), seed, balanced=False).fit(
                        ds.loc[m, X], ds.loc[m, "fail_within_h"].astype(int))
                    fin = results[best].final
                    model = PdMModel(task, est, X, calibrator=fin["calibrator"], threshold_raw=fin["threshold_raw"],
                                     threshold_prob=fin["threshold_prob"], persistence_h=fin["persistence_h"],
                                     horizon_h=float(cfg["targets"]["warning_horizon_hours"]))
                    mlflow.log_params({"threshold_raw": fin["threshold_raw"], "threshold_prob": fin["threshold_prob"],
                                       "persistence_h": fin["persistence_h"]})
                mlflow.log_params({"task": task, "model": base, "n_train_rows": int(m.sum()), "trained_on": "all machines"})

                # SHAP (tree models and linear models; skipped for anything else)
                sample = ds.loc[m, X].sample(min(3000, int(m.sum())), random_state=seed)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        classes = list(getattr(est, "classes_", [])) if task == "failure_mode" else None
                        imp = explain.global_importance(est, sample, classes)
                    imp.to_csv(fig_dir.parent / f"stage3_shap_{task}.csv")
                    mlflow.log_table(imp.reset_index().head(50), f"explain/shap_global_{task}.json")
                    entry["shap_top"] = imp["total"].head(15).round(4).to_dict()
                    if task == "failure_mode":
                        f = plots.shap_bars(imp, [c for c in cfg["modelling"]["classes"] if c != "healthy"], 12,
                                            f"Failure mode ({base}): top features per class (mean |SHAP|)")
                        mlflow.log_figure(f, "explain/shap_failure_mode.png")
                        f.savefig(fig_dir / "stage3_shap_failure_mode.png", bbox_inches="tight", dpi=130)
                        plt.close(f)
                    if task == "early_warning":
                        lt = results[best].tables["lead_times"]
                        if lt["detected"].any():
                            e = lt.sort_values("lead_time_h", ascending=False).iloc[0]
                            t_alarm = pd.Timestamp(data.events.set_index("event_id").loc[e.event_id, "failure_timestamp"]) - pd.Timedelta(hours=e.lead_time_h)
                            row = ds[(ds["machine_id"] == e.machine_id) & (ds["timestamp"] == t_alarm)]
                            dictionary = pd.read_csv(cfg["paths"]["processed_dir"] / "feature_dictionary.csv")
                            expl = explain.explain_rows(est, row[X], top_k=10, dictionary=dictionary)
                            expl.to_csv(fig_dir.parent / "stage3_shap_alert_example.csv", index=False)
                            f = plots.shap_alert(expl, f"Why the first alarm fired: {e.machine_id} ({e.failure_mode}), "
                                                       f"{t_alarm:%d %b %H:%M}, {e.lead_time_h:.0f} h before failure")
                            mlflow.log_figure(f, "explain/shap_alert_example.png")
                            f.savefig(fig_dir / "stage3_shap_alert_example.png", bbox_inches="tight", dpi=130)
                            plt.close(f)
                            entry["alert_example"] = {"event_id": e.event_id, "machine_id": e.machine_id, "timestamp": str(t_alarm),
                                                      "top_features": expl[["feature", "value", "shap"]].round(4).to_dict(orient="records")}
                except Exception as exc:          # explanation must never block registration
                    warnings.warn(f"SHAP failed for {task}/{base}: {exc}")

                promote = gate["passes_gate"] and data_label == "real"
                reg = cfg["modelling"]["registry"]
                out = log_and_register(model, ds.loc[m], reg["names"][task] if register else None,
                                       reg["alias"] if (register and promote) else None,
                                       tags={"primary_metric": gate["primary_metric"], "value": round(gate["value"], 4),
                                             "passes_gate": gate["passes_gate"], "data_label": data_label,
                                             "dataset_hash": data.dataset_hash})
                entry["registered"] = {"name": reg["names"][task] if register else None, "version": out["version"],
                                       "alias": reg["alias"] if (register and promote) else None, "model_uri": out["model_uri"]}
                log(f"{task}: final {base} logged" + (f" as {reg['names'][task]} v{out['version']}" if out["version"] else "")
                    + (f" @{reg['alias']}" if entry["registered"]["alias"] else ""))

        report["runtime_s"] = round(time.time() - t0, 1)
        out_dir = cfg["paths"]["reports_dir"]
        (out_dir / "stage3_model_comparison.json").write_text(json.dumps(report, indent=2, default=_json_default))
        (out_dir / "stage3_model_comparison.md").write_text(markdown_report(report, cfg))
        mlflow.log_artifact(str(out_dir / "stage3_model_comparison.json"))
        mlflow.log_artifact(str(out_dir / "stage3_model_comparison.md"))
    log("done")
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:,.{nd}f}" if isinstance(v, float) else str(v)


def markdown_report(report: dict, cfg: dict) -> str:
    L = [f"# Stage 3 model comparison ({report['data_label']} data)", "",
         f"Leave-one-machine-out, out-of-fold. Dataset `{report['dataset_hash']}`, {report['n_features']} features. "
         f"MLflow parent run `{report.get('mlflow_parent_run_id', '')}`.", ""]
    cols = {"failure_mode": [("macro_f1_seen", "macro-F1 (seen folds)", 3), ("macro_f1_all", "macro-F1 (all folds)", 3),
                             ("detection_recall", "detection recall", 3), ("healthy_false_positive_rate", "healthy FPR", 3),
                             ("unseen_fold5_detection_recall", "HPU_06 detection", 3), ("events_mode_correct", "events correct", 0)],
            "rul": [("mae_critical", "MAE last 14 d (h)", 1), ("rmse_critical", "RMSE last 14 d (h)", 1), ("nasa_critical", "NASA last 14 d", 3),
                    ("mae_all", "MAE all (h)", 1), ("rmse_all", "RMSE all (h)", 1)],
            "early_warning": [("lead_time_mean_h", "mean lead (h)", 1), ("lead_time_median_h", "median lead (h)", 1),
                              ("events_detected", "events detected", 0), ("false_alarm_rate", "false-alarm rate", 4),
                              ("precision", "precision", 3), ("recall", "recall", 3), ("pr_auc", "PR-AUC", 3), ("brier", "Brier", 3)]}
    titles = {"failure_mode": "Failure-mode classification", "rul": "RUL regression", "early_warning": "Early warning (7 days)"}
    for task, e in report["tasks"].items():
        g = e["gate"]
        L += [f"## {titles[task]}", "", "| model | " + " | ".join(c[1] for c in cols[task]) + " |",
              "|---|" + "---:|" * len(cols[task])]
        for n, m in e["models"].items():
            s = m["summary"]
            vals = []
            for k, _, nd in cols[task]:
                v = s.get(k)
                if k == "events_mode_correct":
                    v = f"{s.get('events_mode_correct')}/{s.get('events_total')}"
                elif k == "events_detected":
                    v = f"{s.get('events_detected')}/{s.get('events_total')}"
                elif v is not None and nd == 0:
                    v = str(int(v))
                vals.append(_fmt(v, nd) if not isinstance(v, str) else v)
            name = f"**{n}**" if n == g.get("selected") else n
            L.append(f"| {name} | " + " | ".join(vals) + " |")
        if g.get("selected"):
            reg = e.get("registered", {})
            L += ["", f"Selected **{g['selected']}** on `{g['primary_metric']}` = {_fmt(g['value'])}; "
                      f"beats baselines: {g['beats_baselines']}; gate passed: **{g['passes_gate']}**."
                      + (f" Registered as `{reg.get('name')}` v{reg.get('version')}" if reg.get("version") else "")
                      + (f" with alias `@{reg.get('alias')}`." if reg.get("alias") else ".")]
        L.append("")
    return "\n".join(L)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="Stage 3: train, evaluate (LOMO), track and register models")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--models", nargs="+", default=None, help="restrict the non-baseline candidates, e.g. lightgbm")
    ap.add_argument("--data-label", default="real", help="tag for the data source; only 'real' models get the champion alias")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)
    rep = run(load_config(a.config), a.tasks, a.models, a.data_label, register=not a.no_register)
    print(json.dumps({t: e["gate"] for t, e in rep["tasks"].items()}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
