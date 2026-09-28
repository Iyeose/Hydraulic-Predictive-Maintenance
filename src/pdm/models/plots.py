"""Figures for Stage 3 (matplotlib only; used by the training CLI, MLflow and notebook 03)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same colour per failure mode as notebooks 01 and 02
MODE_COLORS = {"healthy": "#9a9892", "pump_wear": "#2a78d6", "valve_leakage": "#eb6834",
               "contamination": "#1baf7a", "cylinder_drift": "#4a3aa7", "repair": "#e34948"}
MODEL_COLORS = {"rules": "#b5b3ab", "constant": "#b5b3ab", "logreg": "#7d7b74", "ridge": "#7d7b74",
                "lightgbm": "#2a78d6", "xgboost": "#eb6834"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
                     "axes.titlesize": 11, "grid.color": "#e6e5e0", "legend.frameon": False, "figure.dpi": 100})


def confusion_heatmap(cm: pd.DataFrame, title: str):
    norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(norm.fillna(0).to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = norm.iat[i, j]
            ax.text(j, i, f"{cm.iat[i, j]:,}\n{v:.0%}" if pd.notna(v) else "–", ha="center", va="center", fontsize=8,
                    color="white" if pd.notna(v) and v > 0.6 else "#333")
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=30, ha="right")
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    return fig


def per_fold_bars(frames: dict[str, pd.DataFrame], metric: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    names = list(frames)
    w = 0.8 / len(names)
    base = frames[names[0]]
    x = np.arange(len(base))
    for k, n in enumerate(names):
        ax.bar(x + (k - (len(names) - 1) / 2) * w, frames[n][metric], width=w, label=n, color=MODEL_COLORS.get(n.split("@")[0], None))
    ax.set_xticks(x, base["test_machine"] + "\n" + base["test_modes"].str.replace("none (healthy reference)", "healthy", regex=False),
                  fontsize=8)
    ax.set(ylabel=ylabel, title=title)
    ax.legend(ncol=len(names), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def lead_time_bars(lead: dict[str, pd.DataFrame], title: str):
    """Lead time per failure event for each model, over the event's warning window (grey)."""
    first = next(iter(lead.values())).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(first) + 1.4))
    y = np.arange(len(first))
    ax.barh(y, first["window_h"], color="#eeede8", height=0.8, label="warning window")
    h = 0.8 / len(lead)
    for k, (n, lt) in enumerate(lead.items()):
        lt = lt.set_index("event_id").reindex(first["event_id"])
        ax.barh(y - 0.4 + h * (k + 0.5), lt["lead_time_h"], height=h, label=n, color=MODEL_COLORS.get(n.split("@")[0], None))
    ax.set_yticks(y, first["machine_id"] + " · " + first["failure_mode"], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="hours of warning before failure", title=title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def far_curve(sweeps: dict[str, pd.DataFrame], rule_point: dict | None, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for n, s in sweeps.items():
        axes[0].plot(s["realised_far"] * 100, s["lead_time_median_h"], "o-", label=n, color=MODEL_COLORS.get(n))
        axes[1].plot(s["realised_far"] * 100, s["recall"], "o-", label=n, color=MODEL_COLORS.get(n))
    if rule_point:
        axes[0].scatter([rule_point["false_alarm_rate"] * 100], [rule_point["lead_time_median_h"]], marker="s", s=60,
                        color=MODEL_COLORS["rules"], label="rules (EDA §8.3)", zorder=3)
        axes[1].scatter([rule_point["false_alarm_rate"] * 100], [rule_point["recall"]], marker="s", s=60, color=MODEL_COLORS["rules"], zorder=3)
    axes[0].set(xlabel="false-alarm rate (% of healthy hours in alarm)", ylabel="median lead time (h)", title=title)
    axes[1].set(xlabel="false-alarm rate (% of healthy hours in alarm)", ylabel="hourly recall (fails within 7 d)", title="Recall vs false alarms")
    for a in axes:
        a.grid()
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def rul_traces(oof: dict[str, pd.DataFrame], machines: list[str], cap: float, title: str):
    fig, axes = plt.subplots(1, len(machines), figsize=(5.2 * len(machines), 3.4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], machines):
        first = True
        for n, o in oof.items():
            d = o[o["machine_id"] == m].sort_values("timestamp")
            if first:
                ax.plot(d["timestamp"], d["y_true"], color="#333", lw=2, label="true (capped) RUL")
                first = False
            ax.plot(d["timestamp"], d["y_pred"], lw=1.2, label=n, color=MODEL_COLORS.get(n), alpha=0.9)
        ax.set(title=m, ylim=(-10, cap * 1.1))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0][0].set_ylabel("RUL (h)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontweight="semibold")
    fig.tight_layout()
    return fig


def shap_bars(imp: pd.DataFrame, classes: list[str], top: int, title: str):
    cols = [c for c in classes if c in imp.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 0.28 * top + 1.4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        s = imp[c].sort_values(ascending=False).head(top)[::-1]
        ax.barh(range(len(s)), s.to_numpy(), color=MODE_COLORS.get(c, "#2a78d6"))
        ax.set_yticks(range(len(s)), s.index, fontsize=7)
        ax.set(title=c, xlabel="mean |SHAP|")
    fig.suptitle(title, fontweight="semibold")
    fig.tight_layout()
    return fig


def shap_alert(expl: pd.DataFrame, title: str):
    e = expl.sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(e) + 1.2))
    ax.barh(range(len(e)), e["shap"], color=np.where(e["shap"] > 0, "#e34948", "#2a78d6"))
    ax.set_yticks(range(len(e)), [f"{f} = {v:.3g}" for f, v in zip(e["feature"], e["value"])], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set(xlabel="SHAP contribution to log-odds of failure within 7 days", title=title)
    fig.tight_layout()
    return fig
