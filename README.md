# Hydraulic Predictive Maintenance: Bosch Rexroth HPUs

Predict the **failure mode** (pump wear, valve leakage, contamination, cylinder drift) and the
**Remaining Useful Life (RUL)** of hydraulic power units from 1-minute sensor telemetry.

| Stage | Scope | Status |
|---|---|---|
| 1 | Exploratory data analysis (`notebooks/01_…`) | ✅ confirmed |
| 2 | Data validation, cleaning, targets, feature engineering (`src/pdm`, `notebooks/02_…`) | ✅ confirmed (`v0.2.0-stage2`) |
| 3 | Model development and experiment tracking (`src/pdm/models`, `notebooks/03_…`, MLflow) | 🔄 for review |
| 4 | CI/CD with GitHub Actions | planned |
| 5 | Containerisation and deployment (Docker, ECR, EC2) | planned |
| 6 | Dashboard and real-time inference (Streamlit) | planned |
| 7 | Monitoring, drift detection, retraining | planned |

## Project layout

```
bosch_rexroth_pdm/
├── configs/config.yaml        # every threshold and rule; reviewed like code
├── data/
│   ├── raw/                   # source files (not committed)
│   ├── interim/               # bronze (typed raw) and silver (cleaned) Parquet
│   ├── processed/             # dataset.parquet, feature_list.json, folds.csv, baselines
│   └── quarantine/            # rows rejected by validation, with reasons
├── notebooks/                 # 01 EDA, 02 validation & features, 03 modelling (Colab-ready)
├── reports/                   # validation report, pipeline summary, model comparison, figures
├── src/pdm/
│   ├── config.py
│   ├── data/      ingest.py · schemas.py · validate.py · clean.py
│   ├── features/  baseline.py · targets.py · build.py · selection.py · splits.py
│   ├── models/    data.py · baselines.py · estimators.py · metrics.py · cv.py
│   │              tracking.py · explain.py · registry.py · plots.py
│   ├── pipeline.py            # Stage 2 CLI: raw → dataset
│   ├── train.py               # Stage 3 CLI: evaluate, track, explain, register
│   └── synthetic.py           # synthetic raw files for smoke tests (dev/CI only)
├── tests/                     # pytest, synthetic fixtures, runs in seconds
├── mlflow.db, mlartifacts/    # local MLflow tracking store and artefacts (not committed)
```

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
# put sensor_telemetry.csv, failure_labels.csv and maintenance_log.csv in data/raw/
pytest                                   # 41 tests
python -m pdm.pipeline                   # ≈ 30 s → data/processed/dataset.parquet
python -m pdm.train                      # ≈ 8 min → MLflow runs, registered models, reports/stage3_*
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse runs and the Model Registry
```

On Google Colab, open `notebooks/03_Model_Development_and_Experiment_Tracking.ipynb` (or `02_…` for
the data pipeline). Its first cell mounts Drive, installs the package and runs everything it needs.

**Without the private data** (fresh clone, CI): `python -m pdm.synthetic --out data/raw` writes
synthetic files with the real schema and the EDA characteristics, so the whole chain runs. Train with
`--data-label synthetic`: such models are never given the `@champion` alias, and their metrics say
nothing about the real HPUs.

## Stage 2 design decisions

* **Validation gate (pandera):** structural failures reject the batch. Invalid records are
  quarantined with a reason. Physically implausible *values* are warned and fixed in cleaning.
* **Causal cleaning:** gaps are forward-filled for at most 15 min, not linearly interpolated,
  because interpolation uses future readings that a live system would not have.
* **Regime-aware normalisation:** each machine is compared with its own healthy baseline for
  the same day type and hour. This removes shift set points, weekends, the midnight dip and the
  undocumented standby day.
* **Targets rebuilt from `failure_labels.csv`:** capped RUL (336 h) with right-censoring, a
  failure-mode label, and a 7-day early-warning flag. Repair downtime is excluded.
* **Leakage controls:** label columns are dropped, all features look backwards (enforced by
  `test_no_look_ahead`), and an identity-leak screen removes features that reveal *which
  machine* a row came from. Validation uses **leave-one-machine-out** folds.


## Stage 3 design decisions

* **Leave-one-machine-out, out-of-fold everything.** Every reported number comes from a model that
  never saw the machine it scored. The cylinder-drift fold (HPU_06, the only such machine) is
  reported separately as a detection rate, not averaged into macro-F1.
* **Baselines first.** The EDA §8.3 rule detector (alarm, plus a mode from the rule that fired) and
  logistic regression for classification and early warning; constant and ridge for RUL. A model is
  given the registry alias `@champion` only if it beats every baseline on the task's primary metric.
* **Primary metrics:** macro-F1 on seen-mode folds; RUL MAE in the last 14 days before failure (plus
  RMSE and the asymmetric NASA score); early-warning mean lead time per event at a fixed
  false-alarm rate (0.2 % of healthy hours, 3-hour persistence), compared at the rule detector's
  own false-alarm rate when that is lower.
* **Early warning without leakage:** isotonic calibration and the alarm threshold are fitted on an
  inner leave-one-machine-out over the nine training machines of each fold.
* **No hyperparameter tuning** with 9 failure events; fixed conservative LightGBM / XGBoost settings
  in `configs/config.yaml → modelling`.
* **Tracking:** one MLflow parent run per training call, one nested run per task × model, tagged with
  git commit, config hash and dataset fingerprint; per-fold metrics as step series.
* **Serving contract:** each selected model is an `mlflow.pyfunc` bundle (estimator, feature list,
  classes, calibrator, threshold) registered as `hpu-failure-mode`, `hpu-rul`, `hpu-early-warning`.
* **Sequence model not built:** 9 degradation trajectories are too few, and a win over boosting
  could not be shown within the evaluation noise (notebook 03 §9).
