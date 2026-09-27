# Hydraulic Predictive Maintenance: Bosch Rexroth HPUs

Predict the **failure mode** (pump wear, valve leakage, contamination, cylinder drift) and the
**Remaining Useful Life (RUL)** of hydraulic power units from 1-minute sensor telemetry.

| Stage | Scope | Status |
|---|---|---|
| 1 | Exploratory data analysis (`notebooks/01_…`) | ✅ confirmed |
| 2 | Data validation, cleaning, targets, feature engineering (`src/pdm`, `notebooks/02_…`) | 🔄 for review |
| 3 | Model development and experiment tracking (MLflow) | planned |
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
├── notebooks/                 # 01 EDA, 02 validation & features (Colab-ready)
├── reports/                   # validation report, pipeline summary, figures
├── src/pdm/
│   ├── config.py
│   ├── data/      ingest.py · schemas.py · validate.py · clean.py
│   ├── features/  baseline.py · targets.py · build.py · selection.py · splits.py
│   └── pipeline.py            # end-to-end CLI
└── tests/                     # pytest, synthetic fixtures, runs in seconds
```

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
# put sensor_telemetry.csv, failure_labels.csv and maintenance_log.csv in data/raw/
pytest                                   # 27 tests
python -m pdm.pipeline                   # ≈ 30 s → data/processed/dataset.parquet
```

On Google Colab, open `notebooks/02_Data_Validation_and_Feature_Engineering.ipynb`. Its first cell
mounts Drive, installs the package and runs the pipeline.

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
