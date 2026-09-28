# Stage 3 model comparison (real data)

Leave-one-machine-out, out-of-fold. Dataset `0f087751a4eb`, 171 features. MLflow parent run `3bf75f1ca15f42d48d9d240383965748`.

## Failure-mode classification

| model | macro-F1 (seen folds) | macro-F1 (all folds) | detection recall | healthy FPR | HPU_06 detection | events correct |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.623 | 0.655 | 0.837 | 0.000 | 0.646 | 6/9 |
| logreg | 0.944 | 0.734 | 0.958 | 0.004 | 0.995 | 8/9 |
| **lightgbm** | 0.988 | 0.745 | 0.962 | 0.000 | 0.906 | 8/9 |
| xgboost | 0.988 | 0.767 | 0.964 | 0.000 | 0.917 | 8/9 |

Selected **lightgbm** on `macro_f1_seen` = 0.988; beats baselines: {'rules': True, 'logreg': True}; gate passed: **True**. Registered as `hpu-failure-mode` v1 with alias `@champion`.

## RUL regression

| model | MAE last 14 d (h) | RMSE last 14 d (h) | NASA last 14 d | MAE all (h) | RMSE all (h) |
|---|---:|---:|---:|---:|---:|
| constant | 120.9 | 146.4 | 0.752 | 84.0 | 101.5 |
| ridge | 45.6 | 68.4 | 0.234 | 25.6 | 43.7 |
| lightgbm | 41.2 | 67.5 | 0.216 | 23.0 | 44.1 |
| **xgboost** | 39.9 | 64.9 | 0.206 | 24.0 | 45.3 |

Selected **xgboost** on `mae_critical` = 39.886; beats baselines: {'constant': True, 'ridge': True}; gate passed: **True**. Registered as `hpu-rul` v1 with alias `@champion`.

## Early warning (7 days)

| model | mean lead (h) | median lead (h) | events detected | false-alarm rate | precision | recall | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 195.3 | 201.0 | 9/9 | 0.0000 | 1.000 | 0.895 | – | – |
| logreg | 205.7 | 214.0 | 9/9 | 0.0101 | 0.960 | 0.863 | 0.901 | 0.025 |
| **lightgbm** | 220.9 | 219.0 | 9/9 | 0.0000 | 1.000 | 0.931 | 0.945 | 0.017 |
| xgboost | 217.1 | 221.0 | 9/9 | 0.0043 | 0.984 | 0.847 | 0.890 | 0.033 |
| lightgbm@rules_far | 219.8 | 219.0 | 9/9 | 0.0000 | 1.000 | 0.903 | 0.945 | 0.017 |
| xgboost@rules_far | 213.7 | 212.0 | 9/9 | 0.0026 | 0.990 | 0.833 | 0.890 | 0.033 |

Selected **lightgbm** on `lead_time_mean_h` = 220.889; beats baselines: {'rules': True, 'logreg': True}; gate passed: **True**. Registered as `hpu-early-warning` v1 with alias `@champion`.
