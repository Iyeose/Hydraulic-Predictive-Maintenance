"""Stage 3: model development, evaluation and experiment tracking.

data       loading the Stage 2 artefacts, task row masks, leakage guard
baselines  EDA §8.3 rule detector, constant RUL
estimators logistic / ridge / LightGBM / XGBoost factories
metrics    macro-F1, NASA score, alarms, false-alarm rate, lead time
cv         leave-one-machine-out runners (nested calibration + threshold for early warning)
tracking   MLflow runs, tags, artifacts
explain    SHAP (global and per alert)
registry   pyfunc packaging and Model Registry
plots      figures
"""
