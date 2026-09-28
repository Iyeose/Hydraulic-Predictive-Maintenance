.PHONY: install test pipeline train mlflow-ui synthetic clean

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest

pipeline:
	python -m pdm.pipeline --config configs/config.yaml

train:
	python -m pdm.train --config configs/config.yaml

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Synthetic raw files for a smoke test without the private data (never report its metrics)
synthetic:
	python -m pdm.synthetic --out data/raw

clean:
	rm -rf data/interim/* data/processed/* data/quarantine/*
