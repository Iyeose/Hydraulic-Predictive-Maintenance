.PHONY: install test pipeline clean

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest

pipeline:
	python -m pdm.pipeline --config configs/config.yaml

clean:
	rm -rf data/interim/* data/processed/* data/quarantine/*
