.PHONY: install format lint test ci run api ingest features train evaluate train-multimodal evaluate-multimodal rank docker-build docker-up mlflow-ui

install:
	pip install -e .[dev]
	pre-commit install

format:
	black src tests
	isort src tests

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

test:
	pytest -q

# Mirrors CI locally (Unix make: set MLFLOW_DISABLE for pytest).
ci: lint
	MLFLOW_DISABLE=1 pytest -q

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api: run

ingest:
	python scripts/run_data_ingestion.py

features:
	python scripts/run_feature_pipeline.py

train:
	python scripts/run_train_pipeline.py

evaluate:
	python scripts/run_evaluation.py

train-multimodal:
	python scripts/run_multimodal_train.py

evaluate-multimodal:
	python scripts/run_multimodal_evaluation.py

rank:
	python scripts/run_ranking.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up --build

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns
