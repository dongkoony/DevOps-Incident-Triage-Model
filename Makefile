PYTHON_VERSION=3.12
MODEL_DIR=models/devops-incident-triage
DATA_INPUT=data/sample/incidents_synthetic.csv

.PHONY: help install prep-data train eval predict api gradio docker-build docker-run test lint

help:
	@echo "Targets:"
	@echo "  install      Install dependencies with uv"
	@echo "  prep-data    Prepare train/validation/test splits"
	@echo "  train        Train baseline model"
	@echo "  eval         Evaluate trained model"
	@echo "  predict      Run a sample local prediction"
	@echo "  api          Start FastAPI server"
	@echo "  gradio       Start Gradio demo app"
	@echo "  lint         Run ruff"
	@echo "  test         Run pytest"
	@echo "  docker-build Build API Docker image"
	@echo "  docker-run   Run API Docker container"

install:
	uv python install $(PYTHON_VERSION)
	uv sync --extra dev --extra api --extra viz --extra peft --extra gradio

prep-data:
	uv run ditri-data-prep --input-path $(DATA_INPUT) --output-dir data/processed --seed 42

train:
	uv run ditri-train --data-dir data/processed --output-dir $(MODEL_DIR) --model-name distilbert-base-uncased

eval:
	uv run ditri-eval --model-path $(MODEL_DIR) --data-dir data/processed --report-dir reports

predict:
	uv run ditri-predict --model-path $(MODEL_DIR) --text "EKS nodes became NotReady after CNI upgrade and deployments are pending."

api:
	uv run ditri-api

gradio:
	uv run ditri-gradio

lint:
	uv run ruff check .

test:
	uv run pytest -q

docker-build:
	docker build -t devops-incident-triage:latest .

docker-run:
	docker run --rm -p 8000:8000 -e MODEL_PATH=/app/models/devops-incident-triage devops-incident-triage:latest
