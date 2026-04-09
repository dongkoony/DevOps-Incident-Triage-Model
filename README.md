[![CI](https://github.com/dongkoony/DevOps-Incident-Triage-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/dongkoony/DevOps-Incident-Triage-Model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dongkoony/DevOps-Incident-Triage-Model)](https://github.com/dongkoony/DevOps-Incident-Triage-Model/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

# DevOps Incident Triage Model

English | [한국어](README.ko.md)

Portfolio-grade NLP and MLOps project for classifying DevOps incident text into the most relevant operational domain for first-pass routing.

This repository focuses on production-minded engineering rather than demo-only modeling:

- reproducible training and evaluation pipelines
- local inference, FastAPI serving, and async batch jobs
- observability with request tracing and Prometheus-style metrics
- Docker, CI, release workflow, and Hugging Face publishing
- explicit documentation of data limitations and operational scope

## Overview

The model takes incident summaries, deployment failures, and log-style operational messages and predicts which domain should review the issue first.

Current label set:

| Label | Description |
|---|---|
| `k8s_cluster` | Kubernetes scheduling, node, or cluster-state issues |
| `cicd_pipeline` | CI/CD build, test, or deployment pipeline failures |
| `aws_iam_network` | AWS IAM, VPC, network, or permission-related issues |
| `deployment_release` | Helm, rollout, or release operation issues |
| `container_runtime` | Docker, containerd, image, or runtime issues |
| `observability_alerting` | Monitoring, logging, tracing, or alerting issues |
| `database_state` | Database connectivity, replication, lock, or storage-state issues |

## Repository Scope

This repository includes more than a trained classifier.

- Model training with `transformers`
- Evaluation with confusion matrix and threshold-based review analysis
- CLI inference for single and batch inputs
- FastAPI endpoints for real-time and batch inference
- Async batch job API for queue-like inference workflows
- Benchmark automation across multiple backbone models
- Release and documentation flow suitable for a portfolio-grade MLOps project

## Data Honesty

The starter dataset in `data/sample/incidents_synthetic.csv` is synthetic.

- It is not collected from a real production environment.
- Reported scores should not be interpreted as validated real-world generalization.
- Real anonymized incident data is required before any serious operational use.

That limitation is a deliberate part of the project documentation and not hidden in the evaluation results.

## Model And Experimentation

Baseline model:

- `distilbert-base-uncased`

Why this baseline:

- DevOps logs and error messages are often English-dominant
- training and iteration cost remain practical on a personal environment
- the same pipeline can be reused for multilingual backbones such as `xlm-roberta-base`

Benchmark workflow:

```bash
uv run ditri-benchmark \
  --data-dir data/processed \
  --models distilbert-base-uncased,sentence-transformers/all-MiniLM-L6-v2,xlm-roberta-base \
  --epochs 4 \
  --skip-existing
```

Generated outputs:

- `reports/model_benchmark.json`
- `reports/model_benchmark.md`
- `models/benchmarks/<model-slug>/`
- `reports/benchmarks/<model-slug>/`

## Quickstart

### 1. Environment

```bash
uv python install 3.12
uv sync --extra dev --extra api --extra viz --extra peft --extra gradio
```

### 2. Data Preparation

Using the synthetic starter dataset:

```bash
uv run ditri-data-prep \
  --input-path data/sample/incidents_synthetic.csv \
  --output-dir data/processed \
  --seed 42
```

Using anonymized real data:

```bash
uv run ditri-ingest-raw \
  --input-path data/raw/incidents_template.csv \
  --output-canonical-path data/raw/incidents_canonical.csv \
  --output-training-path data/raw/incidents_training_ready.csv \
  --report-path reports/raw_ingestion_report.json

uv run ditri-data-prep \
  --input-path data/raw/incidents_training_ready.csv \
  --output-dir data/processed \
  --seed 42
```

### 3. Training

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --epochs 4
```

Optional PEFT:

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --use-peft
```

### 4. Evaluation

```bash
uv run ditri-eval \
  --model-path models/devops-incident-triage \
  --data-dir data/processed \
  --report-dir reports \
  --confidence-thresholds 0.4,0.5,0.6,0.7
```

Key artifacts:

- `reports/evaluation_metrics.json`
- `reports/per_label_metrics.json`
- `reports/threshold_metrics.json`
- `reports/confusion_matrix.csv`
- `reports/figures/confusion_matrix.png`
- `reports/sample_predictions.jsonl`

## Inference And Serving

### CLI Prediction

Single input:

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage \
  --text "EKS worker nodes became NotReady after CNI upgrade."
```

Batch input:

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --input-file data/sample/incidents_synthetic.csv \
  --text-column text \
  --output-file reports/batch_predictions.jsonl
```

### Demo Showcase Report

For live demos and portfolio sharing, generate a curated showcase report from representative incident examples:

```bash
uv run ditri-demo-showcase \
  --model-path models/devops-incident-triage \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage
```

Generated artifacts:

- `reports/demo_showcase.json`
- `reports/demo_showcase.md`

This workflow is useful when preparing terminal demos, README evidence, or GIF/video walkthroughs because the terminal summary and saved report come from the same curated examples.

### FastAPI

```bash
CONFIDENCE_THRESHOLD=0.6 REVIEW_QUEUE=sre_manual_triage BATCH_MAX_ITEMS=32 uv run ditri-api
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `POST /predict/batch/async`
- `GET /predict/batch/async/{job_id}`
- `GET /metrics`

Operational features:

- `X-Request-ID` response header for traceability
- confidence threshold based human review routing
- async batch job flow for queue-like consumption
- Prometheus-compatible metrics exposure

## Delivery And Release

This repository follows a lightweight GitFlow-style process:

- `main`: release-ready branch
- `develop`: integration branch
- `feature/*`: scoped feature work
- `release/*`: release stabilization

Current project release:

- `v0.3.0`

Related operational documentation:

- [Branch strategy](docs/branch_strategy.md)
- [Release checklist](docs/release_checklist.md)
- [Architecture](docs/architecture.md)
- [Model benchmarking guide](docs/model_benchmarking.md)
- [Portfolio notes](docs/portfolio_notes.md)

## Hugging Face Publishing

```bash
export HF_TOKEN="hf_xxx"

uv run ditri-publish \
  --model-dir models/devops-incident-triage \
  --repo-id <your-hf-username>/devops-incident-triage
```

The publish flow copies `docs/model_card.md` into the model artifact directory as `README.md` when needed.

## Repository Layout

```text
.
├─ src/devops_incident_triage/
│  ├─ data_prep.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ benchmark_models.py
│  ├─ predict.py
│  ├─ api.py
│  ├─ hf_publish.py
│  └─ ingest_raw.py
├─ tests/
├─ data/
├─ reports/
├─ models/
├─ docs/
├─ .github/workflows/
├─ Dockerfile
├─ Makefile
└─ pyproject.toml
```

## Limitations

- training data is synthetic in the current public baseline
- the task is single-label even though real incidents may span multiple domains
- long multi-line logs and highly noisy contexts need additional validation
- the model is intended for triage support, not autonomous remediation

## License

MIT
