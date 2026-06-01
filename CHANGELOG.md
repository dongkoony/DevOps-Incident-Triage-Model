# Changelog

## release-2026.05-classifier-core

### Status

Stable classifier baseline.

### Added

- Transformer-based DevOps incident classification
- FastAPI inference endpoint
- Batch prediction support
- Async batch job API
- Model evaluation pipeline
- Docker-based local serving
- CI and release workflow

### Validation

- `ruff check .` passed
- `pytest -q` passed with `40 passed, 10 skipped`
- Data preparation smoke passed against the synthetic starter dataset
- Demo showcase smoke generated JSON and Markdown outputs with the portable CI model reference
- FastAPI smoke confirmed `/health`, `/predict`, and `/metrics`

### Notes

- Docker build validation is still pending because the local Docker daemon was unavailable during release preparation.
- RAG, Vector DB, `/retrieve`, `/assist`, and LLM assistant features remain out of scope for this classifier-core release.

## release-2026.06-rag-preview

### Status

In progress preview release.

### Added

- Runbook corpus loading from `docs/runbooks/`
- Domain-aware preview retrieval
- scikit-learn TF-IDF sparse vector index selection
- `POST /retrieve` API
- Evidence-based retrieval response schema with citations
- Prometheus retrieval request and latency metrics
- Retrieval unit and API tests
- RAG evaluation plan

### Notes

- This is a preview retrieval layer, not a production Vector DB deployment.
- `/assist` and LLM-generated remediation guidance remain planned for the incident-assist beta release.

### Validation

- `ruff check .` passed
- `pytest -q` passed with `47 passed, 10 skipped`
- FastAPI smoke confirmed `/health`, `/retrieve`, and `/metrics`

## release-2026.07-incident-assist-beta

### Status

In progress beta release.

### Added

- Classifier + RAG integration
- Deterministic `POST /assist` beta endpoint
- Evidence-grounded assistant response schema
- Root cause candidate generation from retrieved runbook sections
- Recommended actions with citations
- Assistant safety notes
- Prometheus assistant request and latency metrics
- Unit and API tests for assistant behavior

### Notes

- This beta endpoint is LLM-ready but does not call an external LLM yet.
- The assistant does not execute remediation actions.
- Guidance is based on retrieved runbook evidence and should be reviewed by an operator.

### Validation

- `ruff check .` passed
- `pytest -q` passed with `53 passed, 10 skipped`

## release-2026.08-eval-observability

### Status

Planned beta release.

### Planned

- RAG quality evaluation
- Groundedness checks
- Hallucination checks
- Retrieval and generation latency metrics
- Prometheus-style metric expansion

## release-2026.09-cloud-stable

### Status

Planned stable release.

### Planned

- AWS deployment roadmap
- Production-style service architecture
- Vector DB deployment option
- CI/CD release train workflow
- Monitoring and operational documentation
