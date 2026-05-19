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

Planned preview release.

### Planned

- Runbook document structure
- RAG retrieval design
- Vector database selection
- `/retrieve` API design
- Evidence-based retrieval response schema

## release-2026.07-incident-assist-beta

### Status

Planned beta release.

### Planned

- Classifier + RAG integration
- `/assist` API design
- LLM-generated remediation guidance
- Evidence citations
- Root cause candidate generation

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
