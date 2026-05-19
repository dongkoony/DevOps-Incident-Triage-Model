# release-2026.05-classifier-core

## Channel

Stable

## Summary

This release train entry packages the current Transformer-based DevOps incident classifier as the stable classifier-core baseline. It covers first-pass domain classification, CLI inference, FastAPI serving, batch prediction, async batch jobs, evaluation reports, Docker packaging intent, CI workflow, and release documentation.

This is not a RAG release. `/retrieve`, `/assist`, Vector DB integration, and LLM-generated remediation guidance are planned for later release train entries.

## Scope

Included:

- Transformer-based incident classification
- Synthetic starter dataset preparation
- CLI single and batch prediction
- FastAPI `/health`, `/predict`, `/predict/batch`, async batch, and `/metrics` routes
- Evaluation and threshold-analysis reports
- Demo showcase command
- Dockerfile and CI workflow
- Release train documentation and changelog

Excluded:

- RAG backend
- Vector DB
- `/retrieve` API
- `/assist` API
- LLM response generation
- Cloud deployment automation

## Validation Evidence

Run from branch `release/release-2026.05-classifier-core` on 2026-05-18.

```powershell
uv run --extra dev --extra api --extra viz ruff check .
```

Result:

```text
All checks passed!
```

```powershell
uv run --extra dev --extra api --extra viz pytest -q
```

Result:

```text
40 passed, 10 skipped
```

The skipped tests are optional portfolio website checks. The `web/` assets are not part of the classifier-core release package.

```powershell
uv run --extra dev --extra api --extra viz ditri-data-prep --input-path data/sample/incidents_synthetic.csv --output-dir C:\Users\dongh\AppData\Local\Temp\ditri-release-202605-data --seed 42
```

Result:

```text
train: 38
validation: 9
test: 9
total: 56
```

```powershell
uv run --extra dev --extra api --extra viz ditri-demo-showcase --model-path hf-internal-testing/tiny-random-distilbert --confidence-threshold 0.6 --review-queue sre_manual_triage --output-json C:\Users\dongh\AppData\Local\Temp\ditri-release-202605-demo.json --output-markdown C:\Users\dongh\AppData\Local\Temp\ditri-release-202605-demo.md
```

Result:

```text
Examples: 7
Matches: 0/7
Human review: 7
```

The showcase smoke uses the portable CI model reference. It verifies command execution and report generation, not classifier quality.

FastAPI smoke:

```text
GET /health -> 200
POST /predict -> 200
GET /metrics -> 200
```

Environment:

```text
MODEL_PATH=hf-internal-testing/tiny-random-distilbert
CONFIDENCE_THRESHOLD=0.6
REVIEW_QUEUE=sre_manual_triage
```

Docker smoke:

```powershell
docker version
```

Result:

```text
Docker client is installed, but the Docker Desktop Linux engine is not running.
```

Docker build validation remains a local environment follow-up or CI gate.

## Known Limitations

- The public starter dataset is synthetic and should not be treated as real-world performance evidence.
- The portable tiny model used in smoke checks is for workflow validation only.
- RAG and LLM assistant features are documented as future roadmap items and are not implemented in this release.
- Docker build could not be verified in this local Windows session because the Docker daemon was unavailable.

## Release Readiness

Ready for documentation and classifier-core review, with Docker build verification still pending before a production-style stable tag is cut.

