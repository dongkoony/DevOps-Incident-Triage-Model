# CI Hardening Design

## Goal

Strengthen the repository CI so it better represents production-minded MLOps work without introducing heavyweight deployment automation before the deployment target is clearly defined.

## Current State

The existing GitHub Actions workflow already runs:

- dependency install with `uv`
- `ruff check .`
- `pytest -q`
- `ditri-data-prep` smoke test against the synthetic sample dataset

This is a solid baseline, but it still leaves several user-facing repository paths unverified in CI:

- the curated demo showcase command
- the FastAPI application boot path
- the container build path

## Why This Matters

This repository is meant to demonstrate more than model training.
It also showcases:

- CLI workflows
- API serving
- documentation-backed release discipline
- reproducible artifacts and reports

If CI only checks lint, tests, and data prep, it misses some of the most visible parts of the project.

## Non-Goals

This design does not introduce:

- full continuous deployment
- Kubernetes deployment automation
- GitOps controllers such as Argo CD or Flux
- automatic model retraining in CI

Those can be considered later if the project gains a real deployment target.

## Design Principles

- Keep CI fast enough for pull requests.
- Prefer smoke tests over full training/evaluation runs.
- Reuse existing CLI entrypoints to validate the public developer interface.
- Avoid adding secrets-dependent jobs to the default PR workflow.
- Separate "always run" checks from release-only or manual workflows.

## Proposed CI Layers

### Layer 1: Pull Request Baseline

Run on every push and pull request:

- checkout
- Python 3.12 setup
- `uv sync --extra dev --extra api --extra viz`
- `ruff check .`
- `pytest -q`
- `ditri-data-prep --input-path data/sample/incidents_synthetic.csv --output-dir data/processed --seed 42`
- `ditri-demo-showcase --model-path models/devops-incident-triage-smoke --confidence-threshold 0.6 --review-queue sre_manual_triage`

Purpose:

- validate the CLI/reporting path that is most visible in demos and documentation
- ensure showcase artifacts can still be generated after code changes

### Layer 2: API Smoke

Add a lightweight API smoke test in the same CI workflow or a separate parallel job:

- start `uv run ditri-api` with the smoke model
- wait for the server to become reachable
- call:
  - `GET /health`
  - `POST /predict`
  - `GET /metrics`
- fail if any request returns a non-200 response or malformed payload

Purpose:

- validate the server startup path
- confirm the public API contract still works after refactors

### Layer 3: Container Build Smoke

Add a `docker build` smoke job, but make a deliberate choice about when it runs:

Recommended default:

- run on merges to `develop`, release branches, and manual dispatch
- do not require it on every PR unless build time remains acceptable

Reason:

- it adds clear delivery value
- it is more expensive than lint/unit/smoke commands
- it is useful, but not always worth paying on every small PR

### Layer 4: Release-Oriented Checks

Reserve heavier workflows for release branches, tags, or manual dispatch:

- optional benchmark/report regeneration
- optional showcase artifact refresh
- optional Hugging Face publish dry run or gated publish

Purpose:

- keep the everyday developer loop fast
- still demonstrate disciplined release engineering

## Workflow Structure Recommendation

Keep one main CI workflow and add jobs incrementally:

1. `lint-and-test`
2. `showcase-smoke`
3. `api-smoke`
4. `docker-smoke` (conditional trigger)

This is preferable to one very large monolithic job because:

- failures are easier to localize
- parallel jobs reduce total wall-clock time
- it reads more clearly as a portfolio artifact

## Implementation Notes

### Showcase Smoke

Use the already committed smoke model:

- `models/devops-incident-triage-smoke`

Recommended assertions:

- command exits successfully
- `reports/demo_showcase.json` is created
- `reports/demo_showcase.md` is created

Prefer writing outputs into a temporary CI directory when possible so the repository workspace stays clean.

### API Smoke

Recommended environment:

- `MODEL_PATH=models/devops-incident-triage-smoke`
- `CONFIDENCE_THRESHOLD=0.6`
- `REVIEW_QUEUE=sre_manual_triage`

Recommended checks:

- `/health` contains an expected status field
- `/predict` returns a label/confidence payload
- `/metrics` returns text content with known metric names

### Docker Smoke

Recommended scope:

- `docker build -t devops-incident-triage:ci .`

Do not run the full containerized inference path in the first iteration unless build performance remains acceptable.

## Risks And Tradeoffs

### CI Duration Growth

Adding showcase and API smoke checks will increase CI time.
This is acceptable if the workflows remain in the low-minutes range.

### Flaky Server Boot Checks

API smoke can become flaky if startup synchronization is weak.
Use a simple bounded retry loop before making failure calls.

### Artifact Noise

Commands that write reports into the repo root can dirty the workspace.
Use temp output paths in CI where practical.

## Recommendation

Implement the CI hardening in this order:

1. showcase smoke
2. API smoke
3. Docker build smoke

This sequence gives the best value per added complexity and aligns with the repository's current maturity.

## Current Decision

As of 2026-04-15:

- showcase smoke should be part of the default CI workflow
- API smoke should be part of the default CI workflow
- Docker build smoke should not run on every PR yet

Recommended Docker trigger scope for the next phase:

- pushes to `develop`
- release branches
- manual dispatch

## Definition Of Done

The CI hardening effort can be considered complete for this phase when:

- showcase generation is validated in GitHub Actions
- FastAPI startup and basic endpoints are smoke-tested
- a documented decision exists for Docker build smoke scope
- the workflow remains fast enough for normal pull request usage
