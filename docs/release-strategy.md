# Release Strategy

## Overview

This project uses a product-style release train to describe how the DevOps Incident Triage Model evolves from a classifier-focused MLOps project into a future RAG-powered DevOps Incident Triage Assistant.

The existing Transformer classifier remains the stable baseline. It routes incident text into operational domains such as Kubernetes, CI/CD, AWS IAM/networking, deployment/release, container runtime, observability, and database state. Future releases add retrieval, runbook evidence, LLM-assisted remediation guidance, evaluation, observability, and cloud deployment planning.

RAG is planned as a future extension. The current backend should still be understood as classifier-first.

The current public starter dataset is synthetic, so model scores should be treated as reproducibility and portfolio evidence rather than validated real-world generalization.

## Why Not Only v1.0.0

Traditional semantic versioning is useful for code compatibility, but it is not expressive enough for this roadmap. The project is not only shipping a Python package. It is also demonstrating model training, API serving, evaluation, release operations, and a planned assistant architecture.

A release train makes the project easier to review because each release name explains:

- the delivery window
- the product capability being advanced
- the maturity channel
- whether a feature is implemented, planned, or production-style

Semantic versions may still be used for package metadata, but roadmap communication should use the release train names below.

## Release Channel Definitions

| Channel | Definition | Promotion expectation |
|---|---|---|
| `experimental` | Early prototype or internal validation. APIs and data contracts may change. | Keep scoped to design spikes, local experiments, or throwaway validation. |
| `preview` | Feature direction is mostly defined and documented, but not production-ready. | Include design docs, schemas, and clear non-goals before implementation. |
| `beta` | Integrated and tested, but still evolving. | Include tests, representative examples, evaluation notes, and known limitations. |
| `stable` | Production-style release with documentation, tests, monitoring, and operational notes. | Require CI coverage, release notes, validation evidence, and rollback or support notes. |

## Release Tag Naming Convention

Release tags use this pattern:

```text
release-YYYY.MM-short-focus
```

Examples:

- `release-2026.05-classifier-core`
- `release-2026.06-rag-preview`
- `release-2026.07-incident-assist-beta`

This format is intentionally descriptive. It avoids implying that a planned RAG assistant is already production-ready just because a numeric version increased.

## Current Release Roadmap

| Release tag | Channel | Focus | Exit criteria |
|---|---|---|---|
| `release-2026.05-classifier-core` | stable | Current Transformer-based DevOps incident classification, FastAPI inference, batch prediction, async batch jobs, evaluation reports, Docker, and CI workflow | Existing classifier workflows remain reproducible and documented. |
| `release-2026.06-rag-preview` | preview | RAG roadmap, runbook document structure, domain-aware retrieval design, embedding and Vector DB selection, `/retrieve` API design, evidence-based retrieval response schema | Retrieval design is documented with placeholder runbooks and schemas. |
| `release-2026.07-incident-assist-beta` | beta | Classifier + RAG integration design, `/assist` API design, LLM response generation design, root cause candidates, recommended remediation actions, evidence citations | Assistant contract is integrated in design and has testable examples. |
| `release-2026.08-eval-observability` | beta | RAG evaluation plan, retrieval hit rate, groundedness checks, hallucination checks, latency metrics, Prometheus-style observability expansion | Evaluation metrics and observability plan are documented and mapped to future implementation points. |
| `release-2026.09-cloud-stable` | stable | AWS deployment roadmap, production-style FastAPI service, Vector DB deployment option, monitoring, CI/CD release flow, operational documentation | Cloud deployment path has operational docs, monitoring plan, and release process. |

## Promotion Rules

### Preview To Beta

A preview release can move to beta when:

- request and response schemas are documented
- directory structure and data contracts are stable enough for implementation
- expected failure modes are listed
- tests or test plans exist for the main user flows
- limitations are explicit

### Beta To Stable

A beta release can move to stable when:

- CI covers the implemented paths
- documentation includes setup, operation, and validation steps
- metrics and logs support troubleshooting
- evaluation criteria are defined and measured
- the release notes explain known limitations honestly
- rollback or fallback behavior is documented

### Stable Maintenance

Stable does not mean production deployment is already running. In this portfolio project, stable means the work is production-style: documented, tested, observable, and clear about its assumptions.

## GitHub Release Notes Template

```markdown
# <release-tag>

## Channel

<experimental | preview | beta | stable>

## Summary

One short paragraph describing what this release adds or stabilizes.

## Added

- Capability or documentation added in this release.

## Changed

- Existing behavior, docs, or workflow updated in this release.

## Validation

- `uv run ruff check .`
- `uv run pytest -q`
- Any relevant smoke commands or documentation checks.

## Known Limitations

- Dataset, model, retrieval, LLM, or deployment limitations.

## Next Release

- The next release train item and the main decision needed before starting it.
```
