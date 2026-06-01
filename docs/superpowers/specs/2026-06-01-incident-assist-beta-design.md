# Incident Assist Beta Design

## Overview

`release-2026.07-incident-assist-beta` introduces the first assistant-style API on top of
the existing classifier and RAG preview retrieval layer. The goal is to make the project
behave like an evidence-grounded DevOps triage assistant without introducing an external
LLM dependency yet.

This is a practical beta step: the service should produce deterministic, testable
guidance from classifier output and retrieved runbook evidence. A future release can
replace or augment the deterministic response builder with an LLM generator while
preserving the API contract, citations, safety flags, and observability.

## Goals

- Add `POST /assist` as the first Classifier + RAG incident assistant API.
- Reuse the existing `_predict()` classifier path to determine domain, confidence, and
  human-review routing.
- Reuse `RunbookRetriever` to retrieve cited runbook evidence for the predicted domain.
- Return structured guidance with summary, root-cause candidates, recommended actions,
  and citations.
- Keep all assistant guidance grounded in retrieved evidence.
- Add Prometheus metrics for assistant request count and latency.
- Keep the implementation deterministic and CI-safe with no external LLM API call.

## Non-Goals

- No production Vector DB integration.
- No managed embedding model integration.
- No OpenAI, Hugging Face Inference, or other hosted LLM call.
- No automatic remediation execution.
- No claim that the synthetic dataset or placeholder runbooks prove production accuracy.

## API Contract

### Request

`POST /assist`

```json
{
  "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
  "top_k": 5
}
```

`text` follows the existing incident text validation: minimum 5 characters, maximum
5000 characters.

`top_k` follows the retrieval API validation: minimum 1, maximum 10, default 5.

### Response

```json
{
  "incident": {
    "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
    "predicted_domain": "aws_iam_network",
    "classifier_confidence": 0.82,
    "needs_human_review": false,
    "recommended_queue": "aws_iam_network"
  },
  "retrieval": {
    "query": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
    "evidence": [
      {
        "document_id": "runbook-aws-iam-network",
        "domain": "aws_iam_network",
        "title": "AWS IAM And Network Runbook",
        "section": "First Checks",
        "score": 0.88,
        "citation": "docs/runbooks/aws-iam-network.md#first-checks",
        "excerpt": "Verify trust policy, OIDC provider, role ARN, and sts:AssumeRole permissions."
      }
    ]
  },
  "assistant_response": {
    "summary": "Initial triage points to the aws_iam_network domain. Review the cited runbook evidence before taking action.",
    "root_cause_candidates": [
      "AWS IAM And Network Runbook / First Checks may explain the incident symptoms."
    ],
    "recommended_actions": [
      {
        "action": "Review AWS IAM And Network Runbook / First Checks and compare it with the current incident timeline.",
        "citation": "docs/runbooks/aws-iam-network.md#first-checks"
      }
    ],
    "citations": [
      "docs/runbooks/aws-iam-network.md#first-checks"
    ],
    "safety_notes": [
      "This beta assistant does not execute remediation actions.",
      "Validate guidance with an operator before changing production systems."
    ]
  },
  "metadata": {
    "assistant_mode": "deterministic_beta",
    "rag_enabled": true,
    "llm_enabled": false,
    "retrieval_latency_ms": 42.0,
    "generation_latency_ms": 1.2
  }
}
```

## Architecture

```text
POST /assist
  |
  v
_predict(text)
  |
  v
PredictResponse(predicted_label, confidence, needs_human_review, recommended_queue)
  |
  v
RunbookRetriever.retrieve(text, predicted_label, top_k)
  |
  v
AssistantResponseBuilder.build(...)
  |
  v
Evidence-grounded AssistResponse
```

## Components

### `src/devops_incident_triage/assist.py`

This module owns assistant response composition. It should not load models, create a
FastAPI app, or call external services.

Responsibilities:

- Define dataclasses for the assistant domain response.
- Convert classifier and retrieval results into assistant guidance.
- Deduplicate citations.
- Add beta safety notes.
- Track deterministic generation latency.

### `src/devops_incident_triage/api.py`

The API module exposes the endpoint and Pydantic schemas.

Responsibilities:

- Add request and response schemas for `POST /assist`.
- Call `_predict()` exactly as `/predict` does.
- Call `RunbookRetriever` exactly as `/retrieve` does.
- Increment assistant metrics in success and failure paths.
- Return HTTP 500 for retrieval configuration errors.

### Tests

Tests should cover behavior, not implementation details.

Required test coverage:

- `build_assist_response()` returns summary, root-cause candidates, recommended actions,
  deduplicated citations, and safety notes from retrieved evidence.
- `POST /assist` returns classifier, retrieval, assistant, and metadata sections.
- `POST /assist` preserves human-review routing when classifier confidence is low.
- `/metrics` exposes assistant request and latency metrics after an assist request.

## Error Handling

- Prediction failures follow the existing `/predict` runtime protection pattern.
- Retrieval corpus configuration failures return HTTP 500 with the existing
  `RetrievalConfigurationError` detail.
- Validation errors use FastAPI/Pydantic 422 responses for invalid `text` or `top_k`.
- No assistant response should recommend automatic execution of production changes.

## Observability

Add these Prometheus metrics:

- `ditri_assist_requests_total{predicted_domain,route}`
- `ditri_assist_latency_seconds{predicted_domain,route}`

The `route` label should be `human_review` when classifier policy requires review and
`auto_route` otherwise.

## Documentation Updates

- Update `docs/rag-roadmap.md` to mark `/assist` as beta implementation work.
- Update `CHANGELOG.md` under `release-2026.07-incident-assist-beta`.
- Create or update `docs/releases/release-2026.07-incident-assist-beta.md`.
- Keep README changes concise and point to detailed release docs.

## Release Positioning

This is a beta release track feature. It is stronger than the RAG preview because an
end-to-end Classifier + Retriever + Assistant flow exists, but it is not stable because
LLM integration, groundedness evaluation, hallucination checks, and production Vector DB
deployment are still future work.
