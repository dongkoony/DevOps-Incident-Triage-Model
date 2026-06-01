# release-2026.07-incident-assist-beta

## Channel

Beta

## Summary

This release train entry adds the first incident-assist beta flow for the DevOps Incident Triage project.

The release keeps the system classifier-first and evidence-grounded. It combines the existing classifier, the local runbook retriever, and a deterministic assistant response builder to return cited triage guidance through `POST /assist`.

This is not a production LLM assistant. The endpoint is LLM-ready, but it does not call an external LLM API and does not execute remediation actions.

## Scope

Included:

- `POST /assist` FastAPI endpoint
- Classifier + RAG retrieval orchestration
- Deterministic assistant response builder
- Incident, retrieval, assistant response, and metadata sections
- Root cause candidates generated from retrieved runbook evidence
- Recommended actions with citations
- Beta safety notes
- Prometheus metrics for assistant request count and latency
- Unit and API tests for assistant behavior

Excluded:

- Hosted LLM response generation
- Production Vector DB
- Managed embedding service
- Automatic remediation execution
- Claims about real production incident accuracy

## API Example

```json
{
  "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
  "top_k": 5
}
```

Response sections include:

```json
{
  "incident": {
    "predicted_domain": "aws_iam_network",
    "classifier_confidence": 0.82,
    "needs_human_review": false,
    "recommended_queue": "aws_iam_network"
  },
  "retrieval": {
    "query": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
    "evidence": [
      {
        "citation": "docs/runbooks/aws-iam-network.md#first-checks",
        "excerpt": "Verify trust policy, OIDC provider, role ARN, and sts:AssumeRole permissions."
      }
    ]
  },
  "assistant_response": {
    "summary": "Initial triage points to the aws_iam_network domain. Review the cited runbook evidence before taking action.",
    "recommended_actions": [
      {
        "action": "Review AWS IAM And Network Runbook / First Checks and compare it with the current incident timeline.",
        "citation": "docs/runbooks/aws-iam-network.md#first-checks"
      }
    ],
    "citations": [
      "docs/runbooks/aws-iam-network.md#first-checks"
    ]
  },
  "metadata": {
    "assistant_mode": "deterministic_beta",
    "rag_enabled": true,
    "llm_enabled": false
  }
}
```

## Validation Evidence

Run from branch `feature/incident-assist-beta` on 2026-06-01.

```powershell
uv run --extra dev --extra api ruff check .
```

Result:

```text
All checks passed!
```

```powershell
uv run --extra dev --extra api pytest -q
```

Result:

```text
53 passed, 10 skipped
```

## Known Limitations

- The assistant response is deterministic and template-based.
- No external LLM generation is included in this beta.
- Guidance quality depends on the current synthetic dataset and placeholder runbook corpus.
- Groundedness, hallucination, and citation coverage evaluation are planned for `release-2026.08-eval-observability`.
- The endpoint is intended for triage support, not autonomous remediation.
