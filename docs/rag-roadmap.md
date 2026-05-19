# RAG Roadmap

## Current State

The project currently provides a Transformer-based DevOps incident classifier. It accepts incident summaries, deployment failures, and operational messages, then predicts a first-pass routing domain such as `k8s_cluster`, `cicd_pipeline`, `aws_iam_network`, `deployment_release`, `container_runtime`, `observability_alerting`, or `database_state`.

The current implementation includes CLI inference, FastAPI serving, batch prediction, async batch jobs, evaluation reports, Docker, CI, and release workflow documentation. It does not yet implement a RAG backend.

The current public starter dataset is synthetic, so this roadmap treats the classifier as a reproducible engineering baseline rather than a validated production model.

## Target State

The target direction is a Classifier + RAG + LLM DevOps Incident Triage Assistant. The classifier narrows the operational domain, retrieval finds relevant evidence, and an LLM generates remediation guidance grounded in cited runbooks or historical troubleshooting material.

```text
Incident Text
↓
Incident Classifier
↓
Predicted Domain
↓
Domain-aware Retriever
↓
Runbooks / Historical Incidents / Troubleshooting Docs
↓
LLM Response Generator
↓
Evidence-grounded Remediation Guidance
```

## Why Classifier + RAG Is Better Than RAG-Only

A RAG-only assistant must search across all operational knowledge for every request. That can increase latency, add irrelevant context, and make it harder to explain why a runbook was selected.

The classifier provides an initial domain prior. That makes retrieval more focused:

- Kubernetes incidents can prioritize Kubernetes runbooks and cluster troubleshooting docs.
- CI/CD incidents can prioritize pipeline, deployment, and runner documentation.
- AWS IAM/network incidents can prioritize identity, permission, VPC, and routing material.
- Database incidents can prioritize connection, lock, replication, and storage checks.

This does not replace retrieval ranking. It gives retrieval a safer starting point and preserves human review when confidence is low.

## Proposed Directories

Runbook placeholders:

- `docs/runbooks/kubernetes.md`
- `docs/runbooks/cicd.md`
- `docs/runbooks/aws-iam-network.md`
- `docs/runbooks/database.md`
- `docs/runbooks/observability.md`
- `docs/runbooks/container-runtime.md`
- `docs/runbooks/deployment-release.md`

Future implementation directories may include:

- `src/devops_incident_triage/retrieval.py`
- `src/devops_incident_triage/assist.py`
- `tests/test_retrieval.py`
- `tests/test_assist.py`

These implementation files are intentionally not created in this documentation-only task.

## Proposed APIs

### `POST /retrieve`

Purpose:

Retrieve evidence documents relevant to an incident and predicted domain.

Example request:

```json
{
  "text": "EKS worker nodes became NotReady after a CNI upgrade.",
  "predicted_domain": "k8s_cluster",
  "top_k": 5
}
```

Example response shape:

```json
{
  "predicted_domain": "k8s_cluster",
  "retrieval_query": "EKS worker nodes NotReady CNI upgrade pods pending",
  "evidence": [
    {
      "document_id": "runbook-kubernetes",
      "title": "Kubernetes Cluster Runbook",
      "section": "First Checks",
      "score": 0.83,
      "citation": "docs/runbooks/kubernetes.md#first-checks",
      "excerpt": "Check node readiness, recent CNI changes, pod scheduling events, and kubelet status."
    }
  ]
}
```

### `POST /assist`

Purpose:

Generate evidence-grounded triage guidance by combining classifier output, retrieved documents, and an LLM response generator.

Example request:

```json
{
  "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
  "confidence_threshold": 0.6,
  "top_k": 5
}
```

Example response schema:

```json
{
  "incident": {
    "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
    "predicted_domain": "aws_iam_network",
    "classifier_confidence": 0.82,
    "needs_human_review": false
  },
  "retrieval": {
    "query": "GitHub Actions assume production IAM role denied",
    "evidence": [
      {
        "document_id": "runbook-aws-iam-network",
        "title": "AWS IAM And Network Runbook",
        "citation": "docs/runbooks/aws-iam-network.md#first-checks",
        "score": 0.88,
        "excerpt": "Verify trust policy, OIDC provider, role ARN, and sts:AssumeRole permissions."
      }
    ]
  },
  "assistant_response": {
    "summary": "The failure is likely related to IAM role assumption during deployment.",
    "root_cause_candidates": [
      "GitHub OIDC provider trust relationship changed",
      "Deployment role ARN or audience condition is incorrect",
      "The workflow lacks sts:AssumeRole permission"
    ],
    "recommended_actions": [
      "Check the IAM role trust policy for the GitHub Actions OIDC provider.",
      "Verify the workflow uses the expected role ARN and branch condition.",
      "Review recent IAM policy or environment protection changes."
    ],
    "citations": [
      "docs/runbooks/aws-iam-network.md#first-checks"
    ]
  },
  "metadata": {
    "retrieval_latency_ms": 42,
    "generation_latency_ms": 780,
    "model_version": "classifier-core",
    "rag_enabled": true
  }
}
```

## RAG Evaluation Metrics

| Metric | Meaning |
|---|---|
| `retrieval_hit_rate` | Percentage of incidents where the expected supporting document appears in the top-k results |
| `groundedness_score` | Degree to which generated guidance is supported by retrieved evidence |
| `citation_coverage` | Percentage of recommendations that include at least one citation |
| `hallucination_flag_rate` | Percentage of assistant responses flagged for unsupported claims |
| `retrieval_latency_ms` | Time spent in embedding lookup and retrieval |
| `generation_latency_ms` | Time spent generating the assistant response |

## Non-Goals For The Current Task

- No Vector DB is installed.
- No embedding model is selected in code.
- No `/retrieve` or `/assist` endpoint is implemented.
- No LLM integration is added.
- Existing classifier-focused implementation remains intact.
