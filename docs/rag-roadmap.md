# RAG Roadmap

## Current State

The project currently provides a Transformer-based DevOps incident classifier. It accepts incident summaries, deployment failures, and operational messages, then predicts a first-pass routing domain such as `k8s_cluster`, `cicd_pipeline`, `aws_iam_network`, `deployment_release`, `container_runtime`, `observability_alerting`, or `database_state`.

The current implementation includes CLI inference, FastAPI serving, batch prediction, async batch jobs, evaluation reports, Docker, CI, release workflow documentation, a preview retrieval layer over local runbooks, and a deterministic incident-assist beta endpoint.

The retrieval layer uses a scikit-learn TF-IDF sparse vector index for `release-2026.06-rag-preview`. This is intentionally lightweight and local. It proves the evidence retrieval contract before introducing a production Vector DB.

The assistant beta in `release-2026.07-incident-assist-beta` combines classifier output, retrieved evidence, deterministic guidance, citations, and safety notes. It is LLM-ready but does not call an external LLM yet.

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

`src/devops_incident_triage/retrieval.py` is implemented for the preview release. `src/devops_incident_triage/assist.py` and assistant tests are implemented for the incident-assist beta release.

## Proposed APIs

### `POST /retrieve`

Purpose:

Retrieve evidence documents relevant to an incident and predicted domain.

Status: implemented as preview local retrieval using scikit-learn TF-IDF over `docs/runbooks/`.

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

Generate evidence-grounded triage guidance by combining classifier output, retrieved documents, and a deterministic beta response builder.

Status: implemented as a deterministic beta assistant. The response contract is LLM-ready, but no external LLM API call is made in this release.

Example request:

```json
{
  "text": "GitHub Actions deployment failed because the runner could not assume the production IAM role.",
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
        "citation": "docs/runbooks/aws-iam-network.md#first-checks",
        "score": 0.88,
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
    "retrieval_latency_ms": 42,
    "generation_latency_ms": 1.2,
    "assistant_mode": "deterministic_beta",
    "rag_enabled": true,
    "llm_enabled": false
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
- No production embedding model or managed Vector DB is selected in code.
- `/retrieve` is implemented as preview local retrieval.
- `/assist` is implemented as a deterministic beta assistant.
- No external LLM integration is added.
- Existing classifier-focused implementation remains intact.
