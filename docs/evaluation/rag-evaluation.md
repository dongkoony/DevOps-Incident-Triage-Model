# RAG Evaluation Plan

## Scope

This document defines preview evaluation criteria for `release-2026.06-rag-preview`.

The current implementation uses a local scikit-learn TF-IDF sparse vector index over `docs/runbooks/`. It is intended to validate retrieval contracts, citations, and domain-aware ranking before introducing a production Vector DB or LLM assistant.

## Current Preview Metrics

| Metric | Meaning | Preview Measurement |
|---|---|---|
| `retrieval_hit_rate` | Expected runbook appears in top-k results | Use curated incidents with expected domain/runbook labels |
| `citation_coverage` | Evidence items include citations | Assert every returned evidence item has a non-empty `citation` |
| `retrieval_latency_ms` | Time spent retrieving evidence | Read `metadata.retrieval_latency_ms` from `/retrieve` responses |
| `domain_match_rate` | Top evidence domain matches predicted domain | Compare `evidence[0].domain` with request `predicted_domain` |
| `empty_result_rate` | Retrieval returns no evidence | Should remain zero while runbook corpus is available |

## Future Beta Metrics

These metrics become more important in `release-2026.07-incident-assist-beta` and `release-2026.08-eval-observability`:

- `groundedness_score`
- `hallucination_flag_rate`
- `generation_latency_ms`
- answer-level citation coverage
- human review acceptance rate

## Preview Test Set

Use synthetic, portfolio-safe examples only:

- Kubernetes node readiness or CNI incident
- CI/CD runner or pipeline failure
- AWS IAM role assumption or VPC routing failure
- Database lock, replication, or storage issue
- Observability alerting or logging issue
- Container runtime image pull or containerd issue
- Deployment rollout or Helm release issue

## Known Limitations

- TF-IDF is lexical and may miss semantically related wording.
- The corpus is small and runbook-based.
- The current dataset is synthetic.
- No LLM output is generated in this release.
- No production Vector DB is deployed in this release.
