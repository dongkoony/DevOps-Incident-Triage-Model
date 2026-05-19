# release-2026.06-rag-preview

## Channel

Preview

## Summary

This release train entry adds the first RAG retrieval implementation for the DevOps Incident Triage project.

The release keeps the system classifier-first. It adds a lightweight domain-aware retriever that loads local runbooks, builds a scikit-learn TF-IDF sparse vector index, and exposes cited evidence through `POST /retrieve`.

This is not a full incident assistant. `/assist`, LLM response generation, production Vector DB deployment, and cloud retrieval infrastructure remain future work.

## Scope

Included:

- Runbook document loading from `docs/runbooks/`
- Section-level retrieval chunks
- Domain label mapping for existing classifier labels
- Local TF-IDF sparse vector index
- Domain-aware ranking boost
- `POST /retrieve` FastAPI endpoint
- Evidence response schema with document ID, domain, title, section, score, citation, and excerpt
- Retrieval preview metadata including embedding/index type and latency
- Unit and API tests for retrieval behavior
- RAG evaluation plan

Excluded:

- Production Vector DB
- Managed embedding service
- `/assist` API
- LLM-generated remediation guidance
- Automatic remediation execution
- Claims about real production incident accuracy

## API Example

```json
{
  "text": "EKS worker nodes became NotReady after a CNI upgrade.",
  "predicted_domain": "k8s_cluster",
  "top_k": 5
}
```

Response evidence items include:

```json
{
  "document_id": "runbook-kubernetes",
  "domain": "k8s_cluster",
  "title": "Kubernetes Cluster Runbook",
  "section": "First Checks",
  "score": 0.83,
  "citation": "docs/runbooks/kubernetes.md#first-checks",
  "excerpt": "Check node readiness and recent node events."
}
```

## Validation Plan

Run before release:

```powershell
uv run --extra dev --extra api ruff check .
uv run --extra dev --extra api pytest -q
```

Required evidence before promotion:

- Retrieval unit tests pass
- API tests for `/retrieve` pass
- Existing classifier API tests pass
- README and RAG roadmap describe preview scope accurately

## Known Limitations

- TF-IDF is a lexical preview index, not a production embedding model.
- The corpus is currently limited to portfolio runbook placeholders.
- Results depend on caller-provided `predicted_domain`.
- No answer generation or hallucination checks are implemented yet.
- The current public dataset remains synthetic.
