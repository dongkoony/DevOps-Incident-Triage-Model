# RAG Preview Retrieval Design

## Goal

Implement the first `release-2026.06-rag-preview` retrieval layer without turning the project into a full LLM assistant yet.

This release should prove that incident text can be routed through the existing classifier domain model into a domain-aware retrieval step that returns cited runbook evidence.

## Release Scope

Included:

- Load existing runbook Markdown files under `docs/runbooks/` as the retrieval corpus.
- Build a lightweight local vector index using scikit-learn TF-IDF.
- Bias retrieval toward the predicted DevOps domain while still allowing cross-domain evidence when useful.
- Add a `POST /retrieve` FastAPI endpoint.
- Return evidence entries with document ID, domain, title, section, score, citation, and excerpt.
- Add unit and API tests for retrieval behavior.
- Add RAG evaluation planning documentation for preview metrics.
- Update README, README.ko, CHANGELOG, and release documentation for `release-2026.06-rag-preview`.

Excluded:

- No production Vector DB deployment.
- No embedding service.
- No `/assist` endpoint.
- No LLM response generation.
- No automatic remediation execution.
- No claim of real production incident generalization.

## Architecture

The preview retriever is intentionally local and deterministic:

```text
Incident Text
↓
Existing classifier or caller-provided predicted domain
↓
TF-IDF query vector
↓
Runbook section corpus
↓
Domain-aware score adjustment
↓
Cited evidence response
```

The retriever indexes Markdown sections rather than entire files. This gives citations that point to useful anchors such as `docs/runbooks/kubernetes.md#first-checks`.

## Components

### `src/devops_incident_triage/retrieval.py`

Responsibilities:

- Discover runbook Markdown files.
- Parse section headings and text.
- Map runbook files to classifier domain labels.
- Build a TF-IDF matrix from section text.
- Retrieve top-k evidence for a query.
- Apply a small domain boost when evidence belongs to the predicted domain.
- Return typed dataclasses that the API can serialize through Pydantic models.

### `src/devops_incident_triage/api.py`

Responsibilities:

- Add request/response schemas for `POST /retrieve`.
- Validate `top_k`.
- Call the retriever with incident text and predicted domain.
- Expose retrieval latency and metadata in the response.
- Keep existing `/predict`, batch, async, health, and metrics behavior intact.

### `tests/test_retrieval.py`

Responsibilities:

- Verify runbook loading.
- Verify section-level citations.
- Verify domain-aware retrieval prioritizes the predicted domain.
- Verify top-k limits.

### `tests/test_api.py`

Responsibilities:

- Verify `POST /retrieve` response shape.
- Verify invalid `top_k` is rejected by schema validation.
- Verify evidence includes citation and metadata.

### Documentation

Responsibilities:

- Document the preview nature of the TF-IDF retriever.
- Document retrieval metrics and limitations.
- Keep RAG described as preview, not production-ready.

## API Contract

Request:

```json
{
  "text": "EKS worker nodes became NotReady after a CNI upgrade.",
  "predicted_domain": "k8s_cluster",
  "top_k": 5
}
```

Response:

```json
{
  "predicted_domain": "k8s_cluster",
  "retrieval_query": "EKS worker nodes became NotReady after a CNI upgrade.",
  "evidence": [
    {
      "document_id": "runbook-kubernetes",
      "domain": "k8s_cluster",
      "title": "Kubernetes Cluster Runbook",
      "section": "First Checks",
      "score": 0.83,
      "citation": "docs/runbooks/kubernetes.md#first-checks",
      "excerpt": "Check node readiness and recent node events."
    }
  ],
  "metadata": {
    "embedding_model": "scikit-learn-tfidf-preview",
    "index_type": "in_memory_sparse_vector_index",
    "rag_enabled": true,
    "retrieval_latency_ms": 12
  }
}
```

## Error Handling

- Missing runbook directory raises a clear retrieval configuration error.
- Empty runbook corpus raises a clear retrieval configuration error.
- Unknown predicted domains are allowed, but they do not receive a domain boost.
- `top_k` is constrained by the API schema.

## Testing Strategy

- Write retrieval unit tests before implementation.
- Confirm retrieval tests fail because `retrieval.py` does not exist yet.
- Implement the smallest retriever that passes the tests.
- Add API tests for `/retrieve`.
- Run focused tests first, then full `ruff` and `pytest`.

## Success Criteria

- `POST /retrieve` returns cited runbook evidence.
- Retrieval prefers domain-matching runbooks for domain-specific incidents.
- Existing classifier API behavior remains unchanged.
- Documentation clearly says this is a preview retrieval layer, not a full RAG assistant.
- `ruff` and `pytest` pass.
