# RAG Preview Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first RAG preview retrieval layer that loads runbooks, performs domain-aware TF-IDF retrieval, and exposes `POST /retrieve`.

**Architecture:** The retriever parses Markdown runbooks into section-level evidence chunks, builds an in-memory TF-IDF sparse vector index, and applies a small domain boost for the caller-provided predicted domain. FastAPI exposes the retriever through a typed `/retrieve` endpoint while existing classifier endpoints remain unchanged.

**Tech Stack:** Python 3.12, scikit-learn `TfidfVectorizer`, FastAPI, Pydantic, pytest, ruff.

---

## File Map

- Create `src/devops_incident_triage/retrieval.py`: runbook loading, section parsing, TF-IDF index, retrieval ranking, evidence dataclasses.
- Create `tests/test_retrieval.py`: unit tests for corpus loading, citation generation, domain-aware ranking, top-k behavior.
- Modify `src/devops_incident_triage/api.py`: add Pydantic schemas and `POST /retrieve`.
- Modify `tests/test_api.py`: add API tests for `/retrieve`.
- Create `docs/evaluation/rag-evaluation.md`: preview RAG evaluation plan.
- Create `docs/releases/release-2026.06-rag-preview.md`: release evidence scaffold for preview.
- Modify `README.md`, `README.ko.md`, `CHANGELOG.md`, `docs/rag-roadmap.md`: document implemented preview retrieval.

## Task 1: Retrieval Unit Tests

**Files:**

- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests for runbook retrieval behavior.**

The tests should import symbols that do not exist yet:

```python
from pathlib import Path

from devops_incident_triage.retrieval import (
    DEFAULT_RUNBOOK_DIR,
    RunbookRetriever,
    load_runbook_corpus,
)


def test_load_runbook_corpus_reads_section_level_evidence() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    kubernetes = [
        item
        for item in corpus
        if item.document_id == "runbook-kubernetes" and item.section == "First Checks"
    ]

    assert kubernetes
    assert kubernetes[0].domain == "k8s_cluster"
    assert kubernetes[0].citation == "docs/runbooks/kubernetes.md#first-checks"
    assert "node readiness" in kubernetes[0].text.lower()


def test_retriever_prioritizes_predicted_domain_evidence() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="EKS worker nodes became NotReady after a CNI upgrade.",
        predicted_domain="k8s_cluster",
        top_k=3,
    )

    assert response.evidence
    assert response.evidence[0].domain == "k8s_cluster"
    assert response.evidence[0].document_id == "runbook-kubernetes"
    assert response.evidence[0].score > 0


def test_retriever_limits_top_k() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="GitHub Actions runner cannot assume the production IAM role.",
        predicted_domain="aws_iam_network",
        top_k=2,
    )

    assert len(response.evidence) == 2
    assert response.evidence[0].domain == "aws_iam_network"


def test_retriever_response_includes_preview_metadata() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="Database writes are timing out because locks are waiting.",
        predicted_domain="database_state",
        top_k=1,
    )

    assert response.metadata["embedding_model"] == "scikit-learn-tfidf-preview"
    assert response.metadata["index_type"] == "in_memory_sparse_vector_index"
    assert response.metadata["rag_enabled"] is True
```

- [ ] **Step 2: Run tests and verify RED.**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'uv\envs\devops-incident-triage-model'
uv run --extra dev --extra api pytest tests/test_retrieval.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'devops_incident_triage.retrieval'
```

## Task 2: Retrieval Implementation

**Files:**

- Create: `src/devops_incident_triage/retrieval.py`

- [ ] **Step 1: Implement the smallest retriever that passes Task 1.**

Use dataclasses for internal return types and scikit-learn for the preview index.

- [ ] **Step 2: Run focused retrieval tests.**

Run:

```powershell
uv run --extra dev --extra api pytest tests/test_retrieval.py -q
```

Expected:

```text
4 passed
```

## Task 3: API Tests

**Files:**

- Modify: `tests/test_api.py`

- [ ] **Step 1: Add failing `/retrieve` API tests.**

Append tests that call `POST /retrieve` and assert response schema, evidence, metadata, and validation behavior.

- [ ] **Step 2: Run focused API tests and verify RED.**

Run:

```powershell
uv run --extra dev --extra api pytest tests/test_api.py -q
```

Expected:

```text
404 Not Found
```

## Task 4: API Implementation

**Files:**

- Modify: `src/devops_incident_triage/api.py`

- [ ] **Step 1: Add Pydantic request/response models for retrieval.**

Add `RetrieveRequest`, `RetrievedEvidenceItem`, `RetrieveMetadata`, and `RetrieveResponse`.

- [ ] **Step 2: Add `POST /retrieve`.**

The endpoint should call `RunbookRetriever.from_runbook_dir().retrieve(...)` and return cited evidence.

- [ ] **Step 3: Run focused API tests.**

Run:

```powershell
uv run --extra dev --extra api pytest tests/test_api.py -q
```

Expected:

```text
all tests pass
```

## Task 5: Documentation And Release Notes

**Files:**

- Create: `docs/evaluation/rag-evaluation.md`
- Create: `docs/releases/release-2026.06-rag-preview.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/rag-roadmap.md`

- [ ] **Step 1: Document the implemented preview retrieval layer.**

Document that TF-IDF is a preview local vector index, not a production Vector DB.

- [ ] **Step 2: Document preview evaluation metrics.**

Include `retrieval_hit_rate`, `citation_coverage`, `retrieval_latency_ms`, and known limitations.

- [ ] **Step 3: Update release roadmap docs.**

Mark `release-2026.06-rag-preview` as in-progress preview work.

## Task 6: Final Verification

**Files:**

- All changed files

- [ ] **Step 1: Run lint.**

Run:

```powershell
uv run --extra dev --extra api ruff check .
```

Expected:

```text
All checks passed!
```

- [ ] **Step 2: Run full tests.**

Run:

```powershell
uv run --extra dev --extra api pytest -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Commit the completed feature.**

Run:

```powershell
git add src/devops_incident_triage/retrieval.py src/devops_incident_triage/api.py tests/test_retrieval.py tests/test_api.py docs/evaluation/rag-evaluation.md docs/releases/release-2026.06-rag-preview.md README.md README.ko.md CHANGELOG.md docs/rag-roadmap.md docs/superpowers/specs/2026-05-19-rag-preview-retrieval-design.md docs/superpowers/plans/2026-05-19-rag-preview-retrieval.md
git commit -m "feat: add rag preview retrieval layer"
```
