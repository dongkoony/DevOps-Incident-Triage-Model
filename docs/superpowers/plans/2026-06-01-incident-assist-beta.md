# Incident Assist Beta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `POST /assist` beta endpoint that combines classifier output, runbook retrieval evidence, and citation-grounded remediation guidance.

**Architecture:** The API calls the existing classifier path, retrieves runbook evidence for the predicted domain, then delegates response composition to a focused `assist.py` module. The beta assistant does not call an external LLM; it returns deterministic, testable guidance with citations and safety notes.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, prometheus-client, pytest, scikit-learn TF-IDF retrieval.

---

## File Structure

- Create `src/devops_incident_triage/assist.py`: domain dataclasses and deterministic assistant response builder.
- Create `tests/test_assist.py`: unit tests for assistant response composition.
- Modify `src/devops_incident_triage/api.py`: Pydantic schemas, metrics, and `POST /assist`.
- Modify `tests/test_api.py`: API tests for `/assist` response, human-review propagation, validation, and metrics.
- Modify `docs/rag-roadmap.md`: mark `/assist` as beta implementation work.
- Modify `CHANGELOG.md`: record `release-2026.07-incident-assist-beta` implementation progress.
- Create `docs/releases/release-2026.07-incident-assist-beta.md`: release evidence and scope.
- Modify `README.md` and `README.ko.md`: concise beta roadmap update.
- Modify `docs/codex/session_state.md` and `docs/codex/worklog.md`: cross-device handoff notes.

## Task 1: Assistant Response Builder

**Files:**
- Create: `src/devops_incident_triage/assist.py`
- Create: `tests/test_assist.py`

- [ ] **Step 1: Write the failing unit test**

```python
from devops_incident_triage.assist import build_assist_response
from devops_incident_triage.retrieval import RetrievedEvidence, RetrievalResponse


def test_build_assist_response_grounds_actions_in_evidence() -> None:
    retrieval = RetrievalResponse(
        predicted_domain="aws_iam_network",
        retrieval_query="GitHub Actions cannot assume production role.",
        evidence=[
            RetrievedEvidence(
                document_id="runbook-aws-iam-network",
                domain="aws_iam_network",
                title="AWS IAM And Network Runbook",
                section="First Checks",
                score=0.91,
                citation="docs/runbooks/aws-iam-network.md#first-checks",
                excerpt="Verify trust policy, OIDC provider, role ARN, and sts:AssumeRole.",
            )
        ],
        metadata={
            "embedding_model": "scikit-learn-tfidf-preview",
            "index_type": "in_memory_sparse_vector_index",
            "rag_enabled": True,
            "retrieval_latency_ms": 12.3,
        },
    )

    response = build_assist_response(
        incident_text="GitHub Actions cannot assume production role.",
        predicted_domain="aws_iam_network",
        classifier_confidence=0.82,
        needs_human_review=False,
        recommended_queue="aws_iam_network",
        retrieval=retrieval,
    )

    assert response.incident.predicted_domain == "aws_iam_network"
    assert response.incident.needs_human_review is False
    assert response.retrieval.evidence[0].citation == "docs/runbooks/aws-iam-network.md#first-checks"
    assert response.assistant_response.citations == ["docs/runbooks/aws-iam-network.md#first-checks"]
    assert response.assistant_response.recommended_actions[0].citation == "docs/runbooks/aws-iam-network.md#first-checks"
    assert response.metadata.llm_enabled is False
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run --extra dev --extra api pytest tests/test_assist.py -q`

Expected: fails because `devops_incident_triage.assist` does not exist.

- [ ] **Step 3: Implement the minimal builder**

Create dataclasses for `AssistIncident`, `AssistRetrieval`, `RecommendedAction`,
`AssistAssistantResponse`, `AssistMetadata`, and `AssistResponse`. Implement
`build_assist_response()` so every recommended action has a citation from retrieved evidence.

- [ ] **Step 4: Run the unit test and verify GREEN**

Run: `uv run --extra dev --extra api pytest tests/test_assist.py -q`

Expected: `1 passed`.

## Task 2: `/assist` API Endpoint

**Files:**
- Modify: `src/devops_incident_triage/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_assist_returns_classifier_retrieval_and_guidance(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    response = client.post(
        "/assist",
        json={
            "text": "EKS worker nodes became NotReady after a CNI upgrade.",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["incident"]["predicted_domain"] == "k8s_cluster"
    assert payload["retrieval"]["evidence"][0]["citation"].startswith("docs/runbooks/kubernetes.md#")
    assert payload["assistant_response"]["citations"]
    assert payload["metadata"]["assistant_mode"] == "deterministic_beta"
    assert payload["metadata"]["llm_enabled"] is False


def test_assist_preserves_human_review_route(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    response = client.post(
        "/assist",
        json={"text": "Ambiguous failure in mixed logs.", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["incident"]["needs_human_review"] is True
    assert payload["incident"]["recommended_queue"] == "manual_triage"
```

- [ ] **Step 2: Run the API tests and verify RED**

Run: `uv run --extra dev --extra api pytest tests/test_api.py::test_assist_returns_classifier_retrieval_and_guidance tests/test_api.py::test_assist_preserves_human_review_route -q`

Expected: fails with 404 for `/assist`.

- [ ] **Step 3: Implement schemas, metrics, and endpoint**

Add `AssistRequest`, nested response Pydantic models, `ASSIST_REQUESTS_TOTAL`, and
`ASSIST_LATENCY_SECONDS`. The endpoint should call `_predict()`, retrieve evidence with
`RunbookRetriever`, call `build_assist_response()`, and convert dataclasses into Pydantic
responses.

- [ ] **Step 4: Run the API tests and verify GREEN**

Run: `uv run --extra dev --extra api pytest tests/test_api.py::test_assist_returns_classifier_retrieval_and_guidance tests/test_api.py::test_assist_preserves_human_review_route -q`

Expected: `2 passed`.

## Task 3: Assistant Metrics Test

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing metrics test**

```python
def test_metrics_exposes_assist_counters(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    assist_response = client.post(
        "/assist",
        json={
            "text": "EKS worker nodes became NotReady after a CNI upgrade.",
            "top_k": 2,
        },
    )
    metrics_response = client.get("/metrics")

    assert assist_response.status_code == 200
    assert metrics_response.status_code == 200
    metrics_payload = metrics_response.text
    assert 'ditri_assist_requests_total{predicted_domain="k8s_cluster",route="auto_route"}' in metrics_payload
    assert 'ditri_assist_latency_seconds_count{predicted_domain="k8s_cluster",route="auto_route"}' in metrics_payload
```

- [ ] **Step 2: Run the metrics test and verify RED or GREEN**

Run: `uv run --extra dev --extra api pytest tests/test_api.py::test_metrics_exposes_assist_counters -q`

Expected before metrics implementation: fail because metrics are absent. Expected after Task 2 metrics implementation: pass.

- [ ] **Step 3: Adjust metrics implementation if needed**

Ensure labels are exactly `predicted_domain` and `route`, with route values `auto_route`
or `human_review`.

## Task 4: Documentation And Release Evidence

**Files:**
- Modify: `docs/rag-roadmap.md`
- Modify: `CHANGELOG.md`
- Create: `docs/releases/release-2026.07-incident-assist-beta.md`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/codex/session_state.md`
- Modify: `docs/codex/worklog.md`

- [ ] **Step 1: Update docs after code tests pass**

Document that `/assist` is a deterministic beta assistant, not an LLM-powered stable
assistant. Include current limitations and validation commands.

- [ ] **Step 2: Run Markdown/link sanity checks through repository tests**

Run: `uv run --extra dev --extra api pytest -q`

Expected: all tests pass.

## Task 5: Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run lint**

Run: `uv run --extra dev --extra api ruff check .`

Expected: `All checks passed!`

- [ ] **Step 2: Run full test suite**

Run: `uv run --extra dev --extra api pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Review git diff**

Run: `git diff --stat`

Expected: changes are limited to assist implementation, tests, docs, and Codex handoff notes.

- [ ] **Step 4: Commit**

```bash
git add src/devops_incident_triage/assist.py src/devops_incident_triage/api.py tests/test_assist.py tests/test_api.py docs/rag-roadmap.md CHANGELOG.md docs/releases/release-2026.07-incident-assist-beta.md README.md README.ko.md docs/codex/session_state.md docs/codex/worklog.md docs/superpowers/plans/2026-06-01-incident-assist-beta.md
git commit -m "feat: add incident assist beta endpoint"
```
