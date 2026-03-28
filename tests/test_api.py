from fastapi.testclient import TestClient

from devops_incident_triage import api as api_module
from devops_incident_triage.triage_policy import HUMAN_REVIEW_LABEL


def _fake_predict(text: str) -> api_module.PredictResponse:
    needs_review = "ambiguous" in text.lower()
    final_label = HUMAN_REVIEW_LABEL if needs_review else "k8s_cluster"
    queue = "manual_triage" if needs_review else "k8s_cluster"
    confidence = 0.41 if needs_review else 0.91
    return api_module.PredictResponse(
        label=final_label,
        predicted_label="k8s_cluster",
        final_label=final_label,
        needs_human_review=needs_review,
        recommended_queue=queue,
        confidence=confidence,
        confidence_threshold=0.6,
        scores=[
            api_module.ScoreItem(label="k8s_cluster", score=0.91),
            api_module.ScoreItem(label="deployment_release", score=0.09),
        ],
    )


def test_predict_batch_success(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    response = client.post(
        "/predict/batch",
        json={"texts": ["Node NotReady after upgrade.", "Ambiguous failure in mixed logs."]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["auto_route_count"] == 1
    assert payload["human_review_count"] == 1
    assert payload["predictions"][0]["needs_human_review"] is False
    assert payload["predictions"][1]["final_label"] == HUMAN_REVIEW_LABEL


def test_predict_batch_respects_max_items(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    monkeypatch.setattr(api_module, "BATCH_MAX_ITEMS", 1)
    client = TestClient(api_module.app)

    response = client.post(
        "/predict/batch",
        json={"texts": ["Node NotReady after upgrade.", "Second event should fail."]},
    )

    assert response.status_code == 400
    assert "at most 1" in response.json()["detail"]


def test_predict_batch_validates_text_length(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    response = client.post("/predict/batch", json={"texts": ["bad"]})

    assert response.status_code == 400
    assert "at least 5 characters" in response.json()["detail"]


def test_health_includes_batch_max_items() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "batch_max_items" in payload
    assert payload["batch_max_items"] >= 1


def test_health_returns_request_id_header() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("x-request-id")


def test_request_id_echoes_client_header() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-123"


def test_metrics_exposes_prediction_counters(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_predict", _fake_predict)
    client = TestClient(api_module.app)

    predict_response = client.post(
        "/predict",
        json={"text": "Ambiguous release failure with timeout and missing permissions."},
    )
    batch_response = client.post(
        "/predict/batch",
        json={"texts": ["Node NotReady after upgrade.", "Ambiguous failure in mixed logs."]},
    )
    metrics_response = client.get("/metrics")

    assert predict_response.status_code == 200
    assert batch_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    metrics_payload = metrics_response.text
    assert 'ditri_prediction_requests_total{endpoint="/predict"}' in metrics_payload
    assert 'ditri_prediction_requests_total{endpoint="/predict/batch"}' in metrics_payload
    assert 'ditri_triage_decisions_total{route="auto_route"}' in metrics_payload
    assert 'ditri_triage_decisions_total{route="human_review"}' in metrics_payload
