from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.assist import AssistResponse as DomainAssistResponse
from devops_incident_triage.assist import build_assist_response
from devops_incident_triage.retrieval import (
    DEFAULT_RUNBOOK_DIR,
    RetrievalConfigurationError,
    RunbookRetriever,
)
from devops_incident_triage.triage_policy import decide_triage, validate_confidence_threshold

MODEL_PATH = os.getenv("MODEL_PATH", "models/devops-incident-triage")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
REVIEW_QUEUE = os.getenv("REVIEW_QUEUE", "manual_triage")
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "32"))

logger = logging.getLogger(__name__)


def _normalize_model_ref(model_ref: str | Path) -> str:
    return str(model_ref).replace("\\", "/")


def _is_local_model_ref(model_ref: str | Path) -> bool:
    return Path(_normalize_model_ref(model_ref)).exists()

HTTP_REQUESTS_TOTAL = Counter(
    "ditri_http_requests_total",
    "Total number of HTTP requests handled by the API.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ditri_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
PREDICTION_REQUESTS_TOTAL = Counter(
    "ditri_prediction_requests_total",
    "Total number of prediction endpoint calls.",
    ["endpoint"],
)
PREDICTION_FAILURES_TOTAL = Counter(
    "ditri_prediction_failures_total",
    "Total number of prediction endpoint failures.",
    ["endpoint"],
)
TRIAGE_DECISIONS_TOTAL = Counter(
    "ditri_triage_decisions_total",
    "Total number of auto-route vs human-review decisions.",
    ["route"],
)
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "ditri_retrieval_requests_total",
    "Total number of retrieval endpoint calls.",
    ["predicted_domain"],
)
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "ditri_retrieval_latency_seconds",
    "Runbook retrieval latency in seconds.",
    ["predicted_domain"],
)
ASSIST_REQUESTS_TOTAL = Counter(
    "ditri_assist_requests_total",
    "Total number of assistant endpoint calls.",
    ["predicted_domain", "route"],
)
ASSIST_LATENCY_SECONDS = Histogram(
    "ditri_assist_latency_seconds",
    "Assistant endpoint latency in seconds.",
    ["predicted_domain", "route"],
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    validate_confidence_threshold(CONFIDENCE_THRESHOLD)
    if _is_local_model_ref(MODEL_PATH):
        _load_artifacts()
    yield


app = FastAPI(
    title="DevOps Incident Triage API",
    description="Classify DevOps incident text into triage categories.",
    version="0.3.0",
    lifespan=lifespan,
)

_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_id2label: dict[int, str] | None = None


@app.middleware("http")
async def request_id_and_metrics_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response: Response | None = None

    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started_at
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status_code="500",
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=request.url.path).observe(
            elapsed
        )
        logger.exception(
            "Unhandled API exception at %s request_id=%s",
            request.url.path,
            request_id,
        )
        raise

    elapsed = time.perf_counter() - started_at
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=request.url.path,
    ).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Incident summary or error log text.",
    )


class ScoreItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    predicted_label: str
    final_label: str
    needs_human_review: bool
    recommended_queue: str
    confidence: float
    confidence_threshold: float
    scores: list[ScoreItem]


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of incident summaries or error log texts.",
    )


class BatchPredictResponse(BaseModel):
    total: int
    auto_route_count: int
    human_review_count: int
    predictions: list[PredictResponse]


class RetrieveRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Incident summary or error log text to retrieve evidence for.",
    )
    predicted_domain: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Classifier domain label used to bias retrieval.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of evidence sections to return.",
    )


class RetrievedEvidenceItem(BaseModel):
    document_id: str
    domain: str
    title: str
    section: str
    score: float
    citation: str
    excerpt: str


class RetrieveMetadata(BaseModel):
    embedding_model: str
    index_type: str
    rag_enabled: bool
    retrieval_latency_ms: float


class RetrieveResponse(BaseModel):
    predicted_domain: str
    retrieval_query: str
    evidence: list[RetrievedEvidenceItem]
    metadata: RetrieveMetadata


class AssistRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Incident summary or error log text to classify and assist.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of evidence sections to use for guidance.",
    )


class AssistIncidentItem(BaseModel):
    text: str
    predicted_domain: str
    classifier_confidence: float
    needs_human_review: bool
    recommended_queue: str


class AssistRetrievalItem(BaseModel):
    query: str
    evidence: list[RetrievedEvidenceItem]


class RecommendedActionItem(BaseModel):
    action: str
    citation: str


class AssistAssistantResponseItem(BaseModel):
    summary: str
    root_cause_candidates: list[str]
    recommended_actions: list[RecommendedActionItem]
    citations: list[str]
    safety_notes: list[str]


class AssistMetadataItem(BaseModel):
    assistant_mode: str
    rag_enabled: bool
    llm_enabled: bool
    retrieval_latency_ms: float
    generation_latency_ms: float


class AssistResponse(BaseModel):
    incident: AssistIncidentItem
    retrieval: AssistRetrievalItem
    assistant_response: AssistAssistantResponseItem
    metadata: AssistMetadataItem


def _load_artifacts() -> None:
    global _model, _tokenizer, _id2label
    model_ref = _normalize_model_ref(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(model_ref)
    _tokenizer = AutoTokenizer.from_pretrained(model_ref)
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}
    _model.eval()


def _ensure_loaded() -> None:
    if _model is None or _tokenizer is None or _id2label is None:
        _load_artifacts()


def _predict(text: str) -> PredictResponse:
    _ensure_loaded()
    assert _model is not None
    assert _tokenizer is not None
    assert _id2label is not None

    encoded = _tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    ordered_scores = [
        ScoreItem(label=_id2label[i], score=float(probs[i])) for i in range(len(_id2label))
    ]
    ordered_scores = sorted(ordered_scores, key=lambda x: x.score, reverse=True)
    score_map = {item.label: item.score for item in ordered_scores}
    triage = decide_triage(
        score_map,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        review_queue=REVIEW_QUEUE,
    )
    return PredictResponse(
        label=triage["final_label"],
        predicted_label=triage["predicted_label"],
        final_label=triage["final_label"],
        needs_human_review=triage["needs_human_review"],
        recommended_queue=triage["recommended_queue"],
        confidence=triage["confidence"],
        confidence_threshold=triage["confidence_threshold"],
        scores=ordered_scores,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_path": _normalize_model_ref(MODEL_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "review_queue": REVIEW_QUEUE,
        "batch_max_items": BATCH_MAX_ITEMS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, http_request: Request) -> PredictResponse:
    PREDICTION_REQUESTS_TOTAL.labels(endpoint="/predict").inc()
    try:
        prediction = _predict(request.text)
    except Exception as exc:  # pragma: no cover - runtime protection
        PREDICTION_FAILURES_TOTAL.labels(endpoint="/predict").inc()
        logger.exception(
            "Prediction failed at /predict request_id=%s",
            getattr(http_request.state, "request_id", "unknown"),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    route = "human_review" if prediction.needs_human_review else "auto_route"
    TRIAGE_DECISIONS_TOTAL.labels(route=route).inc()
    return prediction


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest, http_request: Request) -> BatchPredictResponse:
    PREDICTION_REQUESTS_TOTAL.labels(endpoint="/predict/batch").inc()
    if len(request.texts) > BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"`texts` can contain at most {BATCH_MAX_ITEMS} items.",
        )

    predictions: list[PredictResponse] = []
    try:
        for idx, text in enumerate(request.texts):
            text_length = len(text)
            if text_length < 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"`texts[{idx}]` must be at least 5 characters.",
                )
            if text_length > 5000:
                raise HTTPException(
                    status_code=400,
                    detail=f"`texts[{idx}]` must be at most 5000 characters.",
                )
            predictions.append(_predict(text))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime protection
        PREDICTION_FAILURES_TOTAL.labels(endpoint="/predict/batch").inc()
        logger.exception(
            "Prediction failed at /predict/batch request_id=%s",
            getattr(http_request.state, "request_id", "unknown"),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    human_review_count = sum(1 for item in predictions if item.needs_human_review)
    auto_route_count = len(predictions) - human_review_count
    TRIAGE_DECISIONS_TOTAL.labels(route="human_review").inc(human_review_count)
    TRIAGE_DECISIONS_TOTAL.labels(route="auto_route").inc(auto_route_count)
    return BatchPredictResponse(
        total=len(predictions),
        auto_route_count=auto_route_count,
        human_review_count=human_review_count,
        predictions=predictions,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    started_at = time.perf_counter()
    try:
        retrieval = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR).retrieve(
            text=request.text,
            predicted_domain=request.predicted_domain,
            top_k=request.top_k,
        )
    except RetrievalConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        RETRIEVAL_REQUESTS_TOTAL.labels(predicted_domain=request.predicted_domain).inc()
        RETRIEVAL_LATENCY_SECONDS.labels(predicted_domain=request.predicted_domain).observe(
            time.perf_counter() - started_at
        )

    return RetrieveResponse(
        predicted_domain=retrieval.predicted_domain,
        retrieval_query=retrieval.retrieval_query,
        evidence=[
            RetrievedEvidenceItem(
                document_id=item.document_id,
                domain=item.domain,
                title=item.title,
                section=item.section,
                score=item.score,
                citation=item.citation,
                excerpt=item.excerpt,
            )
            for item in retrieval.evidence
        ],
        metadata=RetrieveMetadata(**retrieval.metadata),
    )


@app.post("/assist", response_model=AssistResponse)
def assist(request: AssistRequest, http_request: Request) -> AssistResponse:
    started_at = time.perf_counter()
    predicted_domain = "unknown"
    route = "unknown"

    try:
        prediction = _predict(request.text)
        predicted_domain = prediction.predicted_label
        route = "human_review" if prediction.needs_human_review else "auto_route"
        retrieval = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR).retrieve(
            text=request.text,
            predicted_domain=predicted_domain,
            top_k=request.top_k,
        )
        assistant_response = build_assist_response(
            incident_text=request.text,
            predicted_domain=predicted_domain,
            classifier_confidence=prediction.confidence,
            needs_human_review=prediction.needs_human_review,
            recommended_queue=prediction.recommended_queue,
            retrieval=retrieval,
        )
    except RetrievalConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime protection
        PREDICTION_FAILURES_TOTAL.labels(endpoint="/assist").inc()
        logger.exception(
            "Assistant generation failed at /assist request_id=%s",
            getattr(http_request.state, "request_id", "unknown"),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ASSIST_REQUESTS_TOTAL.labels(predicted_domain=predicted_domain, route=route).inc()
        ASSIST_LATENCY_SECONDS.labels(predicted_domain=predicted_domain, route=route).observe(
            time.perf_counter() - started_at
        )

    return _to_assist_response(assistant_response)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _to_retrieved_evidence_item(item: Any) -> RetrievedEvidenceItem:
    return RetrievedEvidenceItem(
        document_id=item.document_id,
        domain=item.domain,
        title=item.title,
        section=item.section,
        score=item.score,
        citation=item.citation,
        excerpt=item.excerpt,
    )


def _to_assist_response(response: DomainAssistResponse) -> AssistResponse:
    return AssistResponse(
        incident=AssistIncidentItem(
            text=response.incident.text,
            predicted_domain=response.incident.predicted_domain,
            classifier_confidence=response.incident.classifier_confidence,
            needs_human_review=response.incident.needs_human_review,
            recommended_queue=response.incident.recommended_queue,
        ),
        retrieval=AssistRetrievalItem(
            query=response.retrieval.query,
            evidence=[_to_retrieved_evidence_item(item) for item in response.retrieval.evidence],
        ),
        assistant_response=AssistAssistantResponseItem(
            summary=response.assistant_response.summary,
            root_cause_candidates=response.assistant_response.root_cause_candidates,
            recommended_actions=[
                RecommendedActionItem(action=item.action, citation=item.citation)
                for item in response.assistant_response.recommended_actions
            ],
            citations=response.assistant_response.citations,
            safety_notes=response.assistant_response.safety_notes,
        ),
        metadata=AssistMetadataItem(
            assistant_mode=response.metadata.assistant_mode,
            rag_enabled=response.metadata.rag_enabled,
            llm_enabled=response.metadata.llm_enabled,
            retrieval_latency_ms=response.metadata.retrieval_latency_ms,
            generation_latency_ms=response.metadata.generation_latency_ms,
        ),
    )


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("devops_incident_triage.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
