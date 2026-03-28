from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.triage_policy import decide_triage, validate_confidence_threshold

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/devops-incident-triage"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
REVIEW_QUEUE = os.getenv("REVIEW_QUEUE", "manual_triage")
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "32"))
BATCH_JOB_MAX_STORED = int(os.getenv("BATCH_JOB_MAX_STORED", "200"))

logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    validate_confidence_threshold(CONFIDENCE_THRESHOLD)
    if MODEL_PATH.exists():
        _load_artifacts()
    yield


app = FastAPI(
    title="DevOps Incident Triage API",
    description="Classify DevOps incident text into triage categories.",
    version="0.2.0",
    lifespan=lifespan,
)

_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_id2label: dict[int, str] | None = None
BatchJobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass(slots=True)
class BatchJobState:
    job_id: str
    texts: list[str]
    status: BatchJobStatus = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    predictions: list[PredictResponse] = field(default_factory=list)
    error: str | None = None


_batch_jobs: dict[str, BatchJobState] = {}
_batch_jobs_lock = Lock()


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


class BatchAsyncCreateResponse(BaseModel):
    job_id: str
    status: BatchJobStatus
    total: int
    status_url: str


class BatchAsyncStatusResponse(BaseModel):
    job_id: str
    status: BatchJobStatus
    total: int
    auto_route_count: int | None = None
    human_review_count: int | None = None
    predictions: list[PredictResponse] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


def _load_artifacts() -> None:
    global _model, _tokenizer, _id2label
    _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
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


def _validate_batch_texts(texts: list[str]) -> None:
    if len(texts) > BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"`texts` can contain at most {BATCH_MAX_ITEMS} items.",
        )

    for idx, text in enumerate(texts):
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


def _cleanup_batch_jobs() -> None:
    if len(_batch_jobs) <= BATCH_JOB_MAX_STORED:
        return
    oldest_jobs = sorted(_batch_jobs.values(), key=lambda item: item.created_at)
    for job in oldest_jobs:
        if len(_batch_jobs) <= BATCH_JOB_MAX_STORED:
            break
        if job.status in {"completed", "failed"}:
            _batch_jobs.pop(job.job_id, None)


def _to_batch_async_status_response(job: BatchJobState) -> BatchAsyncStatusResponse:
    auto_route_count: int | None = None
    human_review_count: int | None = None
    predictions: list[PredictResponse] | None = None

    if job.status == "completed":
        predictions = job.predictions
        human_review_count = sum(1 for item in job.predictions if item.needs_human_review)
        auto_route_count = len(job.predictions) - human_review_count

    return BatchAsyncStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=len(job.texts),
        auto_route_count=auto_route_count,
        human_review_count=human_review_count,
        predictions=predictions,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _run_async_batch_job(job_id: str) -> None:
    with _batch_jobs_lock:
        job = _batch_jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = datetime.now(UTC)
        texts = list(job.texts)

    try:
        predictions = [_predict(text) for text in texts]
    except Exception as exc:  # pragma: no cover - runtime protection
        PREDICTION_FAILURES_TOTAL.labels(endpoint="/predict/batch/async").inc()
        logger.exception("Async batch prediction failed job_id=%s", job_id)
        with _batch_jobs_lock:
            failed_job = _batch_jobs.get(job_id)
            if failed_job is None:
                return
            failed_job.status = "failed"
            failed_job.error = str(exc)
            failed_job.completed_at = datetime.now(UTC)
        return

    human_review_count = sum(1 for item in predictions if item.needs_human_review)
    auto_route_count = len(predictions) - human_review_count
    TRIAGE_DECISIONS_TOTAL.labels(route="human_review").inc(human_review_count)
    TRIAGE_DECISIONS_TOTAL.labels(route="auto_route").inc(auto_route_count)

    with _batch_jobs_lock:
        completed_job = _batch_jobs.get(job_id)
        if completed_job is None:
            return
        completed_job.predictions = predictions
        completed_job.status = "completed"
        completed_job.completed_at = datetime.now(UTC)


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_path": str(MODEL_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "review_queue": REVIEW_QUEUE,
        "batch_max_items": BATCH_MAX_ITEMS,
        "batch_job_max_stored": BATCH_JOB_MAX_STORED,
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
    _validate_batch_texts(request.texts)

    predictions: list[PredictResponse] = []
    try:
        for text in request.texts:
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


@app.post(
    "/predict/batch/async",
    response_model=BatchAsyncCreateResponse,
    status_code=202,
)
def predict_batch_async(
    request: BatchPredictRequest,
    background_tasks: BackgroundTasks,
) -> BatchAsyncCreateResponse:
    PREDICTION_REQUESTS_TOTAL.labels(endpoint="/predict/batch/async").inc()
    _validate_batch_texts(request.texts)

    job_id = str(uuid.uuid4())
    job = BatchJobState(job_id=job_id, texts=list(request.texts))
    with _batch_jobs_lock:
        _batch_jobs[job_id] = job
        _cleanup_batch_jobs()
    background_tasks.add_task(_run_async_batch_job, job_id)

    return BatchAsyncCreateResponse(
        job_id=job_id,
        status=job.status,
        total=len(job.texts),
        status_url=f"/predict/batch/async/{job_id}",
    )


@app.get("/predict/batch/async/{job_id}", response_model=BatchAsyncStatusResponse)
def get_async_batch_job(job_id: str) -> BatchAsyncStatusResponse:
    with _batch_jobs_lock:
        job = _batch_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Batch async job not found: {job_id}")
        return _to_batch_async_status_response(job)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("devops_incident_triage.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
