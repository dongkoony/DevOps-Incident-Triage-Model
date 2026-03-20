from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.triage_policy import decide_triage, validate_confidence_threshold

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/devops-incident-triage"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
REVIEW_QUEUE = os.getenv("REVIEW_QUEUE", "manual_triage")

app = FastAPI(
    title="DevOps Incident Triage API",
    description="Classify DevOps incident text into triage categories.",
    version="0.1.0",
)

_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_id2label: dict[int, str] | None = None


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


@app.on_event("startup")
def startup_event() -> None:
    validate_confidence_threshold(CONFIDENCE_THRESHOLD)
    if MODEL_PATH.exists():
        _load_artifacts()


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_path": str(MODEL_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "review_queue": REVIEW_QUEUE,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return _predict(request.text)
    except Exception as exc:  # pragma: no cover - runtime protection
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("devops_incident_triage.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
