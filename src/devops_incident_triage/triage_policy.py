from __future__ import annotations

from typing import Any

HUMAN_REVIEW_LABEL = "needs_human_review"


def validate_confidence_threshold(confidence_threshold: float) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")


def decide_triage(
    scores: dict[str, float],
    confidence_threshold: float = 0.0,
    review_queue: str = "manual_triage",
) -> dict[str, Any]:
    if not scores:
        raise ValueError("scores must not be empty")
    validate_confidence_threshold(confidence_threshold)

    predicted_label = max(scores, key=scores.get)
    confidence = float(scores[predicted_label])
    needs_human_review = confidence < confidence_threshold
    final_label = HUMAN_REVIEW_LABEL if needs_human_review else predicted_label
    recommended_queue = review_queue if needs_human_review else predicted_label

    return {
        "predicted_label": predicted_label,
        "final_label": final_label,
        "confidence": confidence,
        "confidence_threshold": float(confidence_threshold),
        "needs_human_review": needs_human_review,
        "recommended_queue": recommended_queue,
    }
