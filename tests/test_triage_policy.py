import pytest

from devops_incident_triage.triage_policy import (
    HUMAN_REVIEW_LABEL,
    decide_triage,
    validate_confidence_threshold,
)


def test_validate_confidence_threshold_range() -> None:
    validate_confidence_threshold(0.0)
    validate_confidence_threshold(1.0)
    with pytest.raises(ValueError):
        validate_confidence_threshold(-0.01)
    with pytest.raises(ValueError):
        validate_confidence_threshold(1.01)


def test_decide_triage_auto_route() -> None:
    scores = {
        "k8s_cluster": 0.81,
        "deployment_release": 0.12,
        "aws_iam_network": 0.07,
    }
    decision = decide_triage(scores, confidence_threshold=0.6, review_queue="sre_manual")
    assert decision["predicted_label"] == "k8s_cluster"
    assert decision["final_label"] == "k8s_cluster"
    assert decision["needs_human_review"] is False
    assert decision["recommended_queue"] == "k8s_cluster"


def test_decide_triage_human_review_route() -> None:
    scores = {
        "k8s_cluster": 0.44,
        "deployment_release": 0.33,
        "aws_iam_network": 0.23,
    }
    decision = decide_triage(scores, confidence_threshold=0.6, review_queue="sre_manual")
    assert decision["predicted_label"] == "k8s_cluster"
    assert decision["final_label"] == HUMAN_REVIEW_LABEL
    assert decision["needs_human_review"] is True
    assert decision["recommended_queue"] == "sre_manual"
