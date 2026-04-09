import json
from pathlib import Path

import pytest

from devops_incident_triage.demo_showcase import (
    load_showcase_rows,
    summarize_predictions,
)


def test_load_showcase_rows_reads_curated_examples(tmp_path: Path) -> None:
    input_file = tmp_path / "incidents_showcase.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "demo-001",
                        "title": "EKS nodes not ready",
                        "text": "EKS worker nodes became NotReady after a CNI upgrade.",
                        "expected_label": "k8s_cluster",
                        "note": "Clear Kubernetes case.",
                    }
                ),
                json.dumps(
                    {
                        "id": "demo-002",
                        "title": "IAM role denied during deploy",
                        "text": "GitHub Actions deploy failed because sts:AssumeRole was denied.",
                        "expected_label": "aws_iam_network",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_showcase_rows(input_file)

    assert [row["id"] for row in rows] == ["demo-001", "demo-002"]
    assert rows[0]["note"] == "Clear Kubernetes case."
    assert rows[1]["note"] == ""


def test_load_showcase_rows_requires_title_and_text(tmp_path: Path) -> None:
    input_file = tmp_path / "invalid_showcase.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "demo-003",
                "title": "",
                "text": "Deployment failed with a timeout.",
                "expected_label": "deployment_release",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_showcase_rows(input_file)


def test_summarize_predictions_counts_matches_and_reviews() -> None:
    predictions = [
        {
            "title": "Kubernetes outage",
            "expected_label": "k8s_cluster",
            "predicted_label": "k8s_cluster",
            "needs_human_review": False,
        },
        {
            "title": "Ambiguous release issue",
            "expected_label": "deployment_release",
            "predicted_label": "aws_iam_network",
            "needs_human_review": True,
        },
    ]

    summary = summarize_predictions(predictions)

    assert summary == {
        "total_examples": 2,
        "matched_expected_labels": 1,
        "mismatched_labels": 1,
        "human_review_count": 1,
    }
