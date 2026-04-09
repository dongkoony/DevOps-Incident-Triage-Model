import json
from pathlib import Path

import pytest

from devops_incident_triage import demo_showcase
from devops_incident_triage.demo_showcase import (
    build_markdown_report,
    build_report_payload,
    build_terminal_summary,
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


def test_build_report_payload_includes_metadata_and_summary() -> None:
    rows = [
        {
            "id": "demo-001",
            "title": "EKS nodes not ready",
            "text": "EKS worker nodes became NotReady after a CNI upgrade.",
            "expected_label": "k8s_cluster",
            "note": "Clear Kubernetes case.",
            "predicted_label": "k8s_cluster",
            "final_label": "k8s_cluster",
            "confidence": 0.91,
            "confidence_threshold": 0.6,
            "needs_human_review": False,
            "recommended_queue": "k8s_cluster",
            "scores": {"k8s_cluster": 0.91},
        }
    ]

    payload = build_report_payload(
        predictions=rows,
        model_path="models/devops-incident-triage",
        confidence_threshold=0.6,
        review_queue="sre_manual_triage",
        generated_at="2026-04-09T00:00:00+00:00",
    )

    assert payload["metadata"]["model_path"] == "models/devops-incident-triage"
    assert payload["summary"]["matched_expected_labels"] == 1
    assert payload["predictions"][0]["title"] == "EKS nodes not ready"


def test_build_markdown_report_lists_review_examples() -> None:
    markdown = build_markdown_report(
        predictions=[
            {
                "title": "Ambiguous deploy failure",
                "expected_label": "deployment_release",
                "predicted_label": "aws_iam_network",
                "confidence": 0.44,
                "needs_human_review": True,
                "recommended_queue": "sre_manual_triage",
                "note": "Needs manual triage.",
            }
        ],
        summary={
            "total_examples": 1,
            "matched_expected_labels": 0,
            "mismatched_labels": 1,
            "human_review_count": 1,
        },
        generated_at="2026-04-09T00:00:00+00:00",
        model_path="models/devops-incident-triage",
        confidence_threshold=0.6,
    )

    assert "# Demo Showcase Report" in markdown
    assert (
        "| Ambiguous deploy failure | deployment_release | aws_iam_network | 0.4400 | "
        "yes | sre_manual_triage |" in markdown
    )
    assert "## Review-Required Examples" in markdown
    assert "- **Ambiguous deploy failure**: Needs manual triage." in markdown


def test_build_terminal_summary_is_human_readable() -> None:
    output = build_terminal_summary(
        predictions=[
            {
                "title": "EKS nodes not ready",
                "predicted_label": "k8s_cluster",
                "confidence": 0.9123,
                "needs_human_review": False,
                "recommended_queue": "k8s_cluster",
            }
        ],
        summary={
            "total_examples": 1,
            "matched_expected_labels": 1,
            "mismatched_labels": 0,
            "human_review_count": 0,
        },
        model_path="models/devops-incident-triage",
    )

    assert "Demo showcase for models/devops-incident-triage" in output
    assert "Matches: 1/1" in output
    assert "- EKS nodes not ready -> k8s_cluster (0.9123, auto, queue=k8s_cluster)" in output


def test_run_showcase_generates_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "showcase.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "demo-001",
                "title": "EKS nodes not ready",
                "text": "EKS worker nodes became NotReady after a CNI upgrade.",
                "expected_label": "k8s_cluster",
                "note": "Clear Kubernetes case.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyModel:
        config = type("Config", (), {"id2label": {0: "k8s_cluster"}})()

    class DummyTokenizer:
        pass

    monkeypatch.setattr(
        demo_showcase,
        "load_model_bundle",
        lambda model_path: (DummyModel(), DummyTokenizer()),
    )
    monkeypatch.setattr(
        demo_showcase,
        "predict_batch",
        lambda texts, model, tokenizer, max_length, confidence_threshold, review_queue: [
            {
                "text": texts[0],
                "predicted_label": "k8s_cluster",
                "final_label": "k8s_cluster",
                "confidence": 0.91,
                "confidence_threshold": confidence_threshold,
                "needs_human_review": False,
                "recommended_queue": "k8s_cluster",
                "scores": {"k8s_cluster": 0.91},
            }
        ],
    )

    json_path = tmp_path / "demo_showcase.json"
    markdown_path = tmp_path / "demo_showcase.md"

    terminal_output = demo_showcase.run_showcase(
        model_path=Path("models/devops-incident-triage"),
        input_file=input_file,
        confidence_threshold=0.6,
        review_queue="sre_manual_triage",
        output_json=json_path,
        output_markdown=markdown_path,
        max_length=256,
    )

    assert "Matches: 1/1" in terminal_output
    assert json_path.exists()
    assert markdown_path.exists()
    assert '"matched_expected_labels": 1' in json_path.read_text(encoding="utf-8")
    assert "# Demo Showcase Report" in markdown_path.read_text(encoding="utf-8")
