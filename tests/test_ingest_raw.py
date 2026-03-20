import csv
from pathlib import Path

from devops_incident_triage.ingest_raw import ingest_raw_dataset, mask_sensitive_text


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "incident_id",
                "occurred_at",
                "source",
                "summary",
                "details",
                "service",
                "environment",
                "severity",
                "region",
                "label",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_mask_sensitive_text() -> None:
    text = "Contact kim.devops@example.com from 10.20.30.40 in account 123456789012."
    masked, hits = mask_sensitive_text(text)
    assert "[REDACTED_EMAIL]" in masked
    assert "[REDACTED_IP]" in masked
    assert "[REDACTED_AWS_ACCOUNT]" in masked
    assert hits["email"] == 1
    assert hits["ipv4"] == 1
    assert hits["aws_account_id"] == 1


def test_ingest_raw_dataset_strict_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    canonical = tmp_path / "canonical.csv"
    training = tmp_path / "training.csv"
    report_path = tmp_path / "report.json"

    rows = [
        {
            "incident_id": "INC-1",
            "occurred_at": "2026-03-20T00:00:00Z",
            "source": "pagerduty",
            "summary": "Node NotReady after CNI change",
            "details": "owner: kim.devops@example.com, node 10.20.30.40",
            "service": "platform",
            "environment": "prod",
            "severity": "P1",
            "region": "ap-northeast-2",
            "label": "k8s_cluster",
        },
        {
            "incident_id": "INC-2",
            "occurred_at": "2026-03-20T00:01:00Z",
            "source": "pagerduty",
            "summary": "Unknown bucket issue",
            "details": "",
            "service": "storage",
            "environment": "prod",
            "severity": "P2",
            "region": "ap-northeast-2",
            "label": "unknown_label",
        },
        {
            "incident_id": "INC-3",
            "occurred_at": "2026-03-20T00:02:00Z",
            "source": "pagerduty",
            "summary": "",
            "details": "summary is missing",
            "service": "api",
            "environment": "prod",
            "severity": "P2",
            "region": "ap-northeast-2",
            "label": "deployment_release",
        },
        {
            "incident_id": "INC-1",
            "occurred_at": "2026-03-20T00:03:00Z",
            "source": "pagerduty",
            "summary": "Duplicate same incident id",
            "details": "",
            "service": "platform",
            "environment": "prod",
            "severity": "P1",
            "region": "ap-northeast-2",
            "label": "k8s_cluster",
        },
        {
            "incident_id": "INC-4",
            "occurred_at": "2026-03-20T00:04:00Z",
            "source": "github-actions",
            "summary": "AssumeRole denied in release workflow",
            "details": "account 123456789012",
            "service": "release",
            "environment": "prod",
            "severity": "P1",
            "region": "ap-northeast-2",
            "label": "aws_iam_network",
        },
    ]
    _write_rows(input_path, rows)

    report = ingest_raw_dataset(
        input_path=input_path,
        output_canonical_path=canonical,
        output_training_path=training,
        report_path=report_path,
        strict_labels=True,
        mask_sensitive=True,
        include_metadata=True,
        deduplicate_by="incident_id",
    )

    assert canonical.exists()
    assert training.exists()
    assert report_path.exists()
    assert report["summary"]["total_input_rows"] == 5
    assert report["summary"]["unknown_label_rows"] == 1
    assert report["summary"]["missing_required_rows"] == 1
    assert report["summary"]["deduplicated_dropped_rows"] == 1
    assert report["summary"]["rows_written_training"] == 2
    assert report["pii_hits"]["email"] >= 1
    assert report["pii_hits"]["ipv4"] >= 1
    assert report["pii_hits"]["aws_account_id"] >= 1


def test_ingest_raw_dataset_allow_unknown_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    canonical = tmp_path / "canonical.csv"
    training = tmp_path / "training.csv"
    report_path = tmp_path / "report.json"

    rows = [
        {
            "incident_id": "INC-1",
            "occurred_at": "2026-03-20T00:00:00Z",
            "source": "pagerduty",
            "summary": "Some unknown ticket",
            "details": "",
            "service": "platform",
            "environment": "prod",
            "severity": "P2",
            "region": "ap-northeast-2",
            "label": "legacy_unknown",
        }
    ]
    _write_rows(input_path, rows)

    report = ingest_raw_dataset(
        input_path=input_path,
        output_canonical_path=canonical,
        output_training_path=training,
        report_path=report_path,
        strict_labels=False,
        mask_sensitive=True,
        include_metadata=False,
        deduplicate_by="none",
    )
    assert report["summary"]["unknown_label_rows"] == 0
    assert report["summary"]["rows_written_training"] == 1
