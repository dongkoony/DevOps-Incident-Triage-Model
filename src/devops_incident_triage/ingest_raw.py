from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from devops_incident_triage.labels import INCIDENT_LABELS

REQUIRED_COLUMNS: list[str] = [
    "incident_id",
    "occurred_at",
    "source",
    "summary",
    "label",
]

OPTIONAL_COLUMNS: list[str] = [
    "details",
    "service",
    "environment",
    "severity",
    "region",
]

CANONICAL_COLUMNS: list[str] = [
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
    "text",
    "data_origin",
]

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    ("aws_account_id", re.compile(r"\b\d{12}\b"), "[REDACTED_AWS_ACCOUNT]"),
]


def normalize_value(value: str) -> str:
    return " ".join(value.strip().split())


def mask_sensitive_text(text: str) -> tuple[str, dict[str, int]]:
    masked = text
    hit_counter: dict[str, int] = {name: 0 for name, _, _ in SENSITIVE_PATTERNS}
    for name, pattern, replacement in SENSITIVE_PATTERNS:
        matches = pattern.findall(masked)
        if matches:
            hit_counter[name] += len(matches)
            masked = pattern.sub(replacement, masked)
    return masked, hit_counter


def build_incident_text(row: dict[str, str], include_metadata: bool = True) -> str:
    segments: list[str] = []
    if include_metadata:
        meta_items = []
        for field in ["environment", "service", "severity", "region", "source"]:
            value = normalize_value(row.get(field, ""))
            if value:
                meta_items.append(f"{field}={value}")
        if meta_items:
            segments.append("[" + " | ".join(meta_items) + "]")

    summary = normalize_value(row.get("summary", ""))
    details = normalize_value(row.get("details", ""))
    if summary:
        segments.append(summary)
    if details:
        segments.append(f"Details: {details}")
    return " ".join(segments).strip()


def read_raw_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns in {input_path}: {joined}")

        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {key: normalize_value((raw or {}).get(key, "")) for key in header}
            rows.append(row)
    return rows


def deduplicate_rows(
    rows: list[dict[str, str]],
    deduplicate_by: str,
) -> tuple[list[dict[str, str]], int]:
    if deduplicate_by == "none":
        return rows, 0
    if deduplicate_by not in {"incident_id", "text"}:
        raise ValueError("deduplicate_by must be one of: incident_id, text, none")

    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    dropped = 0
    for row in rows:
        key = row.get(deduplicate_by, "")
        if not key:
            unique_rows.append(row)
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows, dropped


def write_csv(rows: list[dict[str, str]], output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def ingest_raw_dataset(
    input_path: Path,
    output_canonical_path: Path,
    output_training_path: Path,
    report_path: Path,
    strict_labels: bool = True,
    mask_sensitive: bool = True,
    include_metadata: bool = True,
    deduplicate_by: str = "incident_id",
) -> dict[str, Any]:
    raw_rows = read_raw_rows(input_path)

    missing_required_rows = 0
    unknown_label_rows = 0
    empty_text_rows = 0
    pii_hits_total = Counter()
    processed_rows: list[dict[str, str]] = []

    for row in raw_rows:
        if any(not row.get(column, "") for column in REQUIRED_COLUMNS):
            missing_required_rows += 1
            continue

        label = row["label"]
        if strict_labels and label not in INCIDENT_LABELS:
            unknown_label_rows += 1
            continue

        text = build_incident_text(row, include_metadata=include_metadata)
        pii_hits = {name: 0 for name, _, _ in SENSITIVE_PATTERNS}
        if mask_sensitive:
            text, pii_hits = mask_sensitive_text(text)
        pii_hits_total.update(pii_hits)

        if not text:
            empty_text_rows += 1
            continue

        canonical_row = {
            "incident_id": row.get("incident_id", ""),
            "occurred_at": row.get("occurred_at", ""),
            "source": row.get("source", ""),
            "summary": row.get("summary", ""),
            "details": row.get("details", ""),
            "service": row.get("service", ""),
            "environment": row.get("environment", ""),
            "severity": row.get("severity", ""),
            "region": row.get("region", ""),
            "label": label,
            "text": text,
            "data_origin": "raw_ingestion",
        }
        processed_rows.append(canonical_row)

    deduplicated_rows, deduplicated_dropped = deduplicate_rows(
        processed_rows,
        deduplicate_by=deduplicate_by,
    )

    training_rows = [
        {"text": row["text"], "label": row["label"], "source": row.get("source", "unknown")}
        for row in deduplicated_rows
    ]
    write_csv(deduplicated_rows, output_canonical_path, CANONICAL_COLUMNS)
    write_csv(training_rows, output_training_path, ["text", "label", "source"])

    label_distribution = dict(
        sorted(Counter(row["label"] for row in deduplicated_rows).items(), key=lambda x: x[0])
    )
    unknown_labels = sorted(
        set(
            row.get("label", "")
            for row in raw_rows
            if row.get("label", "") and row.get("label", "") not in INCIDENT_LABELS
        )
    )

    report = {
        "input_path": str(input_path),
        "output_canonical_path": str(output_canonical_path),
        "output_training_path": str(output_training_path),
        "strict_labels": strict_labels,
        "mask_sensitive": mask_sensitive,
        "include_metadata": include_metadata,
        "deduplicate_by": deduplicate_by,
        "summary": {
            "total_input_rows": len(raw_rows),
            "missing_required_rows": missing_required_rows,
            "unknown_label_rows": unknown_label_rows,
            "empty_text_rows": empty_text_rows,
            "deduplicated_dropped_rows": deduplicated_dropped,
            "rows_written_canonical": len(deduplicated_rows),
            "rows_written_training": len(training_rows),
        },
        "label_distribution": label_distribution,
        "pii_hits": dict(pii_hits_total),
        "unknown_labels_detected": unknown_labels,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest raw incident CSV into canonical and training-ready dataset files."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/incidents_template.csv"),
        help="Raw incident CSV path.",
    )
    parser.add_argument(
        "--output-canonical-path",
        type=Path,
        default=Path("data/raw/incidents_canonical.csv"),
        help="Normalized canonical CSV output path.",
    )
    parser.add_argument(
        "--output-training-path",
        type=Path,
        default=Path("data/raw/incidents_training_ready.csv"),
        help="Training-ready CSV output path (text,label,source).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/raw_ingestion_report.json"),
        help="Quality report JSON output path.",
    )
    parser.add_argument(
        "--allow-unknown-labels",
        action="store_true",
        help="When enabled, rows with unknown labels are kept.",
    )
    parser.add_argument(
        "--disable-sensitive-masking",
        action="store_true",
        help="Disable masking for email/IP/AWS account IDs.",
    )
    parser.add_argument(
        "--disable-metadata-prefix",
        action="store_true",
        help="Disable metadata prefix in training text.",
    )
    parser.add_argument(
        "--deduplicate-by",
        type=str,
        default="incident_id",
        choices=["incident_id", "text", "none"],
        help="Deduplication key strategy.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"{args.input_path} not found.")

    report = ingest_raw_dataset(
        input_path=args.input_path,
        output_canonical_path=args.output_canonical_path,
        output_training_path=args.output_training_path,
        report_path=args.report_path,
        strict_labels=not args.allow_unknown_labels,
        mask_sensitive=not args.disable_sensitive_masking,
        include_metadata=not args.disable_metadata_prefix,
        deduplicate_by=args.deduplicate_by,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
