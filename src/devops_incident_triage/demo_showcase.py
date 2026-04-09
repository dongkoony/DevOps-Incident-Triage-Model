from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_showcase_rows(input_file: Path) -> list[dict[str, str]]:
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found")

    rows: list[dict[str, str]] = []
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            raw = json.loads(line)
            row = {
                "id": str(raw.get("id", "")).strip(),
                "title": str(raw.get("title", "")).strip(),
                "text": str(raw.get("text", "")).strip(),
                "expected_label": str(raw.get("expected_label", "")).strip(),
                "note": str(raw.get("note", "")).strip(),
            }
            if not row["title"] or not row["text"]:
                raise ValueError(
                    f"Showcase row {line_number} missing required fields: title/text"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid showcase rows found in {input_file}")
    return rows


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, int]:
    matched = sum(
        1 for row in predictions if row["expected_label"] == row["predicted_label"]
    )
    review_count = sum(1 for row in predictions if row["needs_human_review"])
    total = len(predictions)
    return {
        "total_examples": total,
        "matched_expected_labels": matched,
        "mismatched_labels": total - matched,
        "human_review_count": review_count,
    }


def build_report_payload(
    predictions: list[dict[str, Any]],
    model_path: str,
    confidence_threshold: float,
    review_queue: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": generated_at,
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "review_queue": review_queue,
        },
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
    }


def build_markdown_report(
    predictions: list[dict[str, Any]],
    summary: dict[str, int],
    generated_at: str,
    model_path: str,
    confidence_threshold: float,
) -> str:
    lines = [
        "# Demo Showcase Report",
        "",
        "Curated prediction examples for portfolio demos and repository sharing.",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Model path: `{model_path}`",
        f"- Confidence threshold: `{confidence_threshold}`",
        "",
        "## Summary",
        "",
        f"- Total examples: `{summary['total_examples']}`",
        f"- Expected label matches: `{summary['matched_expected_labels']}`",
        f"- Mismatches: `{summary['mismatched_labels']}`",
        f"- Human review count: `{summary['human_review_count']}`",
        "",
        "## Predictions",
        "",
        "| Title | Expected | Predicted | Confidence | Review Required | Queue |",
        "|---|---|---|---:|---|---|",
    ]
    for row in predictions:
        review_required = "yes" if row["needs_human_review"] else "no"
        lines.append(
            f"| {row['title']} | {row['expected_label']} | {row['predicted_label']} | "
            f"{row['confidence']:.4f} | {review_required} | {row['recommended_queue']} |"
        )

    review_rows = [row for row in predictions if row["needs_human_review"]]
    if review_rows:
        lines.extend(["", "## Review-Required Examples", ""])
        for row in review_rows:
            note = row.get("note", "").strip() or "No note provided."
            lines.append(f"- **{row['title']}**: {note}")
    return "\n".join(lines) + "\n"


def build_terminal_summary(
    predictions: list[dict[str, Any]],
    summary: dict[str, int],
    model_path: str,
) -> str:
    lines = [
        f"Demo showcase for {model_path}",
        f"Examples: {summary['total_examples']}",
        f"Matches: {summary['matched_expected_labels']}/{summary['total_examples']}",
        f"Human review: {summary['human_review_count']}",
        "",
    ]
    for row in predictions:
        route = "review" if row["needs_human_review"] else "auto"
        lines.append(
            f"- {row['title']} -> {row['predicted_label']} "
            f"({row['confidence']:.4f}, {route}, queue={row['recommended_queue']})"
        )
    return "\n".join(lines)


def merge_rows_with_predictions(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        merged.append({**row, **prediction})
    return merged


def write_json(payload: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(markdown: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")
