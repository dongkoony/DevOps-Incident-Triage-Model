from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.predict import predict_batch

DEFAULT_MODEL_PATH = Path("models/devops-incident-triage")
DEFAULT_INPUT_FILE = Path("data/demo/incidents_showcase.jsonl")
DEFAULT_OUTPUT_JSON = Path("reports/demo_showcase.json")
DEFAULT_OUTPUT_MARKDOWN = Path("reports/demo_showcase.md")


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


def load_model_bundle(
    model_path: Path,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


def run_showcase(
    model_path: Path,
    input_file: Path,
    confidence_threshold: float,
    review_queue: str,
    output_json: Path,
    output_markdown: Path,
    max_length: int,
) -> str:
    rows = load_showcase_rows(input_file)
    model, tokenizer = load_model_bundle(model_path)
    predictions = predict_batch(
        [row["text"] for row in rows],
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        confidence_threshold=confidence_threshold,
        review_queue=review_queue,
    )
    merged = merge_rows_with_predictions(rows, predictions)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = build_report_payload(
        predictions=merged,
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
        review_queue=review_queue,
        generated_at=generated_at,
    )
    markdown = build_markdown_report(
        predictions=merged,
        summary=payload["summary"],
        generated_at=generated_at,
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
    )
    terminal_output = build_terminal_summary(
        predictions=merged,
        summary=payload["summary"],
        model_path=str(model_path),
    )
    write_json(payload, output_json)
    write_markdown(markdown, output_markdown)
    return terminal_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a curated demo showcase for DevOps incident triage."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--review-queue", type=str, default="sre_manual_triage")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    terminal_output = run_showcase(
        model_path=args.model_path,
        input_file=args.input_file,
        confidence_threshold=args.confidence_threshold,
        review_queue=args.review_queue,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        max_length=args.max_length,
    )
    print(terminal_output)
    print(f"JSON report saved to {args.output_json}")
    print(f"Markdown report saved to {args.output_markdown}")


if __name__ == "__main__":
    main()
