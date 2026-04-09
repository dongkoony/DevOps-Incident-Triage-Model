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
