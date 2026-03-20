from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.triage_policy import decide_triage, validate_confidence_threshold


def load_texts_from_file(input_file: Path, text_column: str = "text") -> list[str]:
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found")
    suffix = input_file.suffix.lower()
    texts: list[str] = []

    if suffix == ".csv":
        with input_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, "").strip()
                if text:
                    texts.append(text)
    elif suffix == ".jsonl":
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get(text_column, "")).strip()
                if text:
                    texts.append(text)
    elif suffix == ".txt":
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)
    else:
        raise ValueError("Unsupported input file format. Use .csv, .jsonl, or .txt")

    if not texts:
        raise ValueError(f"No valid text rows found in {input_file}")
    return texts


def predict_batch(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    confidence_threshold: float = 0.0,
    review_queue: str = "manual_triage",
) -> list[dict[str, Any]]:
    validate_confidence_threshold(confidence_threshold)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    results: list[dict[str, Any]] = []
    for text, probs in zip(texts, probabilities, strict=True):
        score_map = {id2label[i]: float(probs[i]) for i in range(len(id2label))}
        triage = decide_triage(
            score_map,
            confidence_threshold=confidence_threshold,
            review_queue=review_queue,
        )
        results.append(
            {
                "text": text,
                "predicted_label": triage["predicted_label"],
                "final_label": triage["final_label"],
                "confidence": triage["confidence"],
                "confidence_threshold": triage["confidence_threshold"],
                "needs_human_review": triage["needs_human_review"],
                "recommended_queue": triage["recommended_queue"],
                "scores": score_map,
            }
        )
    return results


def write_jsonl(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local inference for DevOps incident triage model."
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/devops-incident-triage"))
    parser.add_argument("--text", action="append", help="Single text input. Can be repeated.")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional batch input file (.csv/.jsonl/.txt).",
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional output file for batch predictions (.jsonl).",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="If confidence is below this threshold, mark as needs_human_review.",
    )
    parser.add_argument(
        "--review-queue",
        type=str,
        default="manual_triage",
        help="Queue name used when prediction falls below confidence threshold.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("Provide --text or --input-file")

    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    texts = list(args.text or [])
    if args.input_file:
        texts.extend(load_texts_from_file(args.input_file, text_column=args.text_column))
    results = predict_batch(
        texts,
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        confidence_threshold=args.confidence_threshold,
        review_queue=args.review_queue,
    )

    if args.output_file:
        write_jsonl(results, args.output_file)
        print(f"Saved {len(results)} predictions to {args.output_file}")
        return
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
