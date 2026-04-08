from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    "distilbert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "xlm-roberta-base",
]


def parse_model_names(raw_models: str) -> list[str]:
    unique_models: list[str] = []
    seen: set[str] = set()
    for token in raw_models.split(","):
        model_name = token.strip()
        if not model_name:
            continue
        if model_name in seen:
            continue
        unique_models.append(model_name)
        seen.add(model_name)
    if not unique_models:
        raise ValueError("At least one model must be provided.")
    return unique_models


def make_model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(command: list[str]) -> float:
    started_at = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started_at


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_markdown_report(
    rows: list[dict[str, Any]],
    generated_at: str,
    best_model: str | None,
) -> str:
    lines = [
        "# Model Benchmark Report",
        "",
        f"- Generated at (UTC): {generated_at}",
    ]
    if best_model is not None:
        lines.append(f"- Best model (test macro F1): `{best_model}`")
    lines.extend(
        [
            "",
            "| Model | Status | Test Macro F1 | Test Accuracy | Weighted F1 | Train(s) | Eval(s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_name"],
                    row["status"],
                    _format_float(row.get("test_macro_f1")),
                    _format_float(row.get("test_accuracy")),
                    _format_float(row.get("weighted_f1")),
                    _format_float(row.get("train_duration_seconds"), digits=2),
                    _format_float(row.get("eval_duration_seconds"), digits=2),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple base models with shared training/evaluation settings."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--base-model-dir", type=Path, default=Path("models/benchmarks"))
    parser.add_argument("--base-report-dir", type=Path, default=Path("reports/benchmarks"))
    parser.add_argument(
        "--summary-json-path",
        type=Path,
        default=Path("reports/model_benchmark.json"),
    )
    parser.add_argument(
        "--summary-markdown-path",
        type=Path,
        default=Path("reports/model_benchmark.md"),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--confidence-thresholds",
        type=str,
        default="0.4,0.5,0.6,0.7",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model if training and evaluation artifacts already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one model fails.",
    )
    return parser


def _extract_scores(
    training_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    validation_metrics = training_metrics.get("validation", {})
    validation_macro_f1 = validation_metrics.get("validation_macro_f1")
    test_accuracy = evaluation_metrics.get("accuracy")
    test_macro_f1 = evaluation_metrics.get("macro_f1")
    weighted_f1 = evaluation_metrics.get("weighted_f1")
    return validation_macro_f1, test_accuracy, test_macro_f1, weighted_f1


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    model_names = parse_model_names(args.models)

    rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(model_names, start=1):
        slug = make_model_slug(model_name)
        model_dir = args.base_model_dir / slug
        report_dir = args.base_report_dir / slug
        training_metrics_path = model_dir / "training_metrics.json"
        evaluation_metrics_path = report_dir / "evaluation_metrics.json"
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_slug": slug,
            "model_dir": str(model_dir),
            "report_dir": str(report_dir),
            "status": "pending",
            "train_duration_seconds": None,
            "eval_duration_seconds": None,
            "validation_macro_f1": None,
            "test_accuracy": None,
            "test_macro_f1": None,
            "weighted_f1": None,
            "error": None,
        }
        rows.append(row)
        print(f"[{idx}/{len(model_names)}] Benchmarking: {model_name}")

        if (
            args.skip_existing
            and training_metrics_path.exists()
            and evaluation_metrics_path.exists()
        ):
            row["status"] = "skipped"
        else:
            train_command = [
                sys.executable,
                "-m",
                "devops_incident_triage.train",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(model_dir),
                "--model-name",
                model_name,
                "--max-length",
                str(args.max_length),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--seed",
                str(args.seed),
            ]
            eval_command = [
                sys.executable,
                "-m",
                "devops_incident_triage.evaluate",
                "--model-path",
                str(model_dir),
                "--data-dir",
                str(args.data_dir),
                "--report-dir",
                str(report_dir),
                "--max-length",
                str(args.max_length),
                "--sample-size",
                str(args.sample_size),
                "--confidence-thresholds",
                args.confidence_thresholds,
            ]
            try:
                row["train_duration_seconds"] = run_command(train_command)
                row["eval_duration_seconds"] = run_command(eval_command)
                row["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                if args.fail_fast:
                    raise

        if training_metrics_path.exists() and evaluation_metrics_path.exists():
            training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
            evaluation_metrics = json.loads(evaluation_metrics_path.read_text(encoding="utf-8"))
            validation_macro_f1, test_accuracy, test_macro_f1, weighted_f1 = _extract_scores(
                training_metrics=training_metrics,
                evaluation_metrics=evaluation_metrics,
            )
            row["validation_macro_f1"] = validation_macro_f1
            row["test_accuracy"] = test_accuracy
            row["test_macro_f1"] = test_macro_f1
            row["weighted_f1"] = weighted_f1

    completed_rows = [row for row in rows if row["status"] in {"completed", "skipped"}]
    scored_rows = [row for row in completed_rows if row["test_macro_f1"] is not None]
    scored_rows = sorted(
        scored_rows,
        key=lambda row: (row["test_macro_f1"], row["test_accuracy"] or 0.0),
        reverse=True,
    )
    best_model = scored_rows[0]["model_name"] if scored_rows else None
    generated_at = datetime.now(UTC).isoformat()

    summary_payload = {
        "generated_at_utc": generated_at,
        "data_dir": str(args.data_dir),
        "models": rows,
        "best_model_by_test_macro_f1": best_model,
    }
    write_json(summary_payload, args.summary_json_path)

    markdown = build_markdown_report(rows=rows, generated_at=generated_at, best_model=best_model)
    args.summary_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
