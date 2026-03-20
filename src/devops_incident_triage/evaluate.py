from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DevOps incident classifier.")
    parser.add_argument("--model-path", type=Path, default=Path("models/devops-incident-triage"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=20)
    return parser


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def maybe_plot_confusion_matrix(cm_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - DevOps Incident Triage")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    test_path = args.data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found. Run data prep first.")

    dataset = load_dataset("json", data_files={"test": str(test_path)})["test"]
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels_order = [id2label[i] for i in sorted(id2label.keys())]

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove_columns = [c for c in tokenized.column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_columns)

    trainer = Trainer(model=model, processing_class=tokenizer)
    predictions = trainer.predict(tokenized)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    overall_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "num_samples": int(len(y_true)),
    }
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels_order))),
        target_names=labels_order,
        output_dict=True,
        zero_division=0,
    )
    per_label_report = {
        label: {
            "precision": float(report_dict[label]["precision"]),
            "recall": float(report_dict[label]["recall"]),
            "f1-score": float(report_dict[label]["f1-score"]),
            "support": int(report_dict[label]["support"]),
        }
        for label in labels_order
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_order))))
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(args.report_dir / "confusion_matrix.csv")
    maybe_plot_confusion_matrix(cm_df, args.report_dir / "figures" / "confusion_matrix.png")

    sample_rows = []
    max_rows = min(args.sample_size, len(dataset))
    for i in range(max_rows):
        logits = predictions.predictions[i]
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.exp(logits).sum()
        label_scores = {labels_order[j]: float(probs[j]) for j in range(len(labels_order))}
        pred_idx = int(y_pred[i])
        true_idx = int(y_true[i])
        sample_rows.append(
            {
                "text": dataset[i]["text"],
                "true_label": id2label[true_idx],
                "predicted_label": id2label[pred_idx],
                "confidence": float(probs[pred_idx]),
                "scores": label_scores,
            }
        )

    with (args.report_dir / "sample_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(overall_metrics, args.report_dir / "evaluation_metrics.json")
    write_json(per_label_report, args.report_dir / "per_label_metrics.json")
    summary = {
        "overall_metrics": overall_metrics,
        "labels": labels_order,
        "artifacts": {
            "per_label": str(args.report_dir / "per_label_metrics.json"),
            "confusion_matrix_csv": str(args.report_dir / "confusion_matrix.csv"),
            "confusion_matrix_png": str(args.report_dir / "figures" / "confusion_matrix.png"),
            "sample_predictions": str(args.report_dir / "sample_predictions.jsonl"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
