from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from devops_incident_triage.labels import INCIDENT_LABELS, id_to_label_map, label_to_id_map


def load_local_dataset(data_dir: Path) -> DatasetDict:
    data_files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
        "test": str(data_dir / "test.jsonl"),
    }
    for split, path in data_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{split} split not found: {path}")
    return load_dataset("json", data_files=data_files)


def build_compute_metrics() -> Any:
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}

    return compute_metrics


def maybe_apply_peft(
    model: Any,
    use_peft: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    if not use_peft:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "PEFT is not installed. Run: uv sync --extra peft"
        ) from exc

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DevOps incident triage classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/devops-incident-triage"))
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_local_dataset(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    if "label_id" not in tokenized["train"].column_names:
        raise ValueError("Expected 'label_id' in processed dataset. Run data prep first.")

    tokenized = tokenized.rename_column("label_id", "labels")
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove_columns = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_columns)

    label2id = label_to_id_map(INCIDENT_LABELS)
    id2label = id_to_label_map(INCIDENT_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(INCIDENT_LABELS),
        label2id=label2id,
        id2label=id2label,
    )
    model = maybe_apply_peft(
        model,
        use_peft=args.use_peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    train_result = trainer.train()
    validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        "train": train_result.metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "model_name": args.model_name,
        "use_peft": args.use_peft,
        "labels": INCIDENT_LABELS,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
