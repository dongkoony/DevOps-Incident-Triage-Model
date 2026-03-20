from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from devops_incident_triage.labels import INCIDENT_LABELS, label_to_id_map, validate_labels


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def read_examples_from_csv(
    input_path: Path,
    text_column: str = "text",
    label_column: str = "label",
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _normalize_text(row.get(text_column, ""))
            label = row.get(label_column, "").strip()
            if not text or not label:
                continue
            examples.append(
                {
                    "text": text,
                    "label": label,
                    "source": row.get("source", "unknown").strip(),
                }
            )
    if not examples:
        raise ValueError(f"No valid rows found in {input_path}")
    validate_labels([item["label"] for item in examples])
    return examples


def write_examples_to_csv(examples: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        for row in examples:
            writer.writerow(row)


def _build_synthetic_text(label: str, rng: random.Random) -> str:
    env = rng.choice(["prod", "staging", "dev"])
    region = rng.choice(["ap-northeast-2", "us-east-1", "eu-west-1"])
    service = rng.choice(["checkout", "billing", "api-gateway", "worker", "metrics"])
    cluster = rng.choice(["eks-main", "gke-ops", "onprem-k8s"])
    pipeline = rng.choice(["github-actions", "jenkins", "gitlab-ci", "argo"])
    db = rng.choice(["postgres", "mysql", "redis", "opensearch"])

    templates: dict[str, list[str]] = {
        "k8s_cluster": [
            f"[{env}] Pod pending due to node affinity mismatch in {cluster}.",
            f"Kubernetes scheduler failed: 0/8 nodes available, insufficient memory in {service}.",
            f"Kubelet on node ip-10-0-8-14 not ready after CNI restart in {region}.",
            f"CrashLoopBackOff spikes after control-plane certificate rotation on {cluster}.",
        ],
        "cicd_pipeline": [
            f"{pipeline} build failed: unit test stage timed out for {service}.",
            f"Artifact upload step returned HTTP 403 in {pipeline} release workflow.",
            "Terraform plan job aborted because backend lock could not be acquired.",
            "Container scan gate failed due to critical CVE in base image.",
        ],
        "aws_iam_network": [
            f"IAM role missing sts:AssumeRole permission for {service} deploy bot.",
            f"ALB target group health checks failing after security group update in {region}.",
            f"Route table misconfiguration blocked private subnet egress in {env}.",
            "EKS worker nodes cannot pull secrets from SSM due to KMS deny.",
        ],
        "deployment_release": [
            f"Helm upgrade failed with immutable field error in {service} deployment.",
            "Blue/green switch caused 502 errors due to stale config map mount.",
            "Canary rollout paused after error rate exceeded SLO threshold.",
            "ArgoCD sync drift detected between git and live cluster resources.",
        ],
        "container_runtime": [
            "Docker daemon reports overlay2 filesystem corruption on worker node.",
            f"Container OOMKilled repeatedly despite memory limit increase in {service}.",
            f"ImagePullBackOff: registry auth token expired in {pipeline} runtime.",
            "containerd failed to create task: shim disconnected under high load.",
        ],
        "observability_alerting": [
            f"Prometheus alert storm triggered by duplicate scrape targets in {env}.",
            "Loki ingest lag exceeded 10m causing delayed incident visibility.",
            "PagerDuty received false positives due to noisy CPU anomaly rule.",
            "Tracing data missing after OpenTelemetry collector config rollback.",
        ],
        "database_state": [
            f"{db} replication lag above 30s after storage burst credit depletion.",
            f"Connection pool exhausted; app cannot reach {db} primary endpoint.",
            "Migration job stuck holding metadata lock, causing write latency spike.",
            f"Read replica in {region} went read-only during failover rehearsal.",
        ],
    }
    return rng.choice(templates[label])


def generate_synthetic_examples(samples_per_label: int, seed: int = 42) -> list[dict[str, str]]:
    if samples_per_label < 1:
        raise ValueError("samples_per_label must be >= 1")
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    for label in INCIDENT_LABELS:
        for _ in range(samples_per_label):
            examples.append(
                {
                    "text": _build_synthetic_text(label, rng),
                    "label": label,
                    "source": "synthetic",
                }
            )
    rng.shuffle(examples)
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    labels = [item["label"] for item in examples]
    try:
        train_val, test = train_test_split(
            examples,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        )
        train_val_labels = [item["label"] for item in train_val]
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=seed,
            stratify=train_val_labels,
        )
    except ValueError:
        train_val, test = train_test_split(
            examples,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return train, val, test


def attach_label_ids(
    examples: list[dict[str, str]], label2id: dict[str, int]
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in examples:
        converted.append(
            {
                "text": item["text"],
                "label": item["label"],
                "label_id": label2id[item["label"]],
                "source": item.get("source", "unknown"),
            }
        )
    return converted


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["label"] for item in rows).items(), key=lambda x: x[0]))


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    text_column: str,
    label_column: str,
) -> dict[str, Any]:
    examples = read_examples_from_csv(
        input_path,
        text_column=text_column,
        label_column=label_column,
    )
    label2id = label_to_id_map(INCIDENT_LABELS)
    train, val, test = split_examples(
        examples=examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_rows = attach_label_ids(train, label2id)
    val_rows = attach_label_ids(val, label2id)
    test_rows = attach_label_ids(test, label2id)

    write_jsonl(train_rows, output_dir / "train.jsonl")
    write_jsonl(val_rows, output_dir / "validation.jsonl")
    write_jsonl(test_rows, output_dir / "test.jsonl")

    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "splits": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "total": len(train_rows) + len(val_rows) + len(test_rows),
        },
        "label_distribution": {
            "train": _distribution(train_rows),
            "validation": _distribution(val_rows),
            "test": _distribution(test_rows),
        },
        "labels": INCIDENT_LABELS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label2id, indent=2),
        encoding="utf-8",
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DevOps incident dataset for training.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/sample/incidents_synthetic.csv"),
        help="CSV file containing text and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write train/validation/test JSONL files.",
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic CSV at --input-path before preparing splits.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=50,
        help="Used when --generate-synthetic is enabled.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.generate_synthetic:
        synthetic_examples = generate_synthetic_examples(
            samples_per_label=args.samples_per_label,
            seed=args.seed,
        )
        write_examples_to_csv(synthetic_examples, args.input_path)

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"{args.input_path} not found. Provide --input-path or use --generate-synthetic."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
