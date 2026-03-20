from __future__ import annotations

from collections.abc import Sequence

INCIDENT_LABELS: list[str] = [
    "k8s_cluster",
    "cicd_pipeline",
    "aws_iam_network",
    "deployment_release",
    "container_runtime",
    "observability_alerting",
    "database_state",
]

LABEL_DESCRIPTIONS: dict[str, str] = {
    "k8s_cluster": "Kubernetes scheduling, node, control-plane, or cluster state issues.",
    "cicd_pipeline": "CI/CD failures across build, test, artifact, and pipeline orchestration.",
    "aws_iam_network": "AWS IAM, VPC, SG, routing, load balancing, and permission problems.",
    "deployment_release": "Rollout, Helm, release strategy, config drift, and deployment issues.",
    "container_runtime": "Docker/container runtime crashes, image pull failures, resource limits.",
    "observability_alerting": "Monitoring, logging, tracing, alert noise, and telemetry gaps.",
    "database_state": "Database connectivity, replication lag, migration lock, storage saturation.",
}


def label_to_id_map(labels: Sequence[str] | None = None) -> dict[str, int]:
    selected = list(labels or INCIDENT_LABELS)
    return {label: idx for idx, label in enumerate(selected)}


def id_to_label_map(labels: Sequence[str] | None = None) -> dict[int, str]:
    selected = list(labels or INCIDENT_LABELS)
    return {idx: label for idx, label in enumerate(selected)}


def validate_labels(labels: Sequence[str]) -> None:
    unknown = sorted(set(labels) - set(INCIDENT_LABELS))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown labels found: {joined}")
