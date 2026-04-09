# Demo Showcase Report

Curated prediction examples for portfolio demos and repository sharing.

- Generated at: `2026-04-09T04:20:36.563497+00:00`
- Model path: `models/devops-incident-triage-smoke`
- Confidence threshold: `0.6`

## Summary

- Total examples: `7`
- Expected label matches: `2`
- Mismatches: `5`
- Human review count: `7`

## Predictions

| Title | Expected | Predicted | Confidence | Review Required | Queue |
|---|---|---|---:|---|---|
| EKS nodes not ready after CNI upgrade | k8s_cluster | k8s_cluster | 0.1516 | yes | sre_manual_triage |
| GitHub Actions deploy blocked by IAM | aws_iam_network | deployment_release | 0.1513 | yes | sre_manual_triage |
| Helm rollout timeout during release | deployment_release | deployment_release | 0.1526 | yes | sre_manual_triage |
| Container image pull back-off | container_runtime | k8s_cluster | 0.1510 | yes | sre_manual_triage |
| Prometheus alerts missing after config reload | observability_alerting | deployment_release | 0.1573 | yes | sre_manual_triage |
| Primary database locked during migration | database_state | aws_iam_network | 0.1520 | yes | sre_manual_triage |
| Ambiguous deploy failure with timeout and permission denied | deployment_release | k8s_cluster | 0.1533 | yes | sre_manual_triage |

## Review-Required Examples

- **EKS nodes not ready after CNI upgrade**: Clear Kubernetes cluster state issue.
- **GitHub Actions deploy blocked by IAM**: Permission and AWS identity failure.
- **Helm rollout timeout during release**: Release orchestration issue.
- **Container image pull back-off**: Container runtime and image retrieval issue.
- **Prometheus alerts missing after config reload**: Monitoring and alerting visibility issue.
- **Primary database locked during migration**: Database lock and write path issue.
- **Ambiguous deploy failure with timeout and permission denied**: Intentionally ambiguous example for human review.
