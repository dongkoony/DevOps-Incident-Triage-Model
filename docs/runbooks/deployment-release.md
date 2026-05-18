# Deployment And Release Runbook

## Scope

Placeholder runbook for Helm, rollout, release orchestration, deployment timeout, and rollback incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- Helm upgrades time out or remain pending.
- Deployments do not reach the desired number of available replicas.
- Rollbacks fail or leave resources in a partial state.
- Release changes correlate with readiness, probe, or configuration errors.

## First Checks

- Check deployment rollout status and replica availability.
- Review Helm release history and rendered values.
- Inspect pod readiness, liveness, and startup probes.
- Compare the failed release with the last successful release.

## Useful Commands

```bash
kubectl rollout status deployment/<deployment-name> -n <namespace>
kubectl describe deployment <deployment-name> -n <namespace>
helm history <release-name> -n <namespace>
helm get values <release-name> -n <namespace>
helm rollback <release-name> <revision> -n <namespace>
```

## Escalation Notes

Escalate to release engineering or service ownership when rollback is unsafe, production availability is impacted, or the release requires coordinated application changes.

## RAG Metadata

- Domain label: `deployment_release`
- Retrieval keywords: `deployment`, `release`, `helm`, `rollout`, `rollback`, `readiness`, `timeout`
- Suggested citations: rollout status, Helm history, probe diagnostics

