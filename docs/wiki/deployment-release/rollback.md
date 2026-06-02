# Deployment Rollback Wiki

## Wiki Metadata

- Wiki ID: `wiki-deployment-rollback`
- Source type: `sop`
- Domain label: `deployment_release`
- Service: `release-management`
- Severity: `medium`
- Owner: `release-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for release rollback decision support, failed rollout triage, and
deployment safety checks. This is not a real production incident record.

## Common Symptoms

- Error rate or latency increases immediately after a deployment.
- A canary or blue-green rollout fails health checks.
- Rollback is considered but database or schema changes may not be reversible.

## First Checks

- Compare the release version with error rate, latency, and saturation changes.
- Confirm whether the deployment includes schema, migration, or feature flag changes.
- Check rollback safety notes before reverting production traffic.

## Useful Commands

```bash
kubectl rollout status deployment/<deployment-name> -n <namespace>
kubectl rollout undo deployment/<deployment-name> -n <namespace>
git log --oneline -5
```

## Escalation Notes

Escalate to the release owner before rollback when the deployment includes migrations,
shared dependencies, or customer-visible data changes.
