# CI/CD Pipeline Failure Wiki

## Wiki Metadata

- Wiki ID: `wiki-cicd-pipeline-failure`
- Source type: `runbook`
- Domain label: `cicd_pipeline`
- Service: `delivery-platform`
- Severity: `medium`
- Owner: `devex-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for CI/CD runner, workflow, artifact, and deployment pipeline failures.
This is not a real production incident record.

## Common Symptoms

- Pipeline jobs fail after dependency, runner image, or credential changes.
- Artifacts are missing between build and deploy stages.
- Deployment jobs time out while earlier build jobs pass.

## First Checks

- Check the failed job logs and compare them with the last successful run.
- Confirm whether runner image, dependency lockfiles, or secrets changed.
- Verify artifact upload and download steps between stages.

## Useful Commands

```bash
gh run view <run-id> --log
gh run list --branch <branch-name>
git diff HEAD~1 -- .github/workflows
```

## Escalation Notes

Escalate to the delivery platform or service owner when multiple repositories fail, when
deployment credentials are involved, or when a rollback requires production approval.
