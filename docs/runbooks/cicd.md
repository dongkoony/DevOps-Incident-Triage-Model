# CI/CD Pipeline Runbook

## Scope

Placeholder runbook for build, test, deployment pipeline, runner, and artifact publishing failures. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- GitHub Actions, GitLab CI, or Jenkins jobs fail unexpectedly.
- Deployment jobs cannot fetch credentials or artifacts.
- Tests pass locally but fail in CI.
- Runners are unavailable, misconfigured, or missing required permissions.

## First Checks

- Identify the failing stage and compare it with the last successful run.
- Check whether dependencies, secrets, runners, or branch protections changed.
- Review logs for permission, timeout, artifact, and cache errors.
- Confirm whether the issue affects one workflow or all pipelines.

## Useful Commands

```bash
gh run list --limit 10
gh run view <run-id> --log
git diff HEAD~1 -- .github/workflows
git status --short --branch
```

## Escalation Notes

Escalate to DevOps or release engineering when the failure blocks production deployment, affects multiple repositories, or involves secrets and environment protection settings.

## RAG Metadata

- Domain label: `cicd_pipeline`
- Retrieval keywords: `ci`, `cd`, `pipeline`, `runner`, `workflow`, `artifact`, `cache`, `deployment job`
- Suggested citations: failing stage identification, runner status, workflow diff

