# AWS IAM Role Assumption Wiki

## Wiki Metadata

- Wiki ID: `wiki-aws-iam-role-assumption`
- Source type: `runbook`
- Domain label: `aws_iam_network`
- Service: `aws-platform`
- Severity: `medium`
- Owner: `cloud-platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for IAM role assumption, OIDC trust, permissions, and network access
failures. This is not a real production incident record.

## Common Symptoms

- Deployment jobs fail with `AccessDenied` or `sts:AssumeRole` errors.
- A workload can authenticate but cannot access a target AWS service.
- Network policy or routing changes correlate with failed AWS API calls.

## First Checks

- Verify the role ARN, trust policy, and OIDC audience conditions.
- Check whether IAM policies or permission boundaries changed recently.
- Confirm the request is coming from the expected branch, account, and environment.

## Useful Commands

```bash
aws sts get-caller-identity
aws iam get-role --role-name <role-name>
aws iam simulate-principal-policy --policy-source-arn <role-arn> --action-names sts:AssumeRole
```

## Escalation Notes

Escalate to the cloud platform owner when IAM trust policy changes, cross-account access,
or production network routing is involved.
