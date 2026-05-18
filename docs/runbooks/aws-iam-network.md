# AWS IAM And Network Runbook

## Scope

Placeholder runbook for AWS IAM, STS, VPC, routing, security group, and network connectivity incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- `AccessDenied`, `UnauthorizedOperation`, or `sts:AssumeRole` failures.
- Services cannot connect across VPC, subnet, or security group boundaries.
- DNS resolution, route table, or NAT gateway behavior changes.
- Deployments fail after IAM trust policy or OIDC changes.

## First Checks

- Verify the IAM role, policy, trust relationship, and caller identity.
- Check whether OIDC provider, audience, or branch conditions changed.
- Inspect route tables, security groups, NACLs, and DNS settings.
- Confirm whether the issue is regional, account-specific, or service-specific.

## Useful Commands

```bash
aws sts get-caller-identity
aws iam get-role --role-name <role-name>
aws iam simulate-principal-policy --policy-source-arn <arn> --action-names sts:AssumeRole
aws ec2 describe-route-tables
aws ec2 describe-security-groups
```

## Escalation Notes

Escalate to cloud platform or security ownership when access failures affect production deployment, cross-account access, or shared network infrastructure.

## RAG Metadata

- Domain label: `aws_iam_network`
- Retrieval keywords: `aws`, `iam`, `sts`, `assume role`, `vpc`, `security group`, `route table`, `oidc`
- Suggested citations: caller identity, trust policy, network path checks

