# Container Image Pull Failure Wiki

## Wiki Metadata

- Wiki ID: `wiki-container-image-pull-failure`
- Source type: `runbook`
- Domain label: `container_runtime`
- Service: `container-platform`
- Severity: `medium`
- Owner: `platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for image pull, registry authentication, image tag, and runtime startup
failures. This is not a real production incident record.

## Common Symptoms

- Pods enter `ImagePullBackOff` or `ErrImagePull`.
- A deployment references a tag that is missing from the registry.
- Registry credentials expire or are not mounted in the target namespace.

## First Checks

- Verify the image name, tag, digest, and registry path.
- Check image pull secret availability in the namespace.
- Compare runtime events with recent registry or deployment changes.

## Useful Commands

```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get secret -n <namespace>
docker pull <image-ref>
```

## Escalation Notes

Escalate to the platform or registry owner when credentials, registry availability, or
production deployment rollout is affected.
