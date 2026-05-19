# Container Runtime Runbook

## Scope

Placeholder runbook for Docker, containerd, image pull, registry, and container runtime incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- Containers fail to start or restart repeatedly.
- Pods enter `ImagePullBackOff` or `ErrImagePull`.
- Registry authentication fails.
- Runtime disk pressure or image garbage collection affects workloads.

## First Checks

- Inspect image name, tag, digest, and registry credentials.
- Check container runtime logs on the affected node.
- Review recent base image, registry, or secret changes.
- Confirm whether the issue affects one image, one node, or all workloads.

## Useful Commands

```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get secret <image-pull-secret> -n <namespace>
crictl ps -a
crictl images
journalctl -u containerd --since "1 hour ago"
```

## Escalation Notes

Escalate to platform or registry ownership when multiple services cannot pull images, registry credentials are suspected, or runtime failures affect a node group.

## RAG Metadata

- Domain label: `container_runtime`
- Retrieval keywords: `container`, `docker`, `containerd`, `image pull`, `registry`, `ImagePullBackOff`, `runtime`
- Suggested citations: pod event inspection, image pull secret review, runtime logs

