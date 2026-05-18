# Kubernetes Cluster Runbook

## Scope

Placeholder runbook for Kubernetes scheduling, node readiness, kubelet, CNI, and cluster-state incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- Nodes report `NotReady`.
- Pods remain in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff`.
- Deployments stall because replicas cannot be scheduled.
- Recent CNI, node image, or cluster add-on changes correlate with the incident.

## First Checks

- Check node readiness and recent node events.
- Inspect pod events for scheduling, image, and volume errors.
- Review recent CNI, kubelet, or cluster add-on changes.
- Confirm whether the issue is isolated to one namespace, node group, or availability zone.

## Useful Commands

```bash
kubectl get nodes -o wide
kubectl describe node <node-name>
kubectl get pods -A --field-selector=status.phase=Pending
kubectl describe pod <pod-name> -n <namespace>
kubectl get events -A --sort-by=.lastTimestamp
```

## Escalation Notes

Escalate to the platform or Kubernetes ownership team when multiple nodes are affected, scheduling is blocked across namespaces, or CNI changes were recently deployed.

## RAG Metadata

- Domain label: `k8s_cluster`
- Retrieval keywords: `kubernetes`, `k8s`, `node`, `NotReady`, `CNI`, `pending`, `scheduler`, `kubelet`
- Suggested citations: node readiness checks, pod event inspection, CNI change review

