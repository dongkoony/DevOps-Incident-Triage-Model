# Kubernetes Node Readiness Wiki

## Wiki Metadata

- Wiki ID: `wiki-k8s-node-readiness`
- Source type: `runbook`
- Domain label: `k8s_cluster`
- Service: `kubernetes-platform`
- Severity: `medium`
- Owner: `platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for Kubernetes node readiness, CNI, kubelet, and scheduling incidents.
This is not a real production incident record.

## Common Symptoms

- Nodes report `NotReady`.
- Pods remain in `Pending` after a cluster or CNI change.
- Deployments stall because replicas cannot be scheduled.

## First Checks

- Check node readiness and recent node events.
- Inspect CNI and kubelet changes near the incident start time.
- Compare affected nodes by node group, namespace, and availability zone.

## Useful Commands

```bash
kubectl get nodes -o wide
kubectl describe node <node-name>
kubectl get events -A --sort-by=.lastTimestamp
```

## Escalation Notes

Escalate to the platform ownership team when multiple nodes are affected or scheduling
is blocked across namespaces.
