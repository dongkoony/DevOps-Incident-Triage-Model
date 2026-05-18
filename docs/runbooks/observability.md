# Observability And Alerting Runbook

## Scope

Placeholder runbook for monitoring, logging, tracing, metrics, dashboards, and alerting incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- Alerts stop firing or fire too frequently.
- Dashboards show missing, delayed, or inconsistent metrics.
- Logs are unavailable for one service or namespace.
- Traces are incomplete or sampling behavior changes unexpectedly.

## First Checks

- Verify scrape targets, log collectors, and tracing exporters.
- Check recent alert rule, dashboard, or metric label changes.
- Compare raw metrics with dashboard queries.
- Confirm whether the issue is observability-only or reflects an actual service outage.

## Useful Commands

```bash
kubectl get pods -n monitoring
kubectl logs deployment/prometheus -n monitoring
kubectl logs daemonset/fluent-bit -n logging
curl -s http://localhost:9090/-/ready
```

## Escalation Notes

Escalate to observability ownership when alerting reliability is degraded, incident response visibility is reduced, or monitoring gaps affect production support.

## RAG Metadata

- Domain label: `observability_alerting`
- Retrieval keywords: `prometheus`, `grafana`, `alerts`, `logs`, `traces`, `metrics`, `scrape`, `dashboard`
- Suggested citations: scrape target checks, alert rule changes, collector health

