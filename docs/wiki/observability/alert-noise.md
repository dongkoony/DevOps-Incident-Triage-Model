# Observability Alert Noise Wiki

## Wiki Metadata

- Wiki ID: `wiki-observability-alert-noise`
- Source type: `diagnostic`
- Domain label: `observability_alerting`
- Service: `observability-platform`
- Severity: `low`
- Owner: `sre-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for noisy alerts, duplicate pages, metric gaps, and alert routing
issues. This is not a real production incident record.

## Common Symptoms

- Multiple alerts fire for one underlying service degradation.
- Alert severity does not match customer impact.
- Dashboards show missing or delayed metrics during an incident.

## First Checks

- Group alerts by service, region, and deployment version.
- Check alert rule changes and notification routing changes.
- Verify whether metric ingestion delay or label cardinality changed.

## Useful Commands

```bash
curl -s http://localhost:8000/metrics
promtool check rules <rules-file>
```

## Escalation Notes

Escalate to SRE ownership when paging noise hides customer impact or when alert rule
changes could suppress real incidents.
