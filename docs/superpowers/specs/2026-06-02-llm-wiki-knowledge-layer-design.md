# LLM Wiki Knowledge Layer Design

## Overview

This design introduces an LLM Wiki knowledge layer for the DevOps Incident Triage
Model. The goal is to evolve the current `docs/runbooks/` retrieval preview into a
more explicit incident knowledge system inspired by PagerDuty SRE Agent patterns:
runbooks, SOPs, diagnostics, service context, and learnings should be organized as
incident context that can support retrieval and assistant guidance.

The first implementation should not add a hosted LLM API call. It should make the
project more LLM-ready by giving the assistant a clearer knowledge model, richer
metadata, and more auditable citations.

## Reference Pattern

PagerDuty SRE Agent is the primary product reference for this direction. The relevant
patterns are:

- Ingest and analyze runbooks, SOPs, diagnostics, and logs.
- Provide relevant documentation and context for the affected service or architecture.
- Recommend diagnostic and remediation steps.
- Recall similar incidents and past resolutions over time.
- Save learnings after incidents are resolved.
- Structure runbooks by service and incident type.

This project will implement a portfolio-grade local equivalent: a repository-backed
LLM Wiki that can be indexed by the existing retriever and consumed by `/assist`.

## Goals

- Add a first-class `docs/wiki/` knowledge base structure.
- Treat runbooks as one type of Wiki source, not the whole knowledge system.
- Define Wiki metadata for domain, service, severity, source type, owner, review date,
  and confidence level.
- Extend retrieval evidence with Wiki metadata while preserving existing fields.
- Keep `/retrieve` and `/assist` backward compatible for existing clients.
- Keep all assistant guidance citation-grounded.
- Avoid external LLM, Vector DB, SaaS wiki, or PagerDuty integration in this first step.

## Non-Goals

- No OpenAI, Hugging Face Inference, PagerDuty, Confluence, Slack, or Notion API calls.
- No automatic remediation execution.
- No incident memory write-back from live incidents.
- No production Vector DB deployment.
- No claim that placeholder Wiki content reflects real production incidents.

## Proposed Release Track

Release name:

```text
release-2026.08-incident-wiki-preview
```

Channel:

```text
preview
```

Rationale:

The feature is a knowledge-layer preview. It makes retrieval and assistant output more
product-like, but it is not yet a full LLM-powered SRE agent.

## Knowledge Model

Wiki pages should be Markdown files with a structured metadata block near the top.
The metadata block stays human-readable and does not require YAML parsing in the first
iteration.

Example:

```markdown
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

Portfolio example for Kubernetes node readiness and CNI-related incidents.
```

Required metadata:

- `Wiki ID`
- `Source type`
- `Domain label`
- `Service`
- `Severity`
- `Owner`
- `Last reviewed`
- `Confidence level`

Source types:

- `runbook`
- `sop`
- `diagnostic`
- `past_incident`
- `service_profile`

Confidence levels:

- `placeholder`
- `preview`
- `validated`

## Proposed Directories

```text
docs/wiki/
  README.md
  kubernetes/
    node-readiness.md
  cicd/
    pipeline-failure.md
  aws-iam-network/
    role-assumption.md
  database/
    connection-saturation.md
  observability/
    alert-noise.md
  container-runtime/
    image-pull-failure.md
  deployment-release/
    rollback.md
```

The existing `docs/runbooks/` directory should remain in place for compatibility. The
new Wiki pages can reuse and expand the runbook content without deleting the old files.

## Retrieval Changes

The retriever should load both:

- `docs/runbooks/*.md`
- `docs/wiki/**/*.md`

Existing `RunbookRetriever` can remain as the public class name for now to avoid a large
rename. Internally, the corpus item can evolve from runbook-only sections to Wiki-aware
sections.

Add metadata fields to each retrieval section:

- `wiki_id`
- `source_type`
- `service`
- `severity`
- `owner`
- `last_reviewed`
- `confidence_level`

`RetrievedEvidence` should continue to expose:

- `document_id`
- `domain`
- `title`
- `section`
- `score`
- `citation`
- `excerpt`

It should additionally expose:

- `wiki_id`
- `source_type`
- `service`
- `severity`
- `owner`
- `last_reviewed`
- `confidence_level`

## API Changes

### `POST /retrieve`

The response remains backward compatible and adds optional Wiki metadata fields to
evidence items.

Example evidence item:

```json
{
  "document_id": "wiki-k8s-node-readiness",
  "wiki_id": "wiki-k8s-node-readiness",
  "source_type": "runbook",
  "domain": "k8s_cluster",
  "service": "kubernetes-platform",
  "severity": "medium",
  "owner": "platform-team",
  "last_reviewed": "2026-06-02",
  "confidence_level": "preview",
  "title": "Kubernetes Node Readiness Wiki",
  "section": "First Checks",
  "score": 0.87,
  "citation": "docs/wiki/kubernetes/node-readiness.md#first-checks",
  "excerpt": "Check node readiness, recent CNI changes, and kubelet status."
}
```

### `POST /assist`

The assistant response should continue to use retrieved evidence and citations. It can
now describe evidence as Wiki evidence and include source metadata in its retrieval
section.

No external LLM generation is added in this release.

## Evaluation And Observability

This preview should prepare for `release-2026.08-eval-observability` style metrics by
making Wiki source metadata available. Future evaluation can calculate:

- retrieval hit rate by source type
- citation coverage by source type
- stale page rate based on `last_reviewed`
- coverage by domain and service
- assistant recommendations with validated vs preview confidence levels

No new production metrics are required for the first Wiki PR unless the API response
changes need direct coverage.

## Testing Strategy

Tests should be TDD-first and cover:

- Wiki metadata parsing from Markdown.
- Corpus loading from `docs/wiki/**/*.md`.
- Backward compatibility for existing runbook loading.
- Retrieval evidence includes Wiki metadata.
- `/retrieve` returns Wiki metadata fields.
- `/assist` preserves citations and includes enriched evidence metadata.

## Documentation Updates

Update:

- `README.md`
- `README.ko.md`
- `docs/rag-roadmap.md`
- `docs/release-strategy.md`
- `CHANGELOG.md`

Create:

- `docs/wiki/README.md`
- `docs/releases/release-2026.08-incident-wiki-preview.md`

## Risks

- Adding Wiki metadata can make response schemas feel larger. Keep fields explicit and
  stable rather than adding nested arbitrary metadata.
- Wiki pages may look production-like. Every placeholder page must clearly state that it
  is a portfolio example, not a real production incident record.
- Renaming retrieval classes too early could create unnecessary churn. Keep compatibility
  first and rename later only if needed.

## Success Criteria

- Existing `/retrieve` and `/assist` tests continue to pass.
- New tests prove Wiki pages are indexed and returned with metadata.
- Existing runbook citations remain valid.
- Documentation clearly explains that LLM Wiki is a knowledge layer, not an external
  LLM integration.
- No `web/` directory changes are included.
