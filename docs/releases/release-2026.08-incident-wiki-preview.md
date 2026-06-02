# release-2026.08-incident-wiki-preview

## Channel

Preview

## Summary

This release train entry introduces a repository-backed LLM Wiki knowledge layer for incident context.

The feature is inspired by SRE Agent patterns where runbooks, SOPs, diagnostics, and knowledge base articles provide context for incident triage. The Wiki is indexed by the existing local retriever and consumed by `/retrieve` and `/assist`.

This is not a hosted LLM integration. No external LLM, PagerDuty, Confluence, Slack, Notion, or OpenAI API call is added in this release.

## Scope

Included:

- `docs/wiki/` knowledge base structure
- Wiki metadata schema
- Wiki pages for Kubernetes, CI/CD, AWS IAM/network, database, observability, container runtime, and deployment/release domains
- Wiki-aware retrieval evidence
- `/retrieve` and `/assist` evidence metadata enrichment

Excluded:

- External LLM generation
- PagerDuty, Confluence, Slack, Notion, or OpenAI API integration
- Production Vector DB deployment
- Automatic remediation execution
- Incident memory write-back

## API Impact

Existing evidence fields remain available:

- `document_id`
- `domain`
- `title`
- `section`
- `score`
- `citation`
- `excerpt`

New Wiki metadata fields are added:

- `wiki_id`
- `source_type`
- `service`
- `severity`
- `owner`
- `last_reviewed`
- `confidence_level`

## Validation Commands

```powershell
uv run --extra dev --extra api ruff check .
uv run --extra dev --extra api pytest -q
```

## Validation Results

- `uv run --extra dev --extra api ruff check .`: passed
- `uv run --extra dev --extra api pytest -q`: passed with `56 passed, 10 skipped`

## Known Limitations

- Wiki content is portfolio-grade placeholder knowledge.
- The retriever still uses local scikit-learn TF-IDF.
- No incident memory write-back is implemented.
- No external LLM generation is implemented.
- Wiki metadata is human-readable Markdown metadata, not a production CMS schema.
