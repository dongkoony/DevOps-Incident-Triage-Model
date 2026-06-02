# LLM Wiki Knowledge Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository-backed LLM Wiki knowledge layer that enriches retrieval and assist evidence with Wiki metadata while preserving existing runbook behavior.

**Architecture:** Keep `RunbookRetriever` as the compatibility-facing retriever. Extend the corpus model so it can load both `docs/runbooks/*.md` and `docs/wiki/**/*.md`, parse human-readable Wiki metadata, and return enriched evidence fields through `/retrieve` and `/assist`. No external LLM, SaaS wiki, PagerDuty, Confluence, Slack, Notion, or Vector DB integration is added in this release.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pytest, scikit-learn TF-IDF retrieval, Markdown docs.

---

## File Structure

- Modify `src/devops_incident_triage/retrieval.py`: add Wiki metadata parsing, Wiki corpus loading, enriched section/evidence dataclasses, and combined corpus loading.
- Modify `src/devops_incident_triage/api.py`: add Wiki metadata fields to `RetrievedEvidenceItem` so `/retrieve` and `/assist` expose enriched evidence.
- Modify `tests/test_retrieval.py`: add TDD coverage for Wiki metadata parsing, Wiki corpus loading, and backward-compatible runbook metadata defaults.
- Modify `tests/test_api.py`: add response schema assertions for `/retrieve` and `/assist` Wiki metadata.
- Modify `tests/test_assist.py`: update test evidence construction with new metadata defaults if dataclass fields require it.
- Create `docs/wiki/README.md`: explain the LLM Wiki knowledge model and portfolio limitations.
- Create Wiki pages:
  - `docs/wiki/kubernetes/node-readiness.md`
  - `docs/wiki/cicd/pipeline-failure.md`
  - `docs/wiki/aws-iam-network/role-assumption.md`
  - `docs/wiki/database/connection-saturation.md`
  - `docs/wiki/observability/alert-noise.md`
  - `docs/wiki/container-runtime/image-pull-failure.md`
  - `docs/wiki/deployment-release/rollback.md`
- Create `docs/releases/release-2026.08-incident-wiki-preview.md`: release scope, validation commands, and limitations.
- Modify `README.md`, `README.ko.md`, `docs/rag-roadmap.md`, `docs/release-strategy.md`, and `CHANGELOG.md`: document the Wiki preview.
- Modify `docs/codex/session_state.md` and `docs/codex/worklog.md`: update cross-device handoff notes after implementation.

## Task 1: Wiki Metadata Parser

**Files:**
- Modify: `src/devops_incident_triage/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing parser test**

Append this test to `tests/test_retrieval.py`:

```python
from pathlib import Path


def test_parse_wiki_metadata_reads_required_fields(tmp_path: Path) -> None:
    wiki_page = tmp_path / "node-readiness.md"
    wiki_page.write_text(
        """# Kubernetes Node Readiness Wiki

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

Portfolio example for Kubernetes node readiness incidents.
""",
        encoding="utf-8",
    )

    metadata = parse_wiki_metadata(wiki_page)

    assert metadata.wiki_id == "wiki-k8s-node-readiness"
    assert metadata.source_type == "runbook"
    assert metadata.domain == "k8s_cluster"
    assert metadata.service == "kubernetes-platform"
    assert metadata.severity == "medium"
    assert metadata.owner == "platform-team"
    assert metadata.last_reviewed == "2026-06-02"
    assert metadata.confidence_level == "preview"
```

Update the import block in `tests/test_retrieval.py`:

```python
from devops_incident_triage.retrieval import (
    DEFAULT_RUNBOOK_DIR,
    RunbookRetriever,
    load_runbook_corpus,
    parse_wiki_metadata,
)
```

- [ ] **Step 2: Run the parser test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_retrieval.py::test_parse_wiki_metadata_reads_required_fields -q
```

Expected result:

```text
ImportError or AttributeError because parse_wiki_metadata does not exist.
```

- [ ] **Step 3: Implement the minimal metadata parser**

In `src/devops_incident_triage/retrieval.py`, add:

```python
@dataclass(frozen=True)
class WikiMetadata:
    wiki_id: str
    source_type: str
    domain: str
    service: str
    severity: str
    owner: str
    last_reviewed: str
    confidence_level: str


WIKI_METADATA_KEYS = {
    "Wiki ID": "wiki_id",
    "Source type": "source_type",
    "Domain label": "domain",
    "Service": "service",
    "Severity": "severity",
    "Owner": "owner",
    "Last reviewed": "last_reviewed",
    "Confidence level": "confidence_level",
}


def parse_wiki_metadata(path: Path) -> WikiMetadata:
    values: dict[str, str] = {}
    in_metadata = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "## Wiki Metadata":
            in_metadata = True
            continue
        if in_metadata and line.startswith("## "):
            break
        if not in_metadata or not line.startswith("- ") or ":" not in line:
            continue

        key, raw_value = line.removeprefix("- ").split(":", 1)
        field_name = WIKI_METADATA_KEYS.get(key.strip())
        if field_name:
            values[field_name] = raw_value.strip().strip("`")

    missing = sorted(set(WIKI_METADATA_KEYS.values()) - set(values))
    if missing:
        raise RetrievalConfigurationError(
            f"Wiki metadata missing required fields in {path}: {', '.join(missing)}"
        )

    return WikiMetadata(**values)
```

- [ ] **Step 4: Run the parser test and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_retrieval.py::test_parse_wiki_metadata_reads_required_fields -q
```

Expected result:

```text
1 passed
```

## Task 2: Wiki Pages And Corpus Loading

**Files:**
- Create: `docs/wiki/README.md`
- Create: `docs/wiki/kubernetes/node-readiness.md`
- Create: `docs/wiki/cicd/pipeline-failure.md`
- Create: `docs/wiki/aws-iam-network/role-assumption.md`
- Create: `docs/wiki/database/connection-saturation.md`
- Create: `docs/wiki/observability/alert-noise.md`
- Create: `docs/wiki/container-runtime/image-pull-failure.md`
- Create: `docs/wiki/deployment-release/rollback.md`
- Modify: `src/devops_incident_triage/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing corpus tests**

Append these tests to `tests/test_retrieval.py`:

```python
def test_load_runbook_corpus_includes_wiki_pages() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    wiki_sections = [
        item for item in corpus if item.wiki_id == "wiki-k8s-node-readiness"
    ]

    assert wiki_sections
    assert wiki_sections[0].document_id == "wiki-k8s-node-readiness"
    assert wiki_sections[0].source_type == "runbook"
    assert wiki_sections[0].service == "kubernetes-platform"
    assert wiki_sections[0].confidence_level == "preview"
    assert wiki_sections[0].citation.startswith("docs/wiki/kubernetes/node-readiness.md#")


def test_existing_runbooks_get_backward_compatible_wiki_defaults() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    runbook = next(
        item
        for item in corpus
        if item.document_id == "runbook-kubernetes" and item.section == "First Checks"
    )

    assert runbook.wiki_id == "runbook-kubernetes"
    assert runbook.source_type == "runbook"
    assert runbook.service == "general"
    assert runbook.severity == "unknown"
    assert runbook.owner == "portfolio"
    assert runbook.last_reviewed == "unknown"
    assert runbook.confidence_level == "placeholder"
```

- [ ] **Step 2: Run the corpus tests and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_retrieval.py::test_load_runbook_corpus_includes_wiki_pages tests/test_retrieval.py::test_existing_runbooks_get_backward_compatible_wiki_defaults -q
```

Expected result:

```text
Failures because Wiki pages do not exist and RunbookSection has no Wiki metadata fields.
```

- [ ] **Step 3: Create the Wiki README**

Create `docs/wiki/README.md`:

```markdown
# LLM Wiki

This directory contains portfolio-grade DevOps incident knowledge pages for the DevOps
Incident Triage Assistant.

The Wiki is inspired by SRE Agent patterns where runbooks, SOPs, diagnostics, service
profiles, and incident learnings provide context for triage. These pages are local
examples, not real production incident records.

Each Wiki page includes a `Wiki Metadata` section so retrieval and assistant responses
can return auditable source context.

Required metadata:

- Wiki ID
- Source type
- Domain label
- Service
- Severity
- Owner
- Last reviewed
- Confidence level
```

- [ ] **Step 4: Create Wiki pages**

Create `docs/wiki/kubernetes/node-readiness.md`:

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
```

Create `docs/wiki/cicd/pipeline-failure.md`:

```markdown
# CI/CD Pipeline Failure Wiki

## Wiki Metadata

- Wiki ID: `wiki-cicd-pipeline-failure`
- Source type: `runbook`
- Domain label: `cicd_pipeline`
- Service: `delivery-platform`
- Severity: `medium`
- Owner: `devex-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for CI/CD runner, workflow, artifact, and deployment pipeline failures.
This is not a real production incident record.

## Common Symptoms

- Pipeline jobs fail after dependency, runner image, or credential changes.
- Artifacts are missing between build and deploy stages.
- Deployment jobs time out while earlier build jobs pass.

## First Checks

- Check the failed job logs and compare them with the last successful run.
- Confirm whether runner image, dependency lockfiles, or secrets changed.
- Verify artifact upload and download steps between stages.

## Useful Commands

```bash
gh run view <run-id> --log
gh run list --branch <branch-name>
git diff HEAD~1 -- .github/workflows
```

## Escalation Notes

Escalate to the delivery platform or service owner when multiple repositories fail, when
deployment credentials are involved, or when a rollback requires production approval.
```

Create `docs/wiki/aws-iam-network/role-assumption.md`:

```markdown
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
```

Create `docs/wiki/database/connection-saturation.md`:

```markdown
# Database Connection Saturation Wiki

## Wiki Metadata

- Wiki ID: `wiki-database-connection-saturation`
- Source type: `runbook`
- Domain label: `database_state`
- Service: `database-platform`
- Severity: `medium`
- Owner: `data-platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for database connection pool saturation, lock contention, and query
timeout incidents. This is not a real production incident record.

## Common Symptoms

- Application requests time out while database CPU is not fully saturated.
- Connection pool usage reaches the configured maximum.
- Writes or migrations wait on long-running locks.

## First Checks

- Check active connections and pool usage by application instance.
- Inspect long-running queries and lock waits.
- Compare the incident with recent deploys, migrations, or traffic spikes.

## Useful Commands

```bash
psql -c "select state, count(*) from pg_stat_activity group by state;"
psql -c "select pid, wait_event_type, wait_event, query from pg_stat_activity where wait_event is not null;"
```

## Escalation Notes

Escalate to the database owner before terminating sessions, changing pool sizes, or
rolling back migrations in production.
```

Create `docs/wiki/observability/alert-noise.md`:

```markdown
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
```

Create `docs/wiki/container-runtime/image-pull-failure.md`:

```markdown
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
```

Create `docs/wiki/deployment-release/rollback.md`:

```markdown
# Deployment Rollback Wiki

## Wiki Metadata

- Wiki ID: `wiki-deployment-rollback`
- Source type: `sop`
- Domain label: `deployment_release`
- Service: `release-management`
- Severity: `medium`
- Owner: `release-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for release rollback decision support, failed rollout triage, and
deployment safety checks. This is not a real production incident record.

## Common Symptoms

- Error rate or latency increases immediately after a deployment.
- A canary or blue-green rollout fails health checks.
- Rollback is considered but database or schema changes may not be reversible.

## First Checks

- Compare the release version with error rate, latency, and saturation changes.
- Confirm whether the deployment includes schema, migration, or feature flag changes.
- Check rollback safety notes before reverting production traffic.

## Useful Commands

```bash
kubectl rollout status deployment/<deployment-name> -n <namespace>
kubectl rollout undo deployment/<deployment-name> -n <namespace>
git log --oneline -5
```

## Escalation Notes

Escalate to the release owner before rollback when the deployment includes migrations,
shared dependencies, or customer-visible data changes.
```

- [ ] **Step 5: Extend corpus dataclasses and loading**

In `src/devops_incident_triage/retrieval.py`:

- Add `DEFAULT_WIKI_DIR = Path("docs/wiki")`.
- Add Wiki metadata fields to `RunbookSection` and `RetrievedEvidence`.
- Add a helper that creates default metadata for existing runbook files.
- Load `docs/wiki/**/*.md` in addition to `docs/runbooks/*.md`.
- Use `parse_wiki_metadata()` for Wiki pages.
- Use Wiki metadata values for `document_id`, `domain`, and citation generation.

- [ ] **Step 6: Run retrieval tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_retrieval.py -q
```

Expected result:

```text
All tests in tests/test_retrieval.py pass.
```

## Task 3: API Evidence Schema Enrichment

**Files:**
- Modify: `src/devops_incident_triage/api.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_assist.py`

- [ ] **Step 1: Write failing API assertions**

Update `test_retrieve_returns_cited_runbook_evidence()` in `tests/test_api.py` with:

```python
    evidence = payload["evidence"][0]
    assert "wiki_id" in evidence
    assert "source_type" in evidence
    assert "service" in evidence
    assert "severity" in evidence
    assert "owner" in evidence
    assert "last_reviewed" in evidence
    assert "confidence_level" in evidence
```

Update `test_assist_returns_classifier_retrieval_and_guidance()` with:

```python
    assist_evidence = payload["retrieval"]["evidence"][0]
    assert "wiki_id" in assist_evidence
    assert "source_type" in assist_evidence
    assert "confidence_level" in assist_evidence
```

- [ ] **Step 2: Run API tests and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_api.py::test_retrieve_returns_cited_runbook_evidence tests/test_api.py::test_assist_returns_classifier_retrieval_and_guidance -q
```

Expected result:

```text
Failures because RetrievedEvidenceItem does not expose Wiki metadata fields.
```

- [ ] **Step 3: Extend Pydantic schema and conversion**

In `src/devops_incident_triage/api.py`, add these fields to `RetrievedEvidenceItem`:

```python
    wiki_id: str
    source_type: str
    service: str
    severity: str
    owner: str
    last_reviewed: str
    confidence_level: str
```

Update every `RetrievedEvidenceItem(...)` construction, including `_to_retrieved_evidence_item()`, to pass the new fields from the retrieval evidence object.

If `tests/test_assist.py` constructs `RetrievedEvidence` directly, update each instance with:

```python
                wiki_id="runbook-aws-iam-network",
                source_type="runbook",
                service="general",
                severity="unknown",
                owner="portfolio",
                last_reviewed="unknown",
                confidence_level="placeholder",
```

- [ ] **Step 4: Run API and assist tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest tests/test_api.py tests/test_assist.py -q
```

Expected result:

```text
All tests in tests/test_api.py and tests/test_assist.py pass.
```

## Task 4: Documentation And Release Evidence

**Files:**
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `docs/rag-roadmap.md`
- Modify: `docs/release-strategy.md`
- Modify: `CHANGELOG.md`
- Create: `docs/releases/release-2026.08-incident-wiki-preview.md`
- Modify: `docs/codex/session_state.md`
- Modify: `docs/codex/worklog.md`

- [ ] **Step 1: Update README files**

Add a concise `LLM Wiki Preview` subsection near the RAG/Assist sections:

```markdown
### LLM Wiki Preview

`release-2026.08-incident-wiki-preview` introduces a repository-backed LLM Wiki
knowledge layer inspired by SRE Agent patterns. Wiki pages organize runbooks, SOP-like
checks, diagnostics, and service context with explicit metadata.

This preview does not call an external LLM API. It enriches `/retrieve` and `/assist`
evidence with Wiki metadata so future LLM responses can stay citation-grounded.
```

Add the Korean equivalent to `README.ko.md`.

- [ ] **Step 2: Update roadmap and release strategy**

In `docs/rag-roadmap.md`, describe Wiki as the knowledge layer above runbooks:

```text
LLM Wiki is the product-facing knowledge layer. RAG retrieval remains the mechanism
that searches Wiki pages and returns cited evidence.
```

In `docs/release-strategy.md`, add `release-2026.08-incident-wiki-preview` or update
the 2026.08 row to reflect incident Wiki preview plus evaluation preparation.

- [ ] **Step 3: Update CHANGELOG**

Add:

```markdown
## release-2026.08-incident-wiki-preview

### Status

In progress preview release.

### Added

- Repository-backed LLM Wiki structure
- Wiki metadata schema for incident knowledge pages
- Wiki-aware retrieval evidence
- `/retrieve` and `/assist` response metadata enrichment

### Notes

- No external LLM API is called in this release.
- Wiki pages are portfolio examples, not real production incident records.
```

- [ ] **Step 4: Create release evidence**

Create `docs/releases/release-2026.08-incident-wiki-preview.md` with:

```markdown
# release-2026.08-incident-wiki-preview

## Channel

Preview

## Summary

This release train entry introduces a repository-backed LLM Wiki knowledge layer for
incident context.

The feature is inspired by SRE Agent patterns where runbooks, SOPs, diagnostics, and
knowledge base articles provide context for incident triage.

## Scope

Included:

- `docs/wiki/` knowledge base structure
- Wiki metadata schema
- Wiki-aware retrieval evidence
- `/retrieve` and `/assist` evidence metadata enrichment

Excluded:

- External LLM generation
- PagerDuty, Confluence, Slack, Notion, or OpenAI API integration
- Production Vector DB deployment
- Automatic remediation execution

## Validation Commands

```powershell
uv run --extra dev --extra api ruff check .
uv run --extra dev --extra api pytest -q
```

## Known Limitations

- Wiki content is portfolio-grade placeholder knowledge.
- The retriever still uses local TF-IDF.
- No incident memory write-back is implemented.
```

- [ ] **Step 5: Update Codex handoff docs**

Update `docs/codex/session_state.md`:

```markdown
- Current feature branch: `feature/llm-wiki-knowledge-layer`.
- Current feature focus: implement `release-2026.08-incident-wiki-preview` with a repository-backed LLM Wiki knowledge layer and Wiki-aware retrieval evidence.
```

Append a 2026-06-02 worklog note after implementation verification.

## Task 5: Final Verification And Commit

**Files:**
- All changed files

- [ ] **Step 1: Run lint**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api ruff check .
```

Expected result:

```text
All checks passed!
```

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'devops-incident-triage\uv\.venv'
uv run --extra dev --extra api pytest -q
```

Expected result:

```text
All tests pass with zero failures.
```

- [ ] **Step 3: Review scope**

Run:

```powershell
git status --short
git diff --stat
```

Expected result:

```text
Only retrieval, API, tests, Wiki docs, release docs, README/roadmap/changelog, and Codex handoff files changed. No web/ changes.
```

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add src/devops_incident_triage/retrieval.py src/devops_incident_triage/api.py tests/test_retrieval.py tests/test_api.py tests/test_assist.py docs/wiki docs/releases/release-2026.08-incident-wiki-preview.md README.md README.ko.md docs/rag-roadmap.md docs/release-strategy.md CHANGELOG.md docs/codex/session_state.md docs/codex/worklog.md docs/superpowers/plans/2026-06-02-llm-wiki-knowledge-layer.md
git commit -m "feat: add llm wiki knowledge layer"
```
