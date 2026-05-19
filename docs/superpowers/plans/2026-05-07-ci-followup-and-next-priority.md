# CI Follow-Up And Next Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the post-showcase cleanup, verify the hardened CI paths, and pick one focused next implementation target.

**Architecture:** Treat the day as two phases: first stabilize and verify the current `develop` branch, then decide the next small feature branch. Keep operational cleanup separate from feature implementation so each commit stays coherent.

**Tech Stack:** Git, GitHub Actions, uv, pytest, Ruff, FastAPI, Transformers, Docker decision notes.

---

## File Map

- `docs/todo.md`: Tracks today's operational checklist and links to this plan.
- `docs/codex/session_state.md`: Update only if focus, active risks, or next actions change.
- `docs/codex/worklog.md`: Append the final dated execution note before closeout.
- `docs/codex/decisions.md`: Add a durable decision only if Docker smoke scope or workflow policy changes.
- `.github/workflows/ci.yml`: Read during verification; modify only if the Docker smoke decision becomes implementation work today.
- `README.md`: Re-review showcase command and generated artifact descriptions.
- `README.ko.md`: Re-review Korean showcase command and generated artifact descriptions.
- `reports/demo_showcase.md`: Compare sample report wording and path behavior against current CLI output.
- `src/devops_incident_triage/demo_showcase.py`: Read if documentation and actual behavior disagree.
- `src/devops_incident_triage/api.py`: Read if API smoke behavior or health payload differs from docs.

---

### Task 1: Stabilize The Local `develop` Branch

**Files:**
- Modify: none
- Test: git command output

- [x] **Step 1: Confirm current branch and cleanliness**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## develop...origin/develop
```

- [x] **Step 2: Pull latest tags and `develop`**

Run:

```powershell
git pull --tags origin develop
```

Expected:

```text
Already up to date.
```

- [x] **Step 3: Inspect the old safety stash**

Run:

```powershell
git stash show --stat "stash@{0}"
git stash show -p "stash@{0}"
```

Expected:

```text
.gitignore
Makefile
README.ko.md
README.md
pyproject.toml
```

- [x] **Step 4: Decide stash handling**

If the stash only contains changes already merged or intentionally ignored, drop it:

```powershell
git stash drop "stash@{0}"
```

Expected:

```text
Dropped stash@{0}
```

If any change still matters, create a recovery branch before applying it:

```powershell
git switch -c feature/recover-local-workflow-stash
git stash apply "stash@{0}"
git status --short
```

Expected:

```text
M .gitignore
M Makefile
M README.ko.md
M README.md
M pyproject.toml
```

---

### Task 2: Re-Run Local CI Smoke Checks

**Files:**
- Modify: none unless a verification failure reveals a real issue
- Test: local lint, unit tests, showcase smoke, API smoke

- [ ] **Step 1: Run Ruff**

Run:

```powershell
uv run ruff check .
```

Expected:

```text
All checks passed!
```

- [ ] **Step 2: Run the test suite**

Run:

```powershell
uv run pytest -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run showcase smoke with the portable CI model reference**

Run:

```powershell
uv run ditri-demo-showcase --model-path hf-internal-testing/tiny-random-distilbert --confidence-threshold 0.6 --review-queue sre_manual_triage --output-json reports/demo_showcase.local.json --output-markdown reports/demo_showcase.local.md
```

Expected:

```text
Demo Showcase
reports/demo_showcase.local.json
reports/demo_showcase.local.md
```

- [ ] **Step 4: Remove local smoke artifacts if they are generated**

Run:

```powershell
Remove-Item -LiteralPath reports/demo_showcase.local.json,reports/demo_showcase.local.md -ErrorAction SilentlyContinue
```

Expected:

```text
No output
```

- [ ] **Step 5: Run FastAPI smoke manually if CI behavior needs local confirmation**

Run one terminal:

```powershell
$env:MODEL_PATH="hf-internal-testing/tiny-random-distilbert"; uv run ditri-api
```

Run another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"text":"Kubernetes pods are crash looping after the latest deployment."}'
Invoke-WebRequest http://127.0.0.1:8000/metrics
```

Expected:

```text
/health returns status and model_path
/predict returns label, confidence, and routing fields
/metrics returns Prometheus text
```

---

### Task 3: Re-Review Showcase Documentation

**Files:**
- Modify: `README.md` only if drift is found
- Modify: `README.ko.md` only if drift is found
- Modify: `reports/demo_showcase.md` only if committed sample output is stale
- Read: `src/devops_incident_triage/demo_showcase.py`

- [ ] **Step 1: Search all showcase references**

Run:

```powershell
rg "ditri-demo-showcase|demo_showcase|Demo Showcase|데모 쇼케이스" README.md README.ko.md reports src tests docs
```

Expected:

```text
README.md
README.ko.md
reports/demo_showcase.md
src/devops_incident_triage/demo_showcase.py
tests/test_demo_showcase.py
```

- [ ] **Step 2: Compare documented options to parser behavior**

Run:

```powershell
uv run ditri-demo-showcase --help
```

Expected:

```text
--model-path
--confidence-threshold
--review-queue
--output-json
--output-markdown
```

- [ ] **Step 3: Patch docs only if commands or artifact paths are wrong**

If README commands still imply a local-only model path for CI-style verification, update the wording to distinguish:

```markdown
For local trained-model demos, use `models/devops-incident-triage`.
For CI smoke checks, use `hf-internal-testing/tiny-random-distilbert`.
```

- [ ] **Step 4: Verify documentation changes**

Run:

```powershell
uv run ruff check .
uv run pytest -q tests/test_demo_showcase.py
```

Expected:

```text
All checks passed!
passed
```

---

### Task 4: Decide Docker Smoke Scope

**Files:**
- Modify: `docs/codex/decisions.md` if the decision changes or becomes durable
- Modify: `.github/workflows/ci.yml` only if implementing Docker smoke today
- Test: `docker build` if Docker is available

- [ ] **Step 1: Check whether Docker is available locally**

Run:

```powershell
docker version
```

Expected if available:

```text
Client:
Server:
```

Expected if unavailable:

```text
error during connect
```

- [ ] **Step 2: Keep the current policy unless there is a reason to change**

Current policy:

```text
Do not run Docker smoke on every PR.
Consider Docker smoke for pushes to develop, release branches, or manual dispatch.
```

- [ ] **Step 3: If implementing today, create a feature branch**

Run:

```powershell
git switch -c feature/docker-smoke-workflow
```

Expected:

```text
Switched to a new branch 'feature/docker-smoke-workflow'
```

- [ ] **Step 4: Add only a conditional Docker build job**

Target behavior:

```text
Run docker-smoke on workflow_dispatch and pushes to develop or release/*.
Skip docker-smoke for normal pull_request events.
```

- [ ] **Step 5: Verify the workflow edit**

Run:

```powershell
uv run ruff check .
uv run pytest -q
```

Expected:

```text
All checks passed!
passed
```

---

### Task 5: Pick The Next Product-Facing Priority

**Files:**
- Modify: `docs/todo.md`
- Modify: `docs/codex/session_state.md`
- Create: one new spec under `docs/superpowers/specs/` if implementation starts

- [ ] **Step 1: Use this decision matrix**

```text
confidence threshold tuning:
  Choose if the next goal is model quality, evaluation discipline, and clearer operational thresholds.

showcase UX polish:
  Choose if the next goal is portfolio readability, demo flow, and recruiter-friendly presentation.

benchmark/report polish:
  Choose if the next goal is stronger ML engineering evidence and model comparison credibility.
```

- [ ] **Step 2: Recommended choice for today**

```text
Choose benchmark/report polish.
```

Reason:

```text
CI and showcase are now in place. The strongest next portfolio signal is a better benchmark/report story that explains why the selected model is credible.
```

- [ ] **Step 3: Write the next spec if proceeding**

Create:

```text
docs/superpowers/specs/2026-05-07-benchmark-report-polish.md
```

Minimum sections:

```markdown
# Benchmark Report Polish Spec

## Goal

## User Value

## Current Behavior

## Proposed Behavior

## Non-Goals

## Acceptance Criteria

## Verification
```

---

### Task 6: Close The Session Cleanly

**Files:**
- Modify: `docs/codex/session_state.md`
- Modify: `docs/codex/worklog.md`
- Modify: `docs/todo.md`

- [ ] **Step 1: Update session state**

Set:

```markdown
Last updated: 2026-05-07
```

Add the current focus and next action based on the chosen priority.

- [ ] **Step 2: Append worklog entry**

Add:

```markdown
## 2026-05-07

- Revalidated the current `develop` branch after the CI smoke model fix.
- Reviewed showcase documentation against current CLI behavior.
- Decided the next implementation priority and recorded the next action.
```

- [ ] **Step 3: Final verification**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## develop...origin/develop
```

Or, if a feature branch was created:

```text
## feature/<name>
M docs/...
```

---

## Execution Recommendation

Do Task 1 and Task 2 first. They establish that the current base is healthy.

Then do Task 3. If docs are already accurate, skip code changes and keep the day lightweight.

For Task 4, keep Docker smoke as a documented follow-up unless Docker is already running locally and the GitHub Actions workflow needs this portfolio signal today.

For Task 5, the recommended next feature is `benchmark/report polish`.
