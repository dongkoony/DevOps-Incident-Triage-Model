# Demo Showcase Design

## Goal

Add a shareable demo workflow that lets the project owner show model behavior in two ways from the same curated examples:

- readable terminal output for live demos
- generated Markdown/JSON artifacts for GitHub and Hugging Face sharing

The feature should make it easy to answer "what does this model do?" with concrete prediction examples rather than raw command output.

## Context

The repository already supports:

- local inference via `ditri-predict`
- FastAPI serving and async batch jobs
- Gradio demo
- benchmark and evaluation report generation

What is currently missing is a first-class showcase path that packages representative incident examples, runs them consistently, and emits presentation-ready results.

## Scope

This design covers:

- a curated showcase dataset stored in the repository
- a new CLI entrypoint dedicated to showcase generation
- terminal-friendly output for live presentation
- Markdown and JSON showcase reports written to `reports/`
- Makefile and README updates so the feature is discoverable
- automated tests for parsing and report generation logic

This design does not include:

- a major Gradio redesign
- remote deployment
- video or GIF generation
- changes to model training behavior

## User Experience

### Primary use case

The user runs one command and gets:

1. a concise terminal summary for live explanation
2. a detailed Markdown report that can be committed or pasted into docs
3. a machine-readable JSON file for reuse in future automation

### Example command

```bash
uv run ditri-demo-showcase \
  --model-path models/devops-incident-triage \
  --input-file data/demo/incidents_showcase.jsonl \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage \
  --output-json reports/demo_showcase.json \
  --output-markdown reports/demo_showcase.md
```

### Terminal behavior

The command should print:

- model path
- number of showcase examples
- number of exact label matches against `expected_label`
- number of items routed to human review
- a compact per-example line with title, predicted label, confidence, and review status

The terminal output should stay readable in a normal laptop terminal and avoid dumping the full score map unless explicitly added later.

## Data Design

### Showcase input file

Add `data/demo/incidents_showcase.jsonl`.

Each line should be a JSON object with this schema:

```json
{
  "id": "demo-001",
  "title": "EKS nodes not ready after CNI upgrade",
  "text": "EKS worker nodes became NotReady after a CNI plugin upgrade and new pods remain pending.",
  "expected_label": "k8s_cluster",
  "note": "Clear Kubernetes cluster state issue."
}
```

### Content rules

The file should include 6-8 examples total:

- clear examples covering several labels
- at least one ambiguous example intended to trigger `needs_human_review`
- short human-readable titles for presentation use

The dataset is for demonstration, not evaluation. It should be obviously curated and documented as such.

## Implementation Design

### New module

Add `src/devops_incident_triage/demo_showcase.py`.

Responsibilities:

- parse showcase input rows
- load the local model and tokenizer
- reuse existing prediction pipeline through `predict_batch`
- compute showcase summary statistics
- render terminal lines
- write JSON and Markdown outputs

### Reuse strategy

Do not duplicate inference logic.

The new module should import and reuse existing helpers from `src/devops_incident_triage/predict.py` where practical:

- `predict_batch`
- model loading pattern already used by prediction commands

If a tiny shared helper is needed, it may be extracted only if it reduces duplication without widening scope.

### CLI arguments

The new command should support:

- `--model-path`
- `--input-file`
- `--confidence-threshold`
- `--review-queue`
- `--output-json`
- `--output-markdown`
- `--max-length`

Defaults should target the repository's standard paths so `make demo-showcase` can run with minimal arguments.

### Generated JSON

`reports/demo_showcase.json` should contain:

- metadata
  - generated timestamp
  - model path
  - confidence threshold
  - review queue
- summary
  - total examples
  - matched expected labels
  - mismatched labels
  - human review count
- predictions
  - original showcase fields
  - prediction fields from `predict_batch`

### Generated Markdown

`reports/demo_showcase.md` should be presentation-ready and include:

- title
- short description of what the report shows
- generation timestamp and model path
- summary bullets
- main results table with:
  - title
  - expected label
  - predicted label
  - confidence
  - review required
  - recommended queue
- a small section for ambiguous/review-routed examples

The Markdown should be readable on both GitHub and Hugging Face without custom styling.

## Error Handling

The CLI should fail clearly when:

- the input file does not exist
- no valid showcase rows are found
- a showcase row is missing required fields like `title` or `text`
- the confidence threshold is invalid

Errors should be plain and actionable, matching the style of the existing CLI tools.

## Testing Strategy

Add tests in `tests/test_demo_showcase.py`.

Cover:

- input row loading and validation
- summary calculation
- Markdown rendering includes expected sections and table values
- JSON payload structure
- handling of review-routed examples

Tests should avoid heavyweight model inference by using fixed prediction rows or lightweight stubs around formatting functions.

## Documentation Changes

Update:

- `pyproject.toml` with a new script entry such as `ditri-demo-showcase`
- `Makefile` with a `demo-showcase` target
- `README.md`
- `README.ko.md`

README updates should show:

- why this command exists
- how to run it
- what files it generates
- how it helps with demos, portfolio presentation, and future GIF/video capture

## Non-Goals And Future Work

Not part of this task:

- embedding screenshots or GIFs directly in the repository
- syncing showcase outputs to Hugging Face automatically
- richer Gradio layouts

Possible follow-up:

- add example buttons to Gradio using the same curated showcase dataset
- commit a sample generated showcase report
- record a short demo GIF using the generated Markdown as a companion artifact
