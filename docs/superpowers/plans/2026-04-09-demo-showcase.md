# Demo Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated demo showcase command that prints presentation-friendly terminal output and generates Markdown/JSON artifacts from the same incident examples.

**Architecture:** Introduce a small `demo_showcase.py` module that reuses the existing prediction pipeline, reads curated examples from `data/demo/incidents_showcase.jsonl`, computes summary stats, and renders both human-readable and machine-readable outputs. Keep the feature lightweight by unit-testing parsing and rendering helpers separately from model inference, then wire the CLI, Make target, and README documentation.

**Tech Stack:** Python 3.12, `transformers`, existing local inference helpers, `pytest`, `ruff`, `uv`

---

## File Structure

- Create: `data/demo/incidents_showcase.jsonl`
  - Curated presentation examples with `id`, `title`, `text`, `expected_label`, and optional `note`
- Create: `src/devops_incident_triage/demo_showcase.py`
  - Showcase row loading, summary calculation, Markdown/JSON rendering, terminal printing, and CLI entrypoint
- Create: `tests/test_demo_showcase.py`
  - Unit tests for input validation, summary stats, JSON payload, and Markdown rendering
- Modify: `pyproject.toml`
  - Register `ditri-demo-showcase`
- Modify: `Makefile`
  - Add `demo-showcase` target
- Modify: `README.md`
  - Add English demo/showcase usage section
- Modify: `README.ko.md`
  - Add Korean demo/showcase usage section

### Task 1: Add curated showcase data and parsing helpers

**Files:**
- Create: `data/demo/incidents_showcase.jsonl`
- Create: `tests/test_demo_showcase.py`
- Create: `src/devops_incident_triage/demo_showcase.py`

- [ ] **Step 1: Write the failing parsing and summary tests**

```python
import json
from pathlib import Path

import pytest

from devops_incident_triage.demo_showcase import (
    load_showcase_rows,
    summarize_predictions,
)


def test_load_showcase_rows_reads_curated_examples(tmp_path: Path) -> None:
    input_file = tmp_path / "incidents_showcase.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "demo-001",
                        "title": "EKS nodes not ready",
                        "text": "EKS worker nodes became NotReady after a CNI upgrade.",
                        "expected_label": "k8s_cluster",
                        "note": "Clear Kubernetes case.",
                    }
                ),
                json.dumps(
                    {
                        "id": "demo-002",
                        "title": "IAM role denied during deploy",
                        "text": "GitHub Actions deploy failed because sts:AssumeRole was denied.",
                        "expected_label": "aws_iam_network",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_showcase_rows(input_file)

    assert [row["id"] for row in rows] == ["demo-001", "demo-002"]
    assert rows[0]["note"] == "Clear Kubernetes case."
    assert rows[1]["note"] == ""


def test_load_showcase_rows_requires_title_and_text(tmp_path: Path) -> None:
    input_file = tmp_path / "invalid_showcase.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "demo-003",
                "title": "",
                "text": "Deployment failed with a timeout.",
                "expected_label": "deployment_release",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_showcase_rows(input_file)


def test_summarize_predictions_counts_matches_and_reviews() -> None:
    predictions = [
        {
            "title": "Kubernetes outage",
            "expected_label": "k8s_cluster",
            "predicted_label": "k8s_cluster",
            "needs_human_review": False,
        },
        {
            "title": "Ambiguous release issue",
            "expected_label": "deployment_release",
            "predicted_label": "aws_iam_network",
            "needs_human_review": True,
        },
    ]

    summary = summarize_predictions(predictions)

    assert summary == {
        "total_examples": 2,
        "matched_expected_labels": 1,
        "mismatched_labels": 1,
        "human_review_count": 1,
    }
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py -k "load_showcase_rows or summarize_predictions"
```

Expected:

- FAIL because `devops_incident_triage.demo_showcase` does not exist yet

- [ ] **Step 3: Add curated showcase data file**

Create `data/demo/incidents_showcase.jsonl` with seven examples:

```jsonl
{"id":"demo-001","title":"EKS nodes not ready after CNI upgrade","text":"EKS worker nodes became NotReady after a CNI plugin upgrade and new pods remain pending.","expected_label":"k8s_cluster","note":"Clear Kubernetes cluster state issue."}
{"id":"demo-002","title":"GitHub Actions deploy blocked by IAM","text":"GitHub Actions deployment failed because the runner could not assume the production IAM role.","expected_label":"aws_iam_network","note":"Permission and AWS identity failure."}
{"id":"demo-003","title":"Helm rollout timeout during release","text":"Helm upgrade timed out while waiting for the rollout to finish and the release stayed pending.","expected_label":"deployment_release","note":"Release orchestration issue."}
{"id":"demo-004","title":"Container image pull back-off","text":"Pods are stuck in ImagePullBackOff because the registry token expired on the worker nodes.","expected_label":"container_runtime","note":"Container runtime and image retrieval issue."}
{"id":"demo-005","title":"Prometheus alerts missing after config reload","text":"Prometheus reloaded successfully but several alerting rules stopped firing and Grafana panels show no recent metrics.","expected_label":"observability_alerting","note":"Monitoring and alerting visibility issue."}
{"id":"demo-006","title":"Primary database locked during migration","text":"The primary PostgreSQL instance became locked during a schema migration and application writes started timing out.","expected_label":"database_state","note":"Database lock and write path issue."}
{"id":"demo-007","title":"Ambiguous deploy failure with timeout and permission denied","text":"The release failed with both timeout errors and permission denied messages, and it is unclear whether the root cause is IAM or the rollout itself.","expected_label":"deployment_release","note":"Intentionally ambiguous example for human review."}
```

- [ ] **Step 4: Implement the parsing and summary helpers**

Add the initial version of `src/devops_incident_triage/demo_showcase.py`:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devops_incident_triage.predict import predict_batch

DEFAULT_MODEL_PATH = Path("models/devops-incident-triage")
DEFAULT_INPUT_FILE = Path("data/demo/incidents_showcase.jsonl")
DEFAULT_OUTPUT_JSON = Path("reports/demo_showcase.json")
DEFAULT_OUTPUT_MARKDOWN = Path("reports/demo_showcase.md")


def load_showcase_rows(input_file: Path) -> list[dict[str, str]]:
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found")

    rows: list[dict[str, str]] = []
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = {
                "id": str(raw.get("id", "")).strip(),
                "title": str(raw.get("title", "")).strip(),
                "text": str(raw.get("text", "")).strip(),
                "expected_label": str(raw.get("expected_label", "")).strip(),
                "note": str(raw.get("note", "")).strip(),
            }
            if not row["title"] or not row["text"]:
                raise ValueError(
                    f"Showcase row {line_number} missing required fields: title/text"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid showcase rows found in {input_file}")
    return rows


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, int]:
    matched = sum(
        1 for row in predictions if row["expected_label"] == row["predicted_label"]
    )
    review_count = sum(1 for row in predictions if row["needs_human_review"])
    total = len(predictions)
    return {
        "total_examples": total,
        "matched_expected_labels": matched,
        "mismatched_labels": total - matched,
        "human_review_count": review_count,
    }
```

- [ ] **Step 5: Run the focused tests again to verify they pass**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py -k "load_showcase_rows or summarize_predictions"
```

Expected:

- PASS for the three new tests

- [ ] **Step 6: Commit the parsing foundation**

```bash
git add data/demo/incidents_showcase.jsonl tests/test_demo_showcase.py src/devops_incident_triage/demo_showcase.py
git commit -m "feat(demo): add showcase dataset and parsing helpers"
```

### Task 2: Add report payload, Markdown rendering, and terminal summary output

**Files:**
- Modify: `tests/test_demo_showcase.py`
- Modify: `src/devops_incident_triage/demo_showcase.py`

- [ ] **Step 1: Write failing report generation tests**

Append these tests to `tests/test_demo_showcase.py`:

```python
from devops_incident_triage.demo_showcase import (
    build_markdown_report,
    build_report_payload,
    build_terminal_summary,
)


def test_build_report_payload_includes_metadata_and_summary() -> None:
    rows = [
        {
            "id": "demo-001",
            "title": "EKS nodes not ready",
            "text": "EKS worker nodes became NotReady after a CNI upgrade.",
            "expected_label": "k8s_cluster",
            "note": "Clear Kubernetes case.",
            "predicted_label": "k8s_cluster",
            "final_label": "k8s_cluster",
            "confidence": 0.91,
            "confidence_threshold": 0.6,
            "needs_human_review": False,
            "recommended_queue": "k8s_cluster",
            "scores": {"k8s_cluster": 0.91},
        }
    ]

    payload = build_report_payload(
        predictions=rows,
        model_path="models/devops-incident-triage",
        confidence_threshold=0.6,
        review_queue="sre_manual_triage",
        generated_at="2026-04-09T00:00:00+00:00",
    )

    assert payload["metadata"]["model_path"] == "models/devops-incident-triage"
    assert payload["summary"]["matched_expected_labels"] == 1
    assert payload["predictions"][0]["title"] == "EKS nodes not ready"


def test_build_markdown_report_lists_review_examples() -> None:
    markdown = build_markdown_report(
        predictions=[
            {
                "title": "Ambiguous deploy failure",
                "expected_label": "deployment_release",
                "predicted_label": "aws_iam_network",
                "confidence": 0.44,
                "needs_human_review": True,
                "recommended_queue": "sre_manual_triage",
                "note": "Needs manual triage.",
            }
        ],
        summary={
            "total_examples": 1,
            "matched_expected_labels": 0,
            "mismatched_labels": 1,
            "human_review_count": 1,
        },
        generated_at="2026-04-09T00:00:00+00:00",
        model_path="models/devops-incident-triage",
        confidence_threshold=0.6,
    )

    assert "# Demo Showcase Report" in markdown
    assert "| Ambiguous deploy failure | deployment_release | aws_iam_network | 0.4400 | yes | sre_manual_triage |" in markdown
    assert "## Review-Required Examples" in markdown
    assert "- **Ambiguous deploy failure**: Needs manual triage." in markdown


def test_build_terminal_summary_is_human_readable() -> None:
    output = build_terminal_summary(
        predictions=[
            {
                "title": "EKS nodes not ready",
                "predicted_label": "k8s_cluster",
                "confidence": 0.9123,
                "needs_human_review": False,
                "recommended_queue": "k8s_cluster",
            }
        ],
        summary={
            "total_examples": 1,
            "matched_expected_labels": 1,
            "mismatched_labels": 0,
            "human_review_count": 0,
        },
        model_path="models/devops-incident-triage",
    )

    assert "Demo showcase for models/devops-incident-triage" in output
    assert "Matches: 1/1" in output
    assert "- EKS nodes not ready -> k8s_cluster (0.9123, auto, queue=k8s_cluster)" in output
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py -k "build_report_payload or build_markdown_report or build_terminal_summary"
```

Expected:

- FAIL because the rendering helpers do not exist yet

- [ ] **Step 3: Implement report payload and rendering helpers**

Extend `src/devops_incident_triage/demo_showcase.py` with:

```python
def build_report_payload(
    predictions: list[dict[str, Any]],
    model_path: str,
    confidence_threshold: float,
    review_queue: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": generated_at,
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "review_queue": review_queue,
        },
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
    }


def build_markdown_report(
    predictions: list[dict[str, Any]],
    summary: dict[str, int],
    generated_at: str,
    model_path: str,
    confidence_threshold: float,
) -> str:
    lines = [
        "# Demo Showcase Report",
        "",
        "Curated prediction examples for portfolio demos and repository sharing.",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Model path: `{model_path}`",
        f"- Confidence threshold: `{confidence_threshold}`",
        "",
        "## Summary",
        "",
        f"- Total examples: `{summary['total_examples']}`",
        f"- Expected label matches: `{summary['matched_expected_labels']}`",
        f"- Mismatches: `{summary['mismatched_labels']}`",
        f"- Human review count: `{summary['human_review_count']}`",
        "",
        "## Predictions",
        "",
        "| Title | Expected | Predicted | Confidence | Review Required | Queue |",
        "|---|---|---|---:|---|---|",
    ]
    for row in predictions:
        review_required = "yes" if row["needs_human_review"] else "no"
        lines.append(
            f"| {row['title']} | {row['expected_label']} | {row['predicted_label']} | "
            f"{row['confidence']:.4f} | {review_required} | {row['recommended_queue']} |"
        )

    review_rows = [row for row in predictions if row["needs_human_review"]]
    if review_rows:
        lines.extend(["", "## Review-Required Examples", ""])
        for row in review_rows:
            note = row.get("note", "").strip() or "No note provided."
            lines.append(f"- **{row['title']}**: {note}")
    return "\n".join(lines) + "\n"


def build_terminal_summary(
    predictions: list[dict[str, Any]],
    summary: dict[str, int],
    model_path: str,
) -> str:
    lines = [
        f"Demo showcase for {model_path}",
        f"Examples: {summary['total_examples']}",
        f"Matches: {summary['matched_expected_labels']}/{summary['total_examples']}",
        f"Human review: {summary['human_review_count']}",
        "",
    ]
    for row in predictions:
        route = "review" if row["needs_human_review"] else "auto"
        lines.append(
            f"- {row['title']} -> {row['predicted_label']} "
            f"({row['confidence']:.4f}, {route}, queue={row['recommended_queue']})"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Add file writing helpers and a prediction merge helper**

Continue in `src/devops_incident_triage/demo_showcase.py`:

```python
def merge_rows_with_predictions(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        merged.append({**row, **prediction})
    return merged


def write_json(payload: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(markdown: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")
```

- [ ] **Step 5: Run the focused tests again to verify they pass**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py -k "build_report_payload or build_markdown_report or build_terminal_summary"
```

Expected:

- PASS for the rendering tests

- [ ] **Step 6: Commit the report generation layer**

```bash
git add tests/test_demo_showcase.py src/devops_incident_triage/demo_showcase.py
git commit -m "feat(demo): add showcase report generation"
```

### Task 3: Wire the CLI entrypoint and repository commands

**Files:**
- Modify: `tests/test_demo_showcase.py`
- Modify: `src/devops_incident_triage/demo_showcase.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Write a failing CLI workflow test**

Add this test to `tests/test_demo_showcase.py`:

```python
from pathlib import Path

from devops_incident_triage import demo_showcase


def test_run_showcase_generates_json_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_file = tmp_path / "showcase.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "demo-001",
                "title": "EKS nodes not ready",
                "text": "EKS worker nodes became NotReady after a CNI upgrade.",
                "expected_label": "k8s_cluster",
                "note": "Clear Kubernetes case.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyModel:
        config = type("Config", (), {"id2label": {0: "k8s_cluster"}})()

    class DummyTokenizer:
        pass

    monkeypatch.setattr(
        demo_showcase,
        "load_model_bundle",
        lambda model_path: (DummyModel(), DummyTokenizer()),
    )
    monkeypatch.setattr(
        demo_showcase,
        "predict_batch",
        lambda texts, model, tokenizer, max_length, confidence_threshold, review_queue: [
            {
                "text": texts[0],
                "predicted_label": "k8s_cluster",
                "final_label": "k8s_cluster",
                "confidence": 0.91,
                "confidence_threshold": confidence_threshold,
                "needs_human_review": False,
                "recommended_queue": "k8s_cluster",
                "scores": {"k8s_cluster": 0.91},
            }
        ],
    )

    json_path = tmp_path / "demo_showcase.json"
    markdown_path = tmp_path / "demo_showcase.md"

    terminal_output = demo_showcase.run_showcase(
        model_path=Path("models/devops-incident-triage"),
        input_file=input_file,
        confidence_threshold=0.6,
        review_queue="sre_manual_triage",
        output_json=json_path,
        output_markdown=markdown_path,
        max_length=256,
    )

    assert "Matches: 1/1" in terminal_output
    assert json_path.exists()
    assert markdown_path.exists()
    assert "\"matched_expected_labels\": 1" in json_path.read_text(encoding="utf-8")
    assert "# Demo Showcase Report" in markdown_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py::test_run_showcase_generates_json_and_markdown
```

Expected:

- FAIL because `run_showcase` and `load_model_bundle` do not exist yet

- [ ] **Step 3: Implement the runnable showcase workflow and CLI parser**

Finish `src/devops_incident_triage/demo_showcase.py`:

```python
def load_model_bundle(
    model_path: Path,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


def run_showcase(
    model_path: Path,
    input_file: Path,
    confidence_threshold: float,
    review_queue: str,
    output_json: Path,
    output_markdown: Path,
    max_length: int,
) -> str:
    rows = load_showcase_rows(input_file)
    model, tokenizer = load_model_bundle(model_path)
    predictions = predict_batch(
        [row["text"] for row in rows],
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        confidence_threshold=confidence_threshold,
        review_queue=review_queue,
    )
    merged = merge_rows_with_predictions(rows, predictions)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = build_report_payload(
        predictions=merged,
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
        review_queue=review_queue,
        generated_at=generated_at,
    )
    markdown = build_markdown_report(
        predictions=merged,
        summary=payload["summary"],
        generated_at=generated_at,
        model_path=str(model_path),
        confidence_threshold=confidence_threshold,
    )
    terminal_output = build_terminal_summary(
        predictions=merged,
        summary=payload["summary"],
        model_path=str(model_path),
    )
    write_json(payload, output_json)
    write_markdown(markdown, output_markdown)
    return terminal_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a curated demo showcase for DevOps incident triage."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--review-queue", type=str, default="sre_manual_triage")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    terminal_output = run_showcase(
        model_path=args.model_path,
        input_file=args.input_file,
        confidence_threshold=args.confidence_threshold,
        review_queue=args.review_queue,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        max_length=args.max_length,
    )
    print(terminal_output)
    print(f"JSON report saved to {args.output_json}")
    print(f"Markdown report saved to {args.output_markdown}")
```

- [ ] **Step 4: Register the command and Make target**

Modify `pyproject.toml`:

```toml
[project.scripts]
ditri-data-prep = "devops_incident_triage.data_prep:main"
ditri-train = "devops_incident_triage.train:main"
ditri-eval = "devops_incident_triage.evaluate:main"
ditri-predict = "devops_incident_triage.predict:main"
ditri-publish = "devops_incident_triage.hf_publish:main"
ditri-api = "devops_incident_triage.api:main"
ditri-gradio = "devops_incident_triage.gradio_app:main"
ditri-ingest-raw = "devops_incident_triage.ingest_raw:main"
ditri-benchmark = "devops_incident_triage.benchmark_models:main"
ditri-demo-showcase = "devops_incident_triage.demo_showcase:main"
```

Modify `Makefile`:

```make
.PHONY: help install ingest-raw prep-data train eval benchmark predict demo-showcase api gradio docker-build docker-run test lint

help:
	@echo "  demo-showcase Run curated demo showcase and generate Markdown/JSON reports"

demo-showcase:
	uv run ditri-demo-showcase --model-path $(MODEL_DIR)
```

- [ ] **Step 5: Run the focused test again to verify it passes**

Run:

```bash
uv run pytest -q tests/test_demo_showcase.py::test_run_showcase_generates_json_and_markdown
```

Expected:

- PASS and both output files created in the temporary directory

- [ ] **Step 6: Commit the CLI wiring**

```bash
git add tests/test_demo_showcase.py src/devops_incident_triage/demo_showcase.py pyproject.toml Makefile
git commit -m "feat(demo): add showcase CLI command"
```

### Task 4: Document the demo workflow and verify the full feature

**Files:**
- Modify: `README.md`
- Modify: `README.ko.md`

- [ ] **Step 1: Add README showcase sections**

Insert this English section in `README.md` after the CLI prediction examples:

```md
### Demo Showcase Report

For live demos and portfolio sharing, generate a curated showcase report from representative incident examples:

```bash
uv run ditri-demo-showcase \
  --model-path models/devops-incident-triage \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage
```

Generated artifacts:

- `reports/demo_showcase.json`
- `reports/demo_showcase.md`

This workflow is useful when preparing terminal demos, README evidence, or GIF/video walkthroughs because the terminal summary and saved report come from the same curated examples.
```

Insert this Korean section in `README.ko.md` at the matching location:

```md
### 데모 쇼케이스 리포트

라이브 데모나 포트폴리오 공유용으로 대표 인시던트 예시를 한 번에 실행하고, 터미널 요약과 리포트 파일을 함께 만들 수 있습니다.

```bash
uv run ditri-demo-showcase \
  --model-path models/devops-incident-triage \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage
```

생성 산출물:

- `reports/demo_showcase.json`
- `reports/demo_showcase.md`

이 흐름을 사용하면 발표용 터미널 데모, README 근거 자료, 이후 GIF/영상 촬영까지 같은 예시 세트를 기준으로 일관되게 준비할 수 있습니다.
```

- [ ] **Step 2: Run lint and the full test suite**

Run:

```bash
uv run ruff check .
uv run pytest -q
```

Expected:

- `ruff`: all checks passed
- `pytest`: full suite passes with the new showcase tests included

- [ ] **Step 3: Run the new showcase command manually**

Run:

```bash
uv run ditri-demo-showcase \
  --model-path models/devops-incident-triage-smoke \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage
```

Expected:

- terminal summary prints curated examples
- `reports/demo_showcase.json` is written
- `reports/demo_showcase.md` is written

- [ ] **Step 4: Commit docs and verification-complete changes**

```bash
git add README.md README.ko.md reports/demo_showcase.json reports/demo_showcase.md
git commit -m "docs: add demo showcase usage"
```

- [ ] **Step 5: Final status check**

Run:

```bash
git status --short
git log --oneline -3
```

Expected:

- clean working tree
- the three task commits appear in order
