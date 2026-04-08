import pytest

from devops_incident_triage.benchmark_models import (
    build_markdown_report,
    make_model_slug,
    parse_model_names,
)


def test_parse_model_names_deduplicates_and_strips() -> None:
    models = parse_model_names(
        " distilbert-base-uncased, xlm-roberta-base ,distilbert-base-uncased "
    )
    assert models == ["distilbert-base-uncased", "xlm-roberta-base"]


def test_parse_model_names_requires_non_empty() -> None:
    with pytest.raises(ValueError):
        parse_model_names(" ,  , ")


def test_make_model_slug_normalizes_symbols() -> None:
    assert make_model_slug("sentence-transformers/all-MiniLM-L6-v2") == (
        "sentence-transformers-all-minilm-l6-v2"
    )


def test_build_markdown_report_contains_table_rows() -> None:
    rows = [
        {
            "model_name": "distilbert-base-uncased",
            "status": "completed",
            "test_macro_f1": 0.88,
            "test_accuracy": 0.9,
            "weighted_f1": 0.89,
            "train_duration_seconds": 12.34,
            "eval_duration_seconds": 2.11,
        }
    ]
    markdown = build_markdown_report(
        rows=rows,
        generated_at="2026-04-08T00:00:00+00:00",
        best_model="distilbert-base-uncased",
    )
    assert "Model Benchmark Report" in markdown
    assert "Best model (test macro F1): `distilbert-base-uncased`" in markdown
    assert "| distilbert-base-uncased | completed | 0.8800 | 0.9000 | 0.8900 | 12.34 | 2.11 |" in (
        markdown
    )
