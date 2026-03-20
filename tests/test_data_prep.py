from pathlib import Path

from devops_incident_triage.data_prep import (
    generate_synthetic_examples,
    prepare_dataset,
    write_examples_to_csv,
)


def test_generate_synthetic_examples_count() -> None:
    rows = generate_synthetic_examples(samples_per_label=3, seed=7)
    assert len(rows) == 21
    assert all("text" in row and "label" in row for row in rows)


def test_prepare_dataset_outputs(tmp_path: Path) -> None:
    sample_csv = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "processed"
    rows = generate_synthetic_examples(samples_per_label=10, seed=42)
    write_examples_to_csv(rows, sample_csv)

    metadata = prepare_dataset(
        input_path=sample_csv,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        text_column="text",
        label_column="label",
    )

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "label_mapping.json").exists()
    assert metadata["splits"]["total"] == 70
