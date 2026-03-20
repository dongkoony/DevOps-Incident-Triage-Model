from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from devops_incident_triage.labels import INCIDENT_LABELS

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


@dataclass(slots=True)
class DataPrepConfig:
    input_path: Path = SAMPLE_DATA_DIR / "incidents_synthetic.csv"
    output_dir: Path = PROCESSED_DATA_DIR
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    text_column: str = "text"
    label_column: str = "label"


@dataclass(slots=True)
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    data_dir: Path = PROCESSED_DATA_DIR
    output_dir: Path = MODELS_DIR / "devops-incident-triage"
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 4
    seed: int = 42
    labels: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = INCIDENT_LABELS.copy()
