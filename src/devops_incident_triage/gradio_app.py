from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/devops-incident-triage"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))


def _load_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer, dict[int, str]]:
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    model.eval()
    return model, tokenizer, id2label


def _predict_text(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    id2label: dict[int, str],
) -> tuple[str, dict[str, float]]:
    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    score_map = {id2label[i]: float(probs[i]) for i in range(len(id2label))}
    best_label = max(score_map, key=score_map.get)
    return best_label, score_map


def main() -> None:
    model, tokenizer, id2label = _load_model()

    def infer(text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return {"predicted_label": "", "scores": {}}
        label, scores = _predict_text(text.strip(), model, tokenizer, id2label)
        return {"predicted_label": label, "scores": scores}

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(
            lines=8,
            label="Incident Text",
            placeholder="Paste an incident summary or error log",
        ),
        outputs=gr.JSON(label="Prediction"),
        title="DevOps Incident Triage Model",
        description="Portfolio demo for local inference (synthetic-started baseline).",
    )
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
