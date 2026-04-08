---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- devops
- sre
- incident-triage
- text-classification
- mlops
- fastapi
- transformers
- python
base_model: distilbert-base-uncased
---

# devops-incident-triage

`devops-incident-triage` is a multiclass text classification model for routing DevOps incident summaries and error messages to the most likely operational domain.

In short: give it an incident sentence such as a deployment failure, Kubernetes cluster issue, IAM/network error, or database state problem, and it predicts which team/domain should review it first.

## Model Summary

- Task: DevOps incident text classification
- Problem type: multiclass classification
- Base model: `distilbert-base-uncased`
- Project release: `v0.3.0`
- Intended role: first-pass triage support, not autonomous decision-making

## Labels

| Label | Meaning |
|---|---|
| `k8s_cluster` | Kubernetes scheduling, node, or cluster-state issues |
| `cicd_pipeline` | CI/CD build, test, or deployment pipeline failures |
| `aws_iam_network` | AWS IAM, VPC, network, or permission-related issues |
| `deployment_release` | Helm, rollout, release, or deployment operation issues |
| `container_runtime` | Docker, containerd, image, or container runtime issues |
| `observability_alerting` | Monitoring, logging, tracing, or alerting issues |
| `database_state` | Database connectivity, replication, lock, or storage-state issues |

## Intended Use

This model is designed for:

- incident triage assistance in DevOps, Platform, and SRE workflows
- ticket auto-tagging support
- queue recommendation support before a human reviews the issue

This model is not designed for:

- fully autonomous production actions
- incident severity decisions without human review
- root-cause analysis by itself

## Important Scope Note

The published model performs classification only.

Operational behaviors such as:

- confidence threshold gating
- `needs_human_review` fallback
- synchronous batch inference
- asynchronous batch jobs
- API observability and metrics

are implemented in the service layer of the project, not inside the model weights themselves.

Project repository:

- GitHub: `dongkoony/DevOps-Incident-Triage-Model`

## Training Data

This version was trained on a synthetic starter dataset derived from DevOps-style incident examples.

- Source file in project: `data/sample/incidents_synthetic.csv`
- The dataset is not collected from a real production environment.
- The reported behavior should be interpreted as portfolio and pipeline evidence, not as validated real-world generalization.

If this model is to be used beyond demonstration or experimentation, it should be retrained and reevaluated on anonymized real incident data.

## Training Procedure

- Data split: train / validation / test
- Max input length: 256
- Baseline checkpoint: `distilbert-base-uncased`
- Evaluation metrics: accuracy, macro F1, weighted F1, per-label precision/recall/F1

The project also includes a benchmark workflow to compare multiple backbones under the same setup:

- `distilbert-base-uncased`
- `sentence-transformers/all-MiniLM-L6-v2`
- `xlm-roberta-base`

## How To Use

### Transformers pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="dongkoony/devops-incident-triage",
    tokenizer="dongkoony/devops-incident-triage",
)

result = classifier(
    "GitHub Actions deployment failed because IAM role assumption was denied."
)
print(result)
```

### With `AutoTokenizer` and `AutoModelForSequenceClassification`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "dongkoony/devops-incident-triage"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "EKS worker nodes became NotReady after CNI upgrade."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = int(logits.argmax(dim=-1))
print(model.config.id2label[predicted_id])
```

## Evaluation Artifacts

The project evaluation pipeline produces:

- `evaluation_metrics.json`
- `per_label_metrics.json`
- `threshold_metrics.json`
- `confusion_matrix.csv`
- `sample_predictions.jsonl`

These artifacts are generated in the project repository and are intended to make the evaluation process reproducible and inspectable.

## Limitations

- trained on synthetic incident text rather than real anonymized production tickets/logs
- single-label formulation, while real incidents may have multiple contributing domains
- long, noisy, or multi-line logs may require additional preprocessing
- classification confidence should not be treated as an operational decision guarantee

## Ethical and Operational Considerations

- keep a human in the loop for low-confidence or high-impact decisions
- do not use the model as the sole authority for remediation actions
- ensure sensitive log data is anonymized before retraining or evaluation
- review failure cases regularly to avoid silently reinforcing routing bias

## Recommended Next Steps

1. Retrain on anonymized real incident data.
2. Add multilabel classification experiments.
3. Improve labeling guidelines and label quality review.
4. Connect offline evaluation with online drift monitoring.

## Citation

If you reference the project, please cite the GitHub repository and the released model version together so the implementation context and operational assumptions remain clear.
