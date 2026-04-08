[![CI](https://github.com/dongkoony/DevOps-Incident-Triage-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/dongkoony/DevOps-Incident-Triage-Model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dongkoony/DevOps-Incident-Triage-Model)](https://github.com/dongkoony/DevOps-Incident-Triage-Model/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#라이선스)

# DevOps Incident Triage Model

[English](README.md) | 한국어

운영 장애 요약, 에러 로그, 배포 실패 메시지 같은 DevOps 인시던트 텍스트를 입력받아, 어떤 운영 영역이 먼저 검토해야 할지를 분류하는 NLP/MLOps 프로젝트입니다.

이 저장소는 단순한 모델 데모보다 아래를 더 중요하게 다룹니다.

- 재현 가능한 학습 및 평가 파이프라인
- 로컬 추론, FastAPI 서빙, 비동기 배치 잡
- 요청 추적과 Prometheus 스타일 지표 노출
- Docker, CI, 릴리즈 절차, Hugging Face 배포
- 데이터 한계와 운영 범위를 숨기지 않는 문서화

## 프로젝트 개요

모델은 인시던트 문장을 읽고, 어떤 담당 영역으로 초동 라우팅해야 할지 예측합니다.

현재 라벨 셋:

| Label | 설명 |
|---|---|
| `k8s_cluster` | Kubernetes 스케줄링, 노드, 클러스터 상태 이슈 |
| `cicd_pipeline` | CI/CD 빌드, 테스트, 배포 파이프라인 실패 |
| `aws_iam_network` | AWS IAM, VPC, 네트워크, 권한 관련 이슈 |
| `deployment_release` | Helm, rollout, release 운영 이슈 |
| `container_runtime` | Docker, containerd, 이미지, 런타임 이슈 |
| `observability_alerting` | 모니터링, 로깅, 트레이싱, 알림 이슈 |
| `database_state` | 데이터베이스 연결, 복제, 락, 스토리지 상태 이슈 |

## 저장소 범위

이 저장소는 학습된 모델 파일만 담은 레포지토리가 아닙니다.

- `transformers` 기반 학습 파이프라인
- confusion matrix 및 threshold 분석을 포함한 평가 파이프라인
- 단건/배치 CLI 추론
- FastAPI 실시간/배치 추론 API
- 큐형 워크플로우를 위한 비동기 배치 잡 API
- 여러 백본 비교용 벤치마크 자동화
- 포트폴리오 수준의 릴리즈/문서화/MLOps 흐름

## 데이터 정직성

현재 스타터 데이터셋인 `data/sample/incidents_synthetic.csv`는 synthetic 데이터입니다.

- 실제 운영 환경에서 수집한 티켓이나 로그가 아닙니다.
- 현재 공개 점수는 실제 일반화 성능을 보장하지 않습니다.
- 실무 적용 전에는 비식별화된 실제 인시던트 데이터로 재학습과 재평가가 필요합니다.

이 한계는 의도적으로 README와 평가 문서에서 명시하고 있습니다.

## 모델 및 실험

기본 베이스라인:

- `distilbert-base-uncased`

이 베이스라인을 쓰는 이유:

- DevOps 로그와 에러 메시지는 영문 비중이 높은 경우가 많음
- 개인 개발 환경에서도 학습/반복 실험 비용이 현실적임
- 추후 `xlm-roberta-base` 같은 다국어 백본으로 동일 파이프라인 확장이 쉬움

벤치마크 자동화:

```bash
uv run ditri-benchmark \
  --data-dir data/processed \
  --models distilbert-base-uncased,sentence-transformers/all-MiniLM-L6-v2,xlm-roberta-base \
  --epochs 4 \
  --skip-existing
```

산출물:

- `reports/model_benchmark.json`
- `reports/model_benchmark.md`
- `models/benchmarks/<model-slug>/`
- `reports/benchmarks/<model-slug>/`

## 빠른 시작

### 1. 환경 준비

```bash
uv python install 3.12
uv sync --extra dev --extra api --extra viz --extra peft --extra gradio
```

### 2. 데이터 준비

synthetic 데이터 기준:

```bash
uv run ditri-data-prep \
  --input-path data/sample/incidents_synthetic.csv \
  --output-dir data/processed \
  --seed 42
```

실데이터 기준:

```bash
uv run ditri-ingest-raw \
  --input-path data/raw/incidents_template.csv \
  --output-canonical-path data/raw/incidents_canonical.csv \
  --output-training-path data/raw/incidents_training_ready.csv \
  --report-path reports/raw_ingestion_report.json

uv run ditri-data-prep \
  --input-path data/raw/incidents_training_ready.csv \
  --output-dir data/processed \
  --seed 42
```

### 3. 학습

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --epochs 4
```

선택적 PEFT:

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --use-peft
```

### 4. 평가

```bash
uv run ditri-eval \
  --model-path models/devops-incident-triage \
  --data-dir data/processed \
  --report-dir reports \
  --confidence-thresholds 0.4,0.5,0.6,0.7
```

핵심 산출물:

- `reports/evaluation_metrics.json`
- `reports/per_label_metrics.json`
- `reports/threshold_metrics.json`
- `reports/confusion_matrix.csv`
- `reports/figures/confusion_matrix.png`
- `reports/sample_predictions.jsonl`

## 추론 및 서빙

### CLI 추론

단건 추론:

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --confidence-threshold 0.6 \
  --review-queue sre_manual_triage \
  --text "EKS worker nodes became NotReady after CNI upgrade."
```

배치 추론:

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --input-file data/sample/incidents_synthetic.csv \
  --text-column text \
  --output-file reports/batch_predictions.jsonl
```

### FastAPI

```bash
CONFIDENCE_THRESHOLD=0.6 REVIEW_QUEUE=sre_manual_triage BATCH_MAX_ITEMS=32 uv run ditri-api
```

지원 엔드포인트:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `POST /predict/batch/async`
- `GET /predict/batch/async/{job_id}`
- `GET /metrics`

운영 기능:

- `X-Request-ID` 응답 헤더를 통한 요청 추적
- confidence threshold 기반 human review 전환
- 큐형 워크플로우를 위한 비동기 배치 잡
- Prometheus 호환 지표 노출

## 전달 및 릴리즈

이 저장소는 GitFlow-lite 스타일로 운영합니다.

- `main`: 릴리즈 기준 브랜치
- `develop`: 통합 개발 브랜치
- `feature/*`: 기능 단위 브랜치
- `release/*`: 릴리즈 안정화 브랜치

현재 프로젝트 릴리즈:

- `v0.3.0`

관련 문서:

- [브랜치 전략](docs/branch_strategy.md)
- [릴리즈 체크리스트](docs/release_checklist.md)
- [아키텍처](docs/architecture.md)
- [모델 벤치마킹 가이드](docs/model_benchmarking.md)
- [포트폴리오 노트](docs/portfolio_notes.md)

## Hugging Face 배포

```bash
export HF_TOKEN="hf_xxx"

uv run ditri-publish \
  --model-dir models/devops-incident-triage \
  --repo-id <your-hf-username>/devops-incident-triage
```

필요한 경우 `docs/model_card.md`를 모델 아티팩트 디렉터리의 `README.md`로 복사해 모델 카드로 함께 업로드합니다.

## 저장소 구조

```text
.
├─ src/devops_incident_triage/
│  ├─ data_prep.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ benchmark_models.py
│  ├─ predict.py
│  ├─ api.py
│  ├─ hf_publish.py
│  └─ ingest_raw.py
├─ tests/
├─ data/
├─ reports/
├─ models/
├─ docs/
├─ .github/workflows/
├─ Dockerfile
├─ Makefile
└─ pyproject.toml
```

## 한계

- 현재 공개 기준 학습 데이터는 synthetic 데이터 중심입니다.
- 문제 정의는 single-label 분류라서 복합 원인 인시던트를 충분히 반영하지 못합니다.
- 장문의 multi-line 로그나 매우 noisy한 컨텍스트는 추가 검증이 필요합니다.
- 이 모델은 자동 조치용이 아니라 triage 보조용입니다.

## 라이선스

MIT
