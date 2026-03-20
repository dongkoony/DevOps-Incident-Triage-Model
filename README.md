# DevOps-Incident-Triage-Model

실무형 DevOps/MLOps 포트폴리오를 위한 Hugging Face NLP 프로젝트입니다.  
운영 인시던트 텍스트(에러 로그, 장애 요약, 파이프라인 실패 메시지)를 입력받아 **초동 트리아지 카테고리**를 분류합니다.

이 저장소는 "화려한 데모"보다 아래를 우선합니다.

- 재현 가능한 학습/평가 파이프라인
- 로컬 추론 및 API 서빙
- Docker/CI/Hugging Face 배포 경로
- 데이터 현실성에 대한 정직한 문서화

## 1) Problem & Scope

### 목표
운영 인시던트 문장을 아래 도메인으로 분류해, 온콜/플랫폼/서비스팀의 초동 대응 라우팅을 돕습니다.

### 라벨 셋 (v1)

| Label | 설명 |
|---|---|
| `k8s_cluster` | Kubernetes 스케줄링/노드/클러스터 상태 이슈 |
| `cicd_pipeline` | CI/CD 파이프라인 빌드/테스트/배포 흐름 실패 |
| `aws_iam_network` | AWS IAM/VPC/네트워크/권한 문제 |
| `deployment_release` | Helm/롤아웃/릴리즈 운영 이슈 |
| `container_runtime` | Docker/containerd 런타임/이미지/리소스 문제 |
| `observability_alerting` | 모니터링/로그/트레이싱/알림 품질 이슈 |
| `database_state` | DB 연결/복제/락/스토리지 상태 이슈 |

## 2) 왜 Multiclass 먼저 시작했는가

실제 인시던트는 복합 원인을 가질 수 있어 멀티라벨이 이상적일 때가 많습니다.  
하지만 포트폴리오 v1에서는 **단일 주 담당 도메인 분류(multiclass)**가 더 현실적입니다.

- 라벨링 비용이 낮고 합의가 쉬움
- 베이스라인 평가가 명확함(accuracy, macro F1, confusion matrix)
- API/운영 라우팅에 바로 연결하기 쉬움

멀티라벨 확장은 `docs/portfolio_notes.md`에 후속 단계로 정리했습니다.

## 3) 모델 선택과 다국어 전략

기본 체크포인트는 `distilbert-base-uncased`입니다.

- DevOps 로그/오류 텍스트는 실제로 영문 비중이 높아 English-first 베이스라인이 효율적
- 개인 환경에서 학습/실험 비용이 낮고 반복 실험이 빠름
- 추후 다국어 요구가 있으면 `xlm-roberta-base`로 교체해 동일 파이프라인 재사용 가능

## 4) Data Honesty

`data/sample/incidents_synthetic.csv`는 **합성(synthetic) 스타터 데이터**입니다.

- 실제 운영 데이터가 아닙니다.
- 이 데이터 기반 점수는 "실서비스 일반화 성능"을 의미하지 않습니다.
- 실데이터(비식별화된 티켓/로그)로 교체해 재학습하는 것이 핵심입니다.

## 5) Quickstart (uv + Python 3.12)

```bash
uv python install 3.12
uv sync --extra dev --extra api --extra viz --extra peft --extra gradio
```

### 데이터 준비

```bash
# 샘플 CSV를 train/validation/test JSONL로 분할
uv run ditri-data-prep --input-path data/sample/incidents_synthetic.csv --output-dir data/processed --seed 42

# 또는 synthetic 데이터를 새로 생성하면서 준비
uv run ditri-data-prep --generate-synthetic --samples-per-label 60 --input-path data/sample/incidents_synthetic.csv --output-dir data/processed
```

### 학습

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --epochs 4
```

선택: PEFT(LoRA)

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --use-peft
```

### 평가

```bash
uv run ditri-eval \
  --model-path models/devops-incident-triage \
  --data-dir data/processed \
  --report-dir reports
```

생성 산출물:
- `reports/evaluation_metrics.json`
- `reports/per_label_metrics.json`
- `reports/confusion_matrix.csv`
- `reports/figures/confusion_matrix.png` (matplotlib/seaborn 설치 시)
- `reports/sample_predictions.jsonl`

### 로컬 추론

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --text "EKS worker nodes became NotReady after CNI upgrade."
```

배치 입력(.csv/.jsonl/.txt):

```bash
uv run ditri-predict \
  --model-path models/devops-incident-triage \
  --input-file data/sample/incidents_synthetic.csv \
  --text-column text \
  --output-file reports/batch_predictions.jsonl
```

## 6) FastAPI Serving

```bash
uv run ditri-api
```

기본 주소: `http://127.0.0.1:8000`

### Health check

```bash
curl -s http://127.0.0.1:8000/health
```

### Predict API

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"GitHub Actions deployment failed because IAM role assumption was denied."}'
```

예시 응답:

```json
{
  "label": "aws_iam_network",
  "confidence": 0.84,
  "scores": [
    {"label": "aws_iam_network", "score": 0.84},
    {"label": "cicd_pipeline", "score": 0.09}
  ]
}
```

## 7) Optional Gradio App

```bash
uv run ditri-gradio
```

## 8) Docker

```bash
docker build -t devops-incident-triage:latest .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/models/devops-incident-triage \
  devops-incident-triage:latest
```

## 9) Hugging Face Publish

사전 준비:

```bash
export HF_TOKEN="hf_xxx"
```

업로드:

```bash
uv run ditri-publish \
  --model-dir models/devops-incident-triage \
  --repo-id <your-hf-username>/DevOps-Incident-Triage-Model
```

`docs/model_card.md`가 모델 아티팩트에 `README.md`로 포함됩니다(모델 폴더에 README가 없을 때).

## 10) CI

GitHub Actions (`.github/workflows/ci.yml`)에서 다음을 수행합니다.

- `ruff` 린트
- `pytest`
- 데이터 준비 스모크 테스트

## 11) Git Branch & PR Workflow

- 브랜치 전략: `docs/branch_strategy.md`
- PR 템플릿: `.github/pull_request_template.md`
- 릴리스 체크리스트: `docs/release_checklist.md`

핵심:
- `main`: 릴리스 기준
- `develop`: 통합 개발
- `feature/*`, `release/*`, `hotfix/*` 운영

## 12) Repository Layout

```text
.
├─ pyproject.toml
├─ README.md
├─ Makefile
├─ Dockerfile
├─ .github/workflows/ci.yml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sample/
├─ src/devops_incident_triage/
│  ├─ config.py
│  ├─ labels.py
│  ├─ data_prep.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ hf_publish.py
│  └─ api.py
├─ app/gradio_app.py
├─ docs/
│  ├─ model_card.md
│  ├─ architecture.md
│  ├─ portfolio_notes.md
│  ├─ release_checklist.md
│  └─ portfolio_evidence.md
└─ tests/
```

## 13) Limitations

- 기본 샘플은 synthetic 데이터라 성능 해석에 제한이 큼
- 텍스트 길이/도메인 편향/라벨 불균형 대응이 아직 단순함
- 멀티라벨 및 계층형 분류는 후속 버전 범위

## 14) Next Steps (Portfolio 강화)

1. 비식별화된 실제 티켓/장애 리포트 데이터셋으로 교체
2. 라벨 가이드라인과 라벨 품질 점검(IAA) 도입
3. 멀티라벨 실험과 threshold 튜닝 추가
4. Drift/성능 저하 모니터링 대시보드 연동
5. HF Space와 API 배포 자동화(CI/CD) 연결

## 15) 운영형 릴리스 루틴 (권장)

1. `feature/* -> develop` PR로 기능 단위 통합
2. `release/vX.Y.Z` 생성 후 `docs/release_checklist.md` 기준 검증
3. `release/vX.Y.Z -> main` PR 머지 + 태그 생성
4. 실행 증거/지표를 `docs/portfolio_evidence.md`에 기록
