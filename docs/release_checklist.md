# Release Checklist

`DevOps-Incident-Triage-Model`의 공개 릴리스를 위한 실무형 점검표입니다.

## 1) 브랜치/버전 준비

- [ ] `develop` 최신화 완료
- [ ] `release/vX.Y.Z` 브랜치 생성
- [ ] 릴리스 버전 결정 (`v0.1.0` 등)
- [ ] 릴리스 노트 초안 작성

## 2) 품질 게이트

```bash
uv run ruff check .
uv run pytest -q
uv run ditri-data-prep --input-path data/sample/incidents_synthetic.csv --output-dir data/processed --seed 42
```

- [ ] 린트 통과
- [ ] 단위 테스트 통과
- [ ] 데이터 준비 스모크 통과

## 3) 학습/평가 산출물 검증

실데이터 수집 파이프라인 사용 시:

```bash
uv run ditri-ingest-raw \
  --input-path data/raw/incidents_template.csv \
  --output-canonical-path data/raw/incidents_canonical.csv \
  --output-training-path data/raw/incidents_training_ready.csv \
  --report-path reports/raw_ingestion_report.json
```

```bash
uv run ditri-train \
  --data-dir data/processed \
  --output-dir models/devops-incident-triage \
  --model-name distilbert-base-uncased \
  --epochs 4

uv run ditri-eval \
  --model-path models/devops-incident-triage \
  --data-dir data/processed \
  --report-dir reports
```

- [ ] `models/devops-incident-triage/` 생성 확인
- [ ] `reports/evaluation_metrics.json` 확인
- [ ] `reports/per_label_metrics.json` 확인
- [ ] `reports/confusion_matrix.csv` 확인
- [ ] synthetic 데이터 기반 결과임을 문서에 명시

## 4) API/배포 점검

```bash
uv run ditri-api
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"GitHub Actions deployment failed because IAM role assumption was denied."}'
curl -s -X POST http://127.0.0.1:8000/predict/batch/async \
  -H "Content-Type: application/json" \
  -d '{"texts":["Node NotReady after upgrade.","Ambiguous release failure in mixed logs."]}'
curl -s http://127.0.0.1:8000/metrics
```

- [ ] `/health` 응답 확인
- [ ] `/predict` 응답 스키마 확인
- [ ] `/predict/batch/async` 202 응답 + `job_id` 확인
- [ ] `/predict/batch/async/{job_id}` 완료 상태 확인
- [ ] `X-Request-ID` 헤더 반환 확인
- [ ] `/metrics` 노출 지표 확인
- [ ] Docker build/run 검증

## 5) 문서/포트폴리오 점검

- [ ] `README.md` 최신화
- [ ] `docs/model_card.md` 최신화
- [ ] `docs/architecture.md` 최신화
- [ ] `docs/portfolio_notes.md` 최신화
- [ ] `docs/portfolio_evidence.md`에 실행 증거 기록

## 6) Hugging Face 배포

```bash
export HF_TOKEN="hf_xxx"
uv run ditri-publish \
  --model-dir models/devops-incident-triage \
  --repo-id <your-hf-username>/DevOps-Incident-Triage-Model
```

- [ ] 모델 업로드 완료
- [ ] 모델 카드 노출 확인
- [ ] 추론 위젯 동작 확인

## 7) Git 릴리스 절차

- [ ] `release/vX.Y.Z -> main` PR 머지
- [ ] Git tag 생성 (`vX.Y.Z`)
- [ ] 릴리스 노트 게시
- [ ] `main` 변경사항을 `develop`으로 역반영
