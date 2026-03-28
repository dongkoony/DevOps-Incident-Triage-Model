# Architecture

## 목적

`DevOps-Incident-Triage-Model`은 DevOps 인시던트 텍스트를 입력받아
초동 담당 영역을 분류하는 경량 NLP 파이프라인입니다.

## 흐름

1. Data Prep
   - 입력 CSV(`text`, `label`)를 읽고 검증
   - stratified split(train/validation/test)
   - JSONL 포맷 + `label_id` 저장
2. Train
   - Hugging Face `transformers` 기반 sequence classification 학습
   - baseline: `distilbert-base-uncased`
   - 선택적 PEFT LoRA 지원
3. Evaluate
   - accuracy, macro F1, per-label metrics
   - confusion matrix CSV(+PNG)
   - confidence threshold metrics(자동 분류 커버리지 vs 수동 검토 비율)
   - sample prediction dump
4. Inference & Serving
   - CLI 단건/배치 예측
   - FastAPI `/predict`, `/predict/batch` API
   - 저신뢰 예측을 `needs_human_review`로 라우팅하는 threshold gating
   - 배치 요청은 `BATCH_MAX_ITEMS`로 상한 제어
   - `X-Request-ID` 기반 요청 추적(응답 헤더 포함)
   - `/metrics`를 통한 Prometheus 호환 운영 지표 노출
   - Optional Gradio app
5. Publish
   - Hugging Face Hub 모델 리포지토리 업로드
   - 모델 카드 포함

## 설계 선택

- **Multiclass 우선**: 운영 triage에서는 단일 주 담당 분류가 초기 도입 난이도와 실효성이 좋음
- **경량 모델**: 개인 포트폴리오 환경에서 학습/추론 비용 현실성 확보
- **정직한 데이터 문서화**: synthetic 스타터를 명시하고 실데이터 전환 경로 제공

## 주요 아티팩트

- `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`
- `models/devops-incident-triage/`
- `reports/evaluation_metrics.json`
- `reports/per_label_metrics.json`
- `reports/confusion_matrix.csv`
- `reports/sample_predictions.jsonl`
