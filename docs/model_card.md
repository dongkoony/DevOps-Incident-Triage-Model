# Model Card: DevOps-Incident-Triage-Model

## Model Details

- Model name: `DevOps-Incident-Triage-Model`
- Base checkpoint: `distilbert-base-uncased` (baseline)
- Task: DevOps incident text classification (multiclass)
- Labels:
  - `k8s_cluster`
  - `cicd_pipeline`
  - `aws_iam_network`
  - `deployment_release`
  - `container_runtime`
  - `observability_alerting`
  - `database_state`

## Intended Use

이 모델은 DevOps/Platform/SRE 환경에서 인시던트 텍스트를 빠르게 분류하여
초동 대응 라우팅을 보조하기 위한 용도입니다.

- 권장: 온콜 triage 보조, 티켓 자동 태깅 보조
- 비권장: 완전 자동 의사결정, 인적 검토 없는 조치 실행
- 운영 권장: confidence threshold gating을 통해 저신뢰 예측은 `needs_human_review`로 보냄
- 운영 권장: FastAPI `/predict/batch` 사용 시 배치 상한(`BATCH_MAX_ITEMS`)을 둬 API 안정성 확보

## Training Data

현재 버전은 `data/sample/incidents_synthetic.csv` 기반의 synthetic starter 데이터로 학습됩니다.

- 이 데이터는 실서비스에서 직접 수집된 로그/티켓이 아닙니다.
- 실제 일반화 성능은 운영 데이터로 재학습/재평가해야 검증됩니다.

## Training Procedure

- Data split: train/validation/test
- Input max length: 256
- Metrics: accuracy, macro F1, per-label precision/recall/F1
- Optional: PEFT LoRA (`--use-peft`)

## Evaluation

평가 스크립트:

```bash
uv run ditri-eval --model-path models/devops-incident-triage --data-dir data/processed --report-dir reports
```

산출물:
- `reports/evaluation_metrics.json`
- `reports/per_label_metrics.json`
- `reports/confusion_matrix.csv`
- `reports/sample_predictions.jsonl`

## Limitations

- Synthetic 데이터 중심이라 도메인 편향/표현 다양성이 제한적입니다.
- 라벨 정의가 단일 주 라벨(multiclass)이라 복합 원인 인시던트 반영이 약합니다.
- 장문 로그/대량 컨텍스트(멀티라인 스택트레이스) 처리 성능은 추가 검증이 필요합니다.

## Ethical and Operational Considerations

- 모델 예측은 우선순위 판단 보조이며, 최종 판단은 운영자에게 있습니다.
- 오분류 시 잘못된 라우팅과 대응 지연이 발생할 수 있어 human-in-the-loop가 필요합니다.
- 운영 로그에 민감정보가 포함될 수 있으므로 데이터 비식별화가 선행되어야 합니다.

## Recommended Next Improvements

1. 실제 비식별화 인시던트 데이터셋 구축
2. 멀티라벨/계층형 분류 실험
3. 라벨링 가이드 및 품질 지표(IAA) 도입
4. 오프라인 + 온라인 모니터링(데이터/모델 드리프트) 연계
