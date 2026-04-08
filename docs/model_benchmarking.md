# Model Benchmarking Guide

`ditri-benchmark`는 동일 데이터/하이퍼파라미터 조건에서 여러 베이스 모델을 연속으로 학습/평가하고,
비교 가능한 결과 리포트를 생성합니다.

## 실행 예시

```bash
uv run ditri-benchmark \
  --data-dir data/processed \
  --models distilbert-base-uncased,sentence-transformers/all-MiniLM-L6-v2,xlm-roberta-base \
  --epochs 4 \
  --skip-existing
```

## 주요 출력물

- `reports/model_benchmark.json`
  - 모델별 상태(`completed`, `skipped`, `failed`)
  - `test_accuracy`, `test_macro_f1`, `weighted_f1`
  - 학습/평가 소요 시간
  - `best_model_by_test_macro_f1`
- `reports/model_benchmark.md`
  - 포트폴리오/PR 본문에 바로 붙일 수 있는 표 형식 요약
- `models/benchmarks/<model-slug>/`
  - 모델별 학습 아티팩트
- `reports/benchmarks/<model-slug>/`
  - 모델별 평가 리포트

## 실무 사용 팁

- 비교 실험은 같은 split, 같은 epoch/seed를 유지해야 공정합니다.
- synthetic 데이터 기반 점수는 일반화 성능 보장이 아니므로, 실데이터 재평가 결과를 함께 제시하세요.
- 실패한 모델이 있어도 기본 동작은 나머지 모델 평가를 계속 진행합니다.
  - 즉시 중단하려면 `--fail-fast`를 사용하세요.
