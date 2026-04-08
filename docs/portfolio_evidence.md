# Portfolio Evidence Log

포트폴리오 공개 시 "무엇을 실제로 실행했고 어떤 결과가 나왔는지"를 기록하는 템플릿입니다.

## 1) 실행 환경

- 날짜:
- Python:
- uv:
- OS:
- 장비(CPU/GPU/RAM):

## 2) 데이터 정보

- 데이터 유형: synthetic / real (비식별화)
- 샘플 수:
- 라벨 수:
- split 비율:

## 3) 실행 명령 기록

```bash
uv run ditri-data-prep --input-path ... --output-dir ... --seed 42
uv run ditri-train --data-dir ... --output-dir ... --model-name ... --epochs ...
uv run ditri-eval --model-path ... --data-dir ... --report-dir ...
```

## 4) 핵심 지표 요약

| Metric | Value | Notes |
|---|---:|---|
| accuracy |  |  |
| macro_f1 |  |  |
| weighted_f1 |  |  |

추가: label별 precision/recall/f1은 `reports/per_label_metrics.json` 링크

## 4-1) 베이스라인 비교 요약 (선택)

- 실행 명령:

```bash
uv run ditri-benchmark --data-dir data/processed --skip-existing
```

- 결과 파일:
  - `reports/model_benchmark.json`
  - `reports/model_benchmark.md`
- 최종 채택 모델:

## 5) 에러 분석 메모

- 오분류 패턴 1:
- 오분류 패턴 2:
- 데이터/라벨 개선 아이디어:

## 6) API 검증 결과

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"..."}'
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["...","..."]}'
```

- `/health` 결과:
- `/predict` 결과:
- `/predict/batch` 결과:

## 7) 공개 링크

- GitHub Repo:
- GitHub Release:
- Hugging Face Model:
- (Optional) Hugging Face Space:

## 8) 정직성 선언

이 프로젝트의 공개 지표는 아래 데이터 기준입니다.

- [ ] synthetic starter 데이터
- [ ] real 비식별화 데이터

synthetic 기준 결과는 일반화 성능을 보장하지 않으며, 실데이터 재평가가 필요함을 명시합니다.
