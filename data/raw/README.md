# Raw Data

실제 운영 로그/티켓 원문(비식별화 완료본)을 둘 디렉터리입니다.

- 예시: `incidents_raw.csv`, `tickets_export.jsonl`
- 개인정보/민감정보/보안정보는 저장 전에 반드시 마스킹하세요.

## 권장 CSV 스키마 (v1)

필수 컬럼:
- `incident_id`
- `occurred_at`
- `source`
- `summary`
- `label`

선택 컬럼:
- `details`
- `service`
- `environment`
- `severity`
- `region`

샘플 템플릿: `data/raw/incidents_template.csv`

## 변환 CLI

```bash
uv run ditri-ingest-raw \
  --input-path data/raw/incidents_template.csv \
  --output-canonical-path data/raw/incidents_canonical.csv \
  --output-training-path data/raw/incidents_training_ready.csv \
  --report-path reports/raw_ingestion_report.json
```

`incidents_training_ready.csv`는 `ditri-data-prep --input-path ...`에 바로 연결할 수 있습니다.
