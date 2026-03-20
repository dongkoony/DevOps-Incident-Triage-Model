# Real Data Ingestion Skeleton

이 문서는 synthetic starter에서 실데이터 파이프라인으로 넘어갈 때의 최소 구조를 정의합니다.

## 목표

- 운영 인시던트 원문(CSV)을 안전하게 수집
- 민감정보를 마스킹
- 학습 파이프라인(`ditri-data-prep`)에 바로 연결 가능한 포맷 생성
- 품질 리포트(JSON)로 누락/중복/라벨 상태 확인

## 입력 스키마

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

## 실행

```bash
uv run ditri-ingest-raw \
  --input-path data/raw/incidents_template.csv \
  --output-canonical-path data/raw/incidents_canonical.csv \
  --output-training-path data/raw/incidents_training_ready.csv \
  --report-path reports/raw_ingestion_report.json
```

## 출력

- `data/raw/incidents_canonical.csv`
  - 정규화 + 마스킹 + 텍스트 합성 결과를 포함
- `data/raw/incidents_training_ready.csv`
  - `text,label,source` 포맷
  - `ditri-data-prep --input-path ...`에 즉시 사용 가능
- `reports/raw_ingestion_report.json`
  - 누락/중복/라벨 분포/민감정보 패턴 hit 수

## 마스킹 규칙 (기본 활성)

- 이메일 -> `[REDACTED_EMAIL]`
- IPv4 -> `[REDACTED_IP]`
- AWS 12자리 계정 ID -> `[REDACTED_AWS_ACCOUNT]`

비활성화 옵션:

```bash
uv run ditri-ingest-raw --disable-sensitive-masking
```

## 실무 확장 포인트

1. Jira/ServiceNow/Datadog API 수집기로 raw CSV 자동 생성
2. 라벨 매핑 테이블(legacy label -> v1 label) 추가
3. dedup 전략을 `incident_id` 외에 티켓 링크/해시 기준으로 확장
4. DLP 룰 강화(토큰/키 패턴, URL query secret 등)
