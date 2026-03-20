## PR 제목 권장 형식
`<type>: <short summary>`

예시: `feat: add multilabel evaluation report`

## 1) 변경 목적

이 PR이 해결하는 문제/목표를 간단히 작성해 주세요.

## 2) 주요 변경 내용

- 
- 

## 3) 브랜치 전략 체크

- [ ] 출발 브랜치가 정책에 맞음 (`feature/*`, `release/*`, `hotfix/*`)
- [ ] 대상 브랜치가 정책에 맞음 (`develop` 또는 `main`)
- [ ] `main` 직접 푸시 없이 PR로 진행

## 4) 데이터/모델 관련 변경 (해당 시)

- 데이터 출처:
  - [ ] synthetic
  - [ ] real (비식별화)
- 라벨/스키마 변경:
  - [ ] 있음
  - [ ] 없음
- 모델/학습 파라미터 변경:
  - [ ] 있음
  - [ ] 없음

설명:

## 5) 검증 결과

실행한 명령과 결과를 작성해 주세요.

```bash
uv run ruff check .
uv run pytest -q
uv run ditri-data-prep --input-path data/sample/incidents_synthetic.csv --output-dir data/processed --seed 42
```

결과 요약:

## 6) API/서빙 영향 (해당 시)

- [ ] FastAPI 스키마 변경 없음
- [ ] FastAPI 스키마 변경 있음 (하위호환 검토 필요)
- [ ] Docker 영향 있음

## 7) 리스크 및 롤백 계획

- 예상 리스크:
- 롤백 방법:

## 8) 문서 반영

- [ ] README 업데이트
- [ ] docs/model_card.md 업데이트
- [ ] docs/architecture.md 업데이트
- [ ] docs/portfolio_notes.md 업데이트
- [ ] 문서 업데이트 불필요

## 9) 관련 이슈/참고

- Closes #
- Related:
