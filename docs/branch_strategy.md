# Git Branch Strategy (GitFlow Lite)

이 저장소는 포트폴리오 프로젝트 특성과 개인/소규모 협업 현실성을 고려해 **GitFlow Lite**를 사용합니다.

## 1) 브랜치 역할

- `main`
  - 릴리스 기준 브랜치
  - 항상 배포/공개 가능한 상태 유지
- `develop`
  - 다음 릴리스를 위한 통합 개발 브랜치
- `feature/<topic>`
  - 기능/리팩터링/문서 단위 작업 브랜치
- `release/vX.Y.Z`
  - 릴리스 직전 안정화(문서, 버전, 최종 테스트)
- `hotfix/<topic>`
  - `main` 긴급 이슈 수정

## 2) 브랜치 네이밍 규칙

- `feature/add-multilabel-experiment`
- `feature/api-batch-predict`
- `release/v0.1.0`
- `hotfix/fix-api-health-check`

## 3) 머지 정책

### feature -> develop
- PR 필수
- CI 통과(`ruff`, `pytest`, data-prep smoke)
- 리뷰 1회 이상 권장
- 머지 방식: `Squash and merge`

### release -> main
- PR 필수
- 버전/문서/체인지 검토 완료
- 머지 방식: `Create a merge commit` (릴리스 맥락 보존)
- 머지 후 태그 생성: `vX.Y.Z`

### release -> develop (반영)
- `main` 릴리스 완료 후 동일 변경을 `develop`에 역반영
- 머지 방식: `Create a merge commit`

### hotfix -> main, develop
- `hotfix/*`를 `main`에 먼저 반영 후 즉시 태그
- 동일 수정을 `develop`에도 반영(머지 또는 cherry-pick)

## 4) 금지/권장 규칙

- 금지
  - `main` 직접 푸시
  - `develop` 직접 푸시(긴급 상황 제외)
- 권장
  - 작은 단위 PR
  - 변경 의도와 검증 결과를 PR 본문에 명확히 기록
  - synthetic/real data 여부를 PR에 명시

## 5) 권장 커밋 메시지

Conventional Commits 스타일 권장:

- `feat: add multilabel training scaffold`
- `fix: handle empty input in prediction API`
- `docs: add branch strategy and PR workflow`
- `chore: update CI matrix`

## 6) 기본 작업 플로우

1. `develop` 최신화
2. `feature/*` 생성
3. 구현 + 테스트 + 문서
4. `feature/* -> develop` PR
5. 릴리스 시 `release/* -> main` PR 및 태깅
6. 릴리스 브랜치 변경사항을 `develop`으로 역반영

예시:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/add-hf-space-scaffold

# 작업
git add .
git commit -m "feat: add gradio space scaffold and docs"
git push -u origin feature/add-hf-space-scaffold
```

## 7) GitHub 브랜치 보호(권장)

GitHub Settings -> Branch protection rules:

- `main`
  - Require a pull request before merging
  - Require status checks to pass
  - Restrict pushes
- `develop`
  - Require a pull request before merging
  - Require status checks to pass
