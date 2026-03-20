# Portfolio Notes

## 이 프로젝트를 포트폴리오에서 어떻게 말할까

핵심 메시지:
- "운영 인시던트 triage를 자동화 보조하는 실무형 NLP 분류기"
- "모델 성능 과장이 아니라 재현성/평가/배포 관점의 MLOps 완성도"
- "synthetic 시작점을 명확히 밝히고 실데이터 확장 계획을 제시"

## 인터뷰에서 강조 포인트

1. 문제정의
   - DevOps 환경에서 인시던트 라우팅 지연은 MTTR 증가로 직결
   - 텍스트 분류를 통해 초동 triage 시간을 단축
2. 엔지니어링
   - `uv + pyproject.toml` 기반 재현 가능한 환경
   - 학습/평가/추론/API/배포 스크립트 분리
   - Docker + GitHub Actions로 실행 경로 표준화
3. 신뢰성
   - synthetic 데이터 한계를 명시
   - per-label 지표와 confusion matrix로 약점 공개
4. 운영 확장성
   - HF Hub 게시 가능
   - 추후 멀티라벨, 드리프트 모니터링, active learning 연결 가능

## 권장 고도화 로드맵

### 단기 (1~2주)
- 실제 비식별화 인시던트 데이터 500~2,000건 수집
- 라벨링 가이드 문서화 + 라벨 품질 점검
- 베이스라인 2~3개 비교(`distilbert`, `MiniLM`, `xlm-roberta-base`)

### 중기 (2~4주)
- 멀티라벨 분류 실험(BCEWithLogits + threshold tuning)
- 에러 분석 리포트 자동 생성(오분류 패턴별)
- API에 배치 추론 엔드포인트 추가

### 장기 (4주+)
- 온라인 추론 로그 기반 드리프트 감지
- 모델 버전별 A/B 평가
- HF Space 데모 + GitHub Actions 릴리스 자동화
