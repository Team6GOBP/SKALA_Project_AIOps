# Day2 종합실습 - 팀 공통 SPEC (확정)

각자 폴더(`팀원이름/`)에서 구현할 때 아래 규칙을 반드시 지켜주세요.
이 규칙이 안 지켜지면 나중에 결과 비교/병합이 안 됩니다.

## 1. 결측치 처리

- 결측 표기: `na_values="?"` (공백 없이! `" ?"`로 쓰면 결측치가 하나도 안 잡히는 버그가 있음)
- 결측 컬럼: `workclass`, `occupation`, `native-country` (전체의 약 5~7%)
- 처리 방식: **삭제하지 말고 최빈값(mode)으로 대체**
  - 이유: 결측 비율이 크지 않고, 행을 삭제하면 나중에 ML 학습 데이터가 줄어들어
    팀원 간 성능 비교가 불공정해짐
- `fnlwgt`는 결측 없음, 그대로 둠 (단, 아래 4번 참고 — 상관계수/모델 feature에서는 제외)

## 2. t-test 그룹

- **income 그룹(`<=50K` vs `>50K`)의 `hours-per-week` 평균 차이**로 통일
- `scipy.stats.ttest_ind(group_50k_under, group_50k_over, equal_var=False)` 사용
  (두 그룹 분산이 같다고 가정하지 않는 Welch's t-test)
- 출력에 t-statistic, p-value, 그리고 "p < 0.05 → 유의미하다/않다" 해석 문장 반드시 포함

## 3. ML 타깃 인코딩

- `income` 컬럼: `<=50K` → `0`, `>50K` → `1`
- 원본 값에 앞뒤 공백이 섞여 있을 수 있으니 매핑 전 `.str.strip()` 먼저 적용
- 분류 문제이므로 평가지표는 **accuracy + F1-score** 둘 다 출력

## 4. 분석에 사용할 컬럼 고정

- **상관계수 대상 수치형 5개**: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- `fnlwgt`는 제외 (인구총조사 가중치 값이라 소득과 직접적인 인과/상관 의미가 없는 컬럼)
- **ML Pipeline feature**: 위 수치형 5개 + 범주형 전체
  (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`)
  → `fnlwgt`는 여기서도 제외
- **ML Pipeline 공통 설정**: `test_size=0.2`, `random_state=42` (다르면 팀원 간 성능 비교가 안 됨)

## 5. 산출물 파일명 규칙

각자 폴더(`팀원이름/`) 안에 아래 이름 그대로 저장:

| 산출물 | 파일명 |
|---|---|
| Seaborn 정적 차트 | `eda_chart_seaborn.png` |
| Plotly 인터랙티브 차트 | `eda_chart_plotly.html` |
| 학습된 모델 | `model.joblib` |
| 자동 생성 리포트 | `report.md` |
| 메인 실행 스크립트 | `main.py` |

## 6. 병합(PM) 기준

병합 담당자는 아래 5개 항목을 각자 결과물끼리 비교해서 항목별로 제일 나은 것을 골라 `final/` 폴더로 합칩니다 (통째로 한 사람 코드를 쓰지 않음):

1. 결측치·중복 처리 — 예외 처리가 가장 꼼꼼한 버전
2. 시각화 — 라벨·제목이 명확하고 인사이트가 잘 드러나는 버전
3. 통계 검정 — p-value 해석 문장이 정확한 버전
4. ML Pipeline — 데이터 누수 없는 버전 (타깃과 직접 연산되는 컬럼을 feature에 넣지 않았는지 확인)
5. report.md — 자동화 완성도 (수동 편집 없이 스크립트만 돌려도 나오는지)