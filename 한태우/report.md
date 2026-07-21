# Adult Census Income 분석 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 Pandas와 Polars로 처리하고,
시각화·통계 검정·머신러닝 분류를 수행하였다.

## 2. 데이터 정제

- 정제 후 데이터: 32,537행 × 15열
- `<=50K`: 24,698건
- `>50K`: 7,839건
- 제거된 중복: 24건
- 정제 후 결측치: 0건
- 결측치 처리: `workclass`, `occupation`, `native-country` 최빈값 대체
- `fnlwgt`: 상관계수와 ML feature에서 제외

## 3. 시각화

- `eda_chart_seaborn.png`: 소득 그룹별 주당 근무시간
- `eda_chart_plotly.html`: 교육 수준별 고소득자 비율

## 4. Welch t-test

- `<=50K` 평균: 38.843
- `>50K` 평균: 45.473
- t-statistic: -45.095026
- p-value: 0.000000e+00
- 해석: p < 0.05 → 두 소득 그룹의 주당 근무시간 평균 차이는 통계적으로 유의미하다.

## 5. 머신러닝

- 타깃: `<=50K` → 0, `>50K` → 1
- 모델: Logistic Regression
- test_size: 0.2
- random_state: 42
- 학습 데이터: 26,029건
- 테스트 데이터: 6,508건
- Accuracy: 0.8563
- F1-score: 0.6750

## 6. 산출물

- `main.py`
- `eda_chart_seaborn.png`
- `eda_chart_plotly.html`
- `model.joblib`
- `report.md`

## 7. 결론

고소득 그룹의 평균 주당 근무시간이 더 높았으며,
Welch t-test 결과 그 차이는 통계적으로 유의미했다.
분류 모델은 Accuracy 0.8563,
F1-score 0.6750를 기록했다.
