# Day2 종합실습 리포트

## 1. 데이터 준비 (pandas vs polars 비교)

| 라이브러리 | 원본 행 수 | 중복 제거 건수 | 중복 제거 후 행 수 | 소요시간(s) |
|---|---|---|---|---|
| pandas | 32561 | 24 | 32537 | 0.0448 |
| polars | 32561 | 24 | 32537 | 0.0107 |

- 두 라이브러리의 중복 제거 후 행 수 일치: True
- 두 라이브러리의 income 분포 일치: True

## 2. 결측치 처리 (중복 제거 후, mode 대체)

| 컬럼 | 결측 개수(처리 전) | 최빈값 대체 | 결측 개수(처리 후) |
|---|---|---|---|
| workclass | 1836 | Private | 0 |
| occupation | 1843 | Prof-specialty | 0 |
| native_country | 582 | United-States | 0 |

## 3. 소득 그룹별 기초 통계

| income   |   age_mean |   age_median |   hours_per_week_mean |   hours_per_week_median |   education_num_mean |   education_num_median |
|:---------|-----------:|-------------:|----------------------:|------------------------:|---------------------:|-----------------------:|
| <=50K    |      36.79 |           34 |                 38.84 |                      40 |                 9.6  |                      9 |
| >50K     |      44.25 |           44 |                 45.47 |                      40 |                11.61 |                     12 |

**capital_gain 평균 vs 중앙값 (쏠림 확인)**

| income   |    mean |   median |
|:---------|--------:|---------:|
| <=50K    |  148.88 |        0 |
| >50K     | 4007.16 |        0 |

- 전체 평균: 1078.44, 전체 중앙값: 0.00
- 해석: capital_gain 전체 평균은 1078.44달러인데 중앙값은 0.00달러입니다. 평균이 중앙값보다 훨씬 크다는 것은 대다수(중앙값 기준 절반 이상)는 capital_gain이 0이고, 소수의 인원이 매우 큰 자본 이득을 얻으면서 전체 평균을 끌어올리는 극단적인 쏠림(우측 꼬리 분포)이 존재한다는 증거입니다.

## 4. t-test

### income x 주당 근무시간(hours_per_week)

- H0: 두 소득 그룹의 평균 주당 근무시간(hours_per_week)는 같다 / H1: 다르다
- `<=50K` 평균: 38.8429
- `>50K` 평균: 45.4734
- t-statistic: -45.0950
- p-value: < 1e-300
- 해석: p-value(< 1e-300) < 0.05 이므로 귀무가설(H0)을 기각한다: 두 소득 그룹 간 주당 근무시간(hours_per_week) 평균 차이는 통계적으로 유의미하다.

### income x 교육수준(education_num)

- H0: 두 소득 그룹의 평균 교육수준(education_num)는 같다 / H1: 다르다
- `<=50K` 평균: 9.5961
- `>50K` 평균: 11.6122
- t-statistic: -64.8761
- p-value: < 1e-300
- 해석: p-value(< 1e-300) < 0.05 이므로 귀무가설(H0)을 기각한다: 두 소득 그룹 간 교육수준(education_num) 평균 차이는 통계적으로 유의미하다.

## 5. 상관계수 (수치형 5개 컬럼, fnlwgt 제외)

|                |    age |   education_num |   capital_gain |   capital_loss |   hours_per_week |
|:---------------|-------:|----------------:|---------------:|---------------:|-----------------:|
| age            | 1      |          0.0362 |         0.0777 |         0.0577 |           0.0685 |
| education_num  | 0.0362 |          1      |         0.1227 |         0.0799 |           0.1484 |
| capital_gain   | 0.0777 |          0.1227 |         1      |        -0.0316 |           0.0784 |
| capital_loss   | 0.0577 |          0.0799 |        -0.0316 |         1      |           0.0542 |
| hours_per_week | 0.0685 |          0.1484 |         0.0784 |         0.0542 |           1      |

**상관계수 순위 (절댓값 기준 내림차순)**

- education_num - hours_per_week: 0.1484
- education_num - capital_gain: 0.1227
- education_num - capital_loss: 0.0799
- capital_gain - hours_per_week: 0.0784
- age - capital_gain: 0.0777
- age - hours_per_week: 0.0685
- age - capital_loss: 0.0577
- capital_loss - hours_per_week: 0.0542
- age - education_num: 0.0362
- capital_gain - capital_loss: -0.0316

- 가장 연관성이 높은 변수 쌍: **education_num - hours_per_week** (r=0.1484)

## 6. EDA 차트

- `eda_chart_seaborn.png`: 수치형 5개 컬럼 상관관계 히트맵 (static heatmap)
- `eda_chart_plotly.html`: 수치형 5개 컬럼 상관관계 히트맵 (interactive heatmap, hover로 정확한 값 확인)

## 7. ML Pipeline 결과

- Feature 컬럼: age, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country
- Train / Test 크기: 26029 / 6508 (test_size=0.2, random_state=42)
- Accuracy: 0.8637
- F1-score: 0.6759
- 저장된 모델: `model.joblib`
