# Adult Census Income 분석 리포트

생성 시각 : 2026-07-21 16:29:57

---

## 1. 데이터 준비

- 원본 행 수 : 32561행
- 정제 후 행 수 : 32537행
- 정제 기준 : 결측 표기 `"?"` -> 각 컬럼의 최빈값(mode)으로 대체 (행 삭제 없음)

### 결측치 대체 내역

- `workclass` : 1836건 -> 최빈값으로 대체
- `occupation` : 1843건 -> 최빈값으로 대체
- `native_country` : 583건 -> 최빈값으로 대체

### 중복 제거

- 중복행 제거를 수행함 (`drop_duplicates()`)

---

## 2. 통계 분석

### 기술통계

|       |        age |   education_num |   capital_gain |   capital_loss |   hours_per_week |
|:------|-----------:|----------------:|---------------:|---------------:|-----------------:|
| count | 32537      |     32537       |       32537    |     32537      |       32537      |
| mean  |    38.5855 |        10.0818  |        1078.44 |        87.3682 |          40.4403 |
| std   |    13.638  |         2.57163 |        7387.96 |       403.102  |          12.3469 |
| min   |    17      |         1       |           0    |         0      |           1      |
| 25%   |    28      |         9       |           0    |         0      |          40      |
| 50%   |    37      |        10       |           0    |         0      |          40      |
| 75%   |    48      |        12       |           0    |         0      |          45      |
| max   |    90      |        16       |       99999    |      4356      |          99      |

### 상관계수

|                |   age |   education_num |   capital_gain |   capital_loss |   hours_per_week |
|:---------------|------:|----------------:|---------------:|---------------:|-----------------:|
| age            | 1     |           0.036 |          0.078 |          0.058 |            0.069 |
| education_num  | 0.036 |           1     |          0.123 |          0.08  |            0.148 |
| capital_gain   | 0.078 |           0.123 |          1     |         -0.032 |            0.078 |
| capital_loss   | 0.058 |           0.08  |         -0.032 |          1     |            0.054 |
| hours_per_week | 0.069 |           0.148 |          0.078 |          0.054 |            1     |

### t-test (그룹 = income, 비교 변수 = hours_per_week)

- t-statistic : `-45.0950`
- p-value : `0.000e+00`
- `<=50K` 그룹 평균 hours_per_week : `38.843`
- `>50K` 그룹 평균 hours_per_week : `45.473`
- 해석 : p-value < 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하다.

---

## 3. 시각화

- Seaborn 정적 차트(연령 분포) : [`eda_chart_seaborn.png`](eda_chart_seaborn.png)
- Plotly 인터랙티브 차트(소득 그룹별 근로시간 비교) : [`eda_chart_plotly.html`](eda_chart_plotly.html)

---

## 4. ML Pipeline

- 모델 구성 : `ColumnTransformer(StandardScaler + OneHotEncoder)` -> `RandomForestClassifier(n_estimators=200, max_depth=12)`
- 입력 변수 :
  - 수치형 (5개) : `age, education_num, capital_gain, capital_loss, hours_per_week`
  - 범주형 (8개) : `workclass, education, marital_status, occupation, relationship, race, sex, native_country`
  - `fnlwgt`, `income`은 feature에서 제외 (데이터 누수 방지)
- 분할 설정 : `test_size=0.2`, `random_state=42`
- train 건수 : 26029건
- test 건수 : 6508건
- **accuracy** : `0.8649`
- **F1-score** : `0.6791`

---

## 5. 본인 의견 / 개선 사항

> (이 섹션은 자동 생성되지 않습니다. 직접 작성해 주세요.)

-
-
-
