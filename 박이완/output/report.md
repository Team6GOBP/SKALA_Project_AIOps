# End2End 데이터 분석 리포트 — Adult Census Income

- 생성 일시: 2026-07-21 16:40
- 본 리포트는 `main.py` 실행 시 Jinja2 템플릿으로 **자동 생성**됩니다. (수동 편집 없음)

## 1. 데이터 준비 (Pandas vs Polars) — 팀 SPEC 1 적용

| 항목 | Pandas | Polars |
|---|---|---|
| shape (행, 열) | (32537, 15) | (32562, 15) |
| 로딩 시간(초) | 0.043 | 0.05 |

- 원본 결측치 총 **4262건** — 컬럼별: {'workclass': 1836, 'occupation': 1843, 'native-country': 583}
- **처리 정책 (SPEC 1)**: 삭제하지 않고 컬럼별 **최빈값(mode)** 으로 대체
  - 대체에 사용된 최빈값: {'workclass': 'Private', 'occupation': 'Prof-specialty', 'native-country': 'United-States'}
- 중복 행 24건 제거
- income 분포: {'<=50K': 24698, '>50K': 7839}

## 2. 시각화 — 팀 SPEC 5 파일명

- Seaborn 정적 차트: `eda_chart_seaborn.png` (소득 그룹별 주당 근로시간 boxplot)
- Plotly 인터랙티브 차트: `eda_chart_plotly.html` (학력별 평균 근로시간 bar)

## 3. 통계 분석 — 팀 SPEC 2·4

### 3-1. 기술통계 (평균·표준편차·분위수) — 수치형 5개 (fnlwgt 제외, SPEC 4)

|       |      age |   education-num |   capital-gain |   capital-loss |   hours-per-week |
|:------|---------:|----------------:|---------------:|---------------:|-----------------:|
| count | 32537    |        32537    |       32537    |       32537    |         32537    |
| mean  |    38.59 |           10.08 |        1078.44 |          87.37 |            40.44 |
| std   |    13.64 |            2.57 |        7387.96 |         403.1  |            12.35 |
| min   |    17    |            1    |           0    |           0    |             1    |
| 25%   |    28    |            9    |           0    |           0    |            40    |
| 50%   |    37    |           10    |           0    |           0    |            40    |
| 75%   |    48    |           12    |           0    |           0    |            45    |
| max   |    90    |           16    |       99999    |        4356    |            99    |

### 3-2. 상관행렬

|                |   age |   education-num |   capital-gain |   capital-loss |   hours-per-week |
|:---------------|------:|----------------:|---------------:|---------------:|-----------------:|
| age            | 1     |           0.036 |          0.078 |          0.058 |            0.069 |
| education-num  | 0.036 |           1     |          0.123 |          0.08  |            0.148 |
| capital-gain   | 0.078 |           0.123 |          1     |         -0.032 |            0.078 |
| capital-loss   | 0.058 |           0.08  |         -0.032 |          1     |            0.054 |
| hours-per-week | 0.069 |           0.148 |          0.078 |          0.054 |            1     |

### 3-3. t-test — income 그룹별 주당 근로시간

- 인자 순서: `ttest_ind(<=50K, >50K, equal_var=False)` (SPEC 2)
- `<=50K` 평균 **38.84시간** vs `>50K` 평균 **45.47시간**
- t = **-45.095**, p-value = **0**
- **해석**: p < 0.05 이므로 유의미하다 — 두 소득 그룹의 주당 근로시간 평균 차이는 통계적으로 유의미하다 (귀무가설 기각).

## 4. ML Pipeline (소득 >50K 이진 분류) — 팀 SPEC 3·4

- 타깃 인코딩 (SPEC 3): `<=50K → 0`, `>50K → 1`
- 특성 구성 (SPEC 4): 수치형 **5개** + 범주형 **8개** (fnlwgt 제외)
- 공통 설정 (SPEC 4): `test_size=0.2`, `random_state=42`
- 학습/평가 샘플: 26029 / 6508
- **정확도(accuracy): 0.856**
- **F1-score: 0.675**
- 저장 모델: `model.joblib` (joblib 재로딩 예측 일치: True)

## 5. 결론 및 의견

- t-test 결과는 위 3-3 절의 해석 문장을 그대로 반영한다. income 그룹 간 근로시간 차이의 통계적 유의성 여부가 곧 이 프로젝트의 핵심 인사이트다.
- 전처리·모델을 하나의 Pipeline 으로 묶어 `model.joblib` 로 저장했으므로, 배포 환경에서도 동일한 전처리 + 예측이 보장된다 (데이터 누수 위험 없음: income 은 y 로만 사용, fnlwgt 는 X 에서 제외).
- 개선 아이디어: (1) RandomForest·XGBoost 등 모델 비교, (2) capital-gain 왜도 완화(log 변환), (3) 교차검증(cross_val_score) 도입.