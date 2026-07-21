# Day2 종합실습 - Adult Census Income 분석 리포트

이 리포트는 `광주_2반_강주현.py` 실행 결과를 바탕으로 자동 생성되었습니다.
데이터셋: [Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

## 1. 데이터 준비

### 1-1. Pandas vs Polars 로딩 비교

| | Pandas | Polars |
|---|---|---|
| 로딩 시간 | 2.8396초 | 2.8339초 |
| shape | (32561, 15) | (32561, 15) |
| 중복행 | 24건 | 24건 |

원본 결측치: `workclass` 1836건, `occupation` 1843건, `native-country` 583건

### 1-2. 결측치·중복 처리 (SPEC.md 규칙: 최빈값 대체, 중복행 제거)

- `workclass`: 1,836건 → 0건 (대체값: `Private`)
- `occupation`: 1,843건 → 0건 (대체값: `Prof-specialty`)
- `native-country`: 583건 → 0건 (대체값: `United-States`)

- 중복행 제거: 32,561행 → 32,537행 (24건 제거)

### 1-3. income 클래스 비율 (불균형 확인)

- `<=50K`: 75.91%
- `>50K`: 24.09%

> 참고: `<=50K`가 다수 클래스라 ML 평가 시 accuracy만으로는 부족해 F1도 함께 확인했습니다.

## 2. 시각화

| 카테고리 | 라이브러리 | 내용 | 파일 |
|---|---|---|---|
| 상관관계 (필수) | Seaborn | 수치형 5개 변수 상관 히트맵 | [eda_chart_seaborn.png](eda_chart_seaborn.png) |
| 그룹비교 (필수) | Plotly | 학력별 고소득(>50K) 비율 | [eda_chart_plotly.html](eda_chart_plotly.html) |
| 그룹비교 (추가) | Seaborn | income별 근무시간 박스플롯 | [eda_chart_seaborn_groupcompare.png](eda_chart_seaborn_groupcompare.png) |
| 분포 (추가) | Plotly | income별 나이 분포 히스토그램 | [eda_chart_plotly_distribution.html](eda_chart_plotly_distribution.html) |

![상관관계 히트맵](eda_chart_seaborn.png)

![income별 근무시간 박스플롯](eda_chart_seaborn_groupcompare.png)

> Plotly 인터랙티브 차트는 이미지로 표시되지 않으니 위 링크를 눌러 직접 열어서 확인하세요.

## 3. 통계 분석

### 3-1. 기술통계 (평균·표준편차·분위수)

|  | age | education-num | capital-gain | capital-loss | hours-per-week |
|---|---|---|---|---|---|
| count | 32537.0000 | 32537.0000 | 32537.0000 | 32537.0000 | 32537.0000 |
| mean | 38.5855 | 10.0818 | 1078.4437 | 87.3682 | 40.4403 |
| std | 13.6380 | 2.5716 | 7387.9574 | 403.1018 | 12.3469 |
| min | 17.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| 25% | 28.0000 | 9.0000 | 0.0000 | 0.0000 | 40.0000 |
| 50% | 37.0000 | 10.0000 | 0.0000 | 0.0000 | 40.0000 |
| 75% | 48.0000 | 12.0000 | 0.0000 | 0.0000 | 45.0000 |
| max | 90.0000 | 16.0000 | 99999.0000 | 4356.0000 | 99.0000 |

### 3-2. 상관계수 행렬

|  | age | education-num | capital-gain | capital-loss | hours-per-week |
|---|---|---|---|---|---|
| age | 1.0000 | 0.0360 | 0.0780 | 0.0580 | 0.0690 |
| education-num | 0.0360 | 1.0000 | 0.1230 | 0.0800 | 0.1480 |
| capital-gain | 0.0780 | 0.1230 | 1.0000 | -0.0320 | 0.0780 |
| capital-loss | 0.0580 | 0.0800 | -0.0320 | 1.0000 | 0.0540 |
| hours-per-week | 0.0690 | 0.1480 | 0.0780 | 0.0540 | 1.0000 |

### 3-3. t-test : income(<=50K vs >50K) 그룹의 hours-per-week 평균 차이

- `<=50K` 그룹 평균: 38.84시간 (n=24,698)
- `>50K` 그룹 평균: 45.47시간 (n=7,839)
- t-statistic = -45.0950, p-value = 0
- 해석: p-value < 0.05 → 두 income 그룹의 평균 근무시간 차이는 **통계적으로 유의미하다**.

## 4. ML Pipeline

- 모델: `RandomForestClassifier` (ColumnTransformer + Pipeline)
- feature: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- train/test: 26,029건 / 6,508건 (test_size=0.2, stratify=income)
- **Accuracy = 0.8657**
- **F1-score = 0.6749**
- 모델 저장 경로: `/Users/kwngus2/Desktop/SKALA_실습과제/team_gobp/SKALA_Project_AIOps/강주현/model.joblib`

## 5. 핵심 인사이트

- 학력별 고소득(>50K) 비율은 `Doctorate`이 74.1%로 가장 높고, `Preschool`이 0.0%로 가장 낮아 학력과 고소득 여부 사이에 뚜렷한 관계가 관찰된다.
- `>50K` 그룹의 평균 근무시간이 6.6시간 더 길며, 이 차이는 t-test 결과 통계적으로 유의미하다 (p=0).
- 수치형 변수 중 `education-num`와(과) `hours-per-week`의 상관계수가 0.148로 가장 크며, 나머지 변수쌍은 대체로 상관이 약해 서로 독립적인 정보를 담고 있는 것으로 보인다.

---
*이 문서는 `generate_report()` 함수에 의해 자동 생성되었습니다. 수동 편집 없이 스크립트 재실행만으로 갱신됩니다.*
