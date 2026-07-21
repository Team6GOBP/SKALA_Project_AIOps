# End2End 데이터 분석 리포트 — Adult Census Income

- 생성 일시: {{ generated }}
- 본 리포트는 `main.py` 실행 시 Jinja2 템플릿으로 **자동 생성**됩니다. (수동 편집 없음)

## 1. 데이터 준비 (Pandas vs Polars) — 팀 SPEC 1 적용

| 항목 | Pandas | Polars |
|---|---|---|
| shape (행, 열) | {{ data.shape_pandas }} | {{ data.shape_polars if data.shape_polars else "미설치로 생략" }} |
| 로딩 시간(초) | {{ data.load_sec_pandas }} | {{ data.load_sec_polars if data.load_sec_polars is not none else "-" }} |

- 원본 결측치 총 **{{ data.na_total_before }}건** — 컬럼별: {{ data.na_by_col_before }}
- **처리 정책 (SPEC 1)**: 삭제하지 않고 컬럼별 **최빈값(mode)** 으로 대체
  - 대체에 사용된 최빈값: {{ data.mode_values }}
- 중복 행 {{ data.dup_removed }}건 제거
- income 분포: {{ data.income_counts }}

## 2. 시각화 — 팀 SPEC 5 파일명

- Seaborn 정적 차트: `{{ viz.seaborn_png if viz.seaborn_png else "생성 실패" }}` (소득 그룹별 주당 근로시간 boxplot)
- Plotly 인터랙티브 차트: `{{ viz.plotly_html if viz.plotly_html else "생성 실패" }}` (학력별 평균 근로시간 bar)

## 3. 통계 분석 — 팀 SPEC 2·4

### 3-1. 기술통계 (평균·표준편차·분위수) — 수치형 5개 (fnlwgt 제외, SPEC 4)

{{ stats.describe_md }}

### 3-2. 상관행렬

{{ stats.corr_md }}

### 3-3. t-test — income 그룹별 주당 근로시간

- 인자 순서: `ttest_ind(<=50K, >50K, equal_var=False)` (SPEC 2)
- `<=50K` 평균 **{{ stats.mean_under }}시간** vs `>50K` 평균 **{{ stats.mean_over }}시간**
- t = **{{ stats.t_stat }}**, p-value = **{{ stats.p_value }}**
- **해석**: {{ stats.t_interpretation }}

## 4. ML Pipeline (소득 >50K 이진 분류) — 팀 SPEC 3·4

- 타깃 인코딩 (SPEC 3): `<=50K → 0`, `>50K → 1`
- 특성 구성 (SPEC 4): 수치형 **{{ ml.n_features_num }}개** + 범주형 **{{ ml.n_features_cat }}개** (fnlwgt 제외)
- 공통 설정 (SPEC 4): `test_size=0.2`, `random_state=42`
- 학습/평가 샘플: {{ ml.n_train }} / {{ ml.n_test }}
- **정확도(accuracy): {{ ml.accuracy }}**
- **F1-score: {{ ml.f1 }}**
- 저장 모델: `{{ ml.model_file }}` (joblib 재로딩 예측 일치: {{ ml.reload_ok }})

## 5. 결론 및 의견

- t-test 결과는 위 3-3 절의 해석 문장을 그대로 반영한다. income 그룹 간 근로시간 차이의 통계적 유의성 여부가 곧 이 프로젝트의 핵심 인사이트다.
- 전처리·모델을 하나의 Pipeline 으로 묶어 `model.joblib` 로 저장했으므로, 배포 환경에서도 동일한 전처리 + 예측이 보장된다 (데이터 누수 위험 없음: income 은 y 로만 사용, fnlwgt 는 X 에서 제외).
- 개선 아이디어: (1) RandomForest·XGBoost 등 모델 비교, (2) capital-gain 왜도 완화(log 변환), (3) 교차검증(cross_val_score) 도입.
