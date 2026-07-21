"""리포트 자동 생성 모듈.

데이터 준비 / 통계 분석 / 시각화 / ML Pipeline 결과를 종합해 report.md를 생성한다.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    COLUMNS_WITH_MISSING,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    SEABORN_CHART_PATH,
    PLOTLY_CHART_PATH,
    REPORT_PATH,
)


def generate_report(
    rows_before: int,
    rows_after: int,
    stats_result: dict,
    ml_result: dict,
    missing_counts: dict = None,
    ttest_result: dict = None,
    report_path=REPORT_PATH,
) -> None:
    """report.md를 자동 생성한다.

    Parameters
    ----------
    rows_before : int
        정제 전 원본 행 수.
    rows_after : int
        정제 후 행 수.
    stats_result : dict
        {"describe": pd.DataFrame, "corr": pd.DataFrame} (stats_analysis 결과).
    ml_result : dict
        {"accuracy":.., "f1":.., "n_train":.., "n_test":..} (ml_pipeline 결과).
    missing_counts : dict, optional
        컬럼별 결측치(최빈값) 대체 건수.
    ttest_result : dict, optional
        {"t_stat":.., "p_value":.., "mean_under":.., "mean_over":..} t-test 결과.
    report_path : str | Path
        저장 경로. 기본값은 config.REPORT_PATH (프로젝트 루트).
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    missing_counts = missing_counts or {}
    ttest_result = ttest_result or {}

    missing_lines = "\n".join(
        f"- `{col}` : {count}건 -> 최빈값으로 대체" for col, count in missing_counts.items()
    ) or "- (결측치 대체 정보 없음)"

    if ttest_result:
        p_value = ttest_result.get("p_value")
        interpretation = (
            "p-value < 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하다."
            if p_value is not None and p_value < 0.05
            else "p-value >= 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하지 않다."
        )
        ttest_section = f"""
- t-statistic : `{ttest_result.get("t_stat"):.4f}`
- p-value : `{ttest_result.get("p_value"):.3e}`
- `<=50K` 그룹 평균 hours_per_week : `{ttest_result.get("mean_under"):.3f}`
- `>50K` 그룹 평균 hours_per_week : `{ttest_result.get("mean_over"):.3f}`
- 해석 : {interpretation}
"""
    else:
        ttest_section = "\n- (t-test 결과 없음)\n"

    content = f"""# Adult Census Income 분석 리포트

생성 시각 : {now_str}

---

## 1. 데이터 준비

- 원본 행 수 : {rows_before}행
- 정제 후 행 수 : {rows_after}행
- 정제 기준 : 결측 표기 `"?"` -> 각 컬럼의 최빈값(mode)으로 대체 (행 삭제 없음)

### 결측치 대체 내역

{missing_lines}

### 중복 제거

- 중복행 제거를 수행함 (`drop_duplicates()`)

---

## 2. 통계 분석

### 기술통계

{stats_result["describe"].to_markdown()}

### 상관계수

{stats_result["corr"].to_markdown()}

### t-test (그룹 = income, 비교 변수 = hours_per_week)
{ttest_section}
---

## 3. 시각화

- Seaborn 정적 차트(연령 분포) : [`{SEABORN_CHART_PATH.name}`]({SEABORN_CHART_PATH.name})
- Plotly 인터랙티브 차트(소득 그룹별 근로시간 비교) : [`{PLOTLY_CHART_PATH.name}`]({PLOTLY_CHART_PATH.name})

---

## 4. ML Pipeline

- 모델 구성 : `ColumnTransformer(StandardScaler + OneHotEncoder)` -> `RandomForestClassifier(n_estimators=200, max_depth=12)`
- 입력 변수 :
  - 수치형 (5개) : `{", ".join(NUMERIC_FEATURES)}`
  - 범주형 (8개) : `{", ".join(CATEGORICAL_FEATURES)}`
  - `fnlwgt`, `income`은 feature에서 제외 (데이터 누수 방지)
- 분할 설정 : `test_size={TEST_SIZE}`, `random_state={RANDOM_STATE}`
- train 건수 : {ml_result["n_train"]}건
- test 건수 : {ml_result["n_test"]}건
- **accuracy** : `{ml_result["accuracy"]:.4f}`
- **F1-score** : `{ml_result["f1"]:.4f}`

---

## 5. 본인 의견 / 개선 사항

> (이 섹션은 자동 생성되지 않습니다. 직접 작성해 주세요.)

-
-
-
"""

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"report.md 생성 완료 : {report_path}")
    except OSError as e:
        print(f"[오류] report.md 저장 실패 : {e}")
        raise
