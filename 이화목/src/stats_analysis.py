"""통계 분석 모듈.

기술통계, 상관계수, income 그룹 간 hours_per_week에 대한 Welch's t-test를 담당한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NUMERIC_COLS

import pandas as pd
from scipy import stats


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 컬럼(config.NUMERIC_COLS)의 기술통계를 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임.

    Returns
    -------
    pd.DataFrame
        `describe()` 결과 (평균·표준편차·분위수 포함).
    """
    print("=" * 60)
    print("기술통계 (describe)")
    print("=" * 60)
    result = df[NUMERIC_COLS].describe()
    print(result)
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 컬럼(config.NUMERIC_COLS) 간 상관계수 행렬을 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임.

    Returns
    -------
    pd.DataFrame
        반올림 3자리로 표시된 상관계수 행렬.
    """
    print("=" * 60)
    print("상관계수 행렬 (corr)")
    print("=" * 60)
    result = df[NUMERIC_COLS].corr().round(3)
    print(result)
    return result


def run_ttest_income_group(df: pd.DataFrame) -> dict:
    """income 그룹(<=50K vs >50K) 간 hours_per_week 평균 차이를 Welch's t-test로 검정한다.

    [팀 SPEC] 그룹=income, 비교 변수=hours_per_week,
    scipy.stats.ttest_ind(..., equal_var=False) 사용.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임 (income 컬럼 포함).

    Returns
    -------
    dict
        {"t_stat":.., "p_value":.., "mean_under":.., "mean_over":..} (report.md 생성에 사용).
    """
    print("=" * 60)
    print("t-test : income 그룹별 hours_per_week 비교 (Welch's t-test)")
    print("=" * 60)

    income = df["income"].astype(str).str.strip()
    group_under = df.loc[income == "<=50K", "hours_per_week"]
    group_over = df.loc[income == ">50K", "hours_per_week"]

    mean_under = group_under.mean()
    mean_over = group_over.mean()
    print(f"<=50K 그룹 평균 hours_per_week : {mean_under:.3f} (n={len(group_under)})")
    print(f">50K  그룹 평균 hours_per_week : {mean_over:.3f} (n={len(group_over)})")

    t_stat, p_value = stats.ttest_ind(group_under, group_over, equal_var=False)
    print(f"t-statistic : {t_stat:.4f}")
    print(f"p-value     : {p_value:.3e}")

    if p_value < 0.05:
        print("해석 : p-value < 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하다.")
    else:
        print("해석 : p-value >= 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하지 않다.")

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_under": mean_under,
        "mean_over": mean_over,
    }


def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """기술통계 -> 상관계수 -> t-test 순서로 전체 통계 분석을 실행한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임.

    Returns
    -------
    dict
        {"describe": pd.DataFrame, "corr": pd.DataFrame, "ttest": dict} (report.md 생성에 사용).
    """
    describe_result = descriptive_stats(df)
    corr_result = correlation_matrix(df)
    ttest_result = run_ttest_income_group(df)

    return {"describe": describe_result, "corr": corr_result, "ttest": ttest_result}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_prep import prepare_data

    cleaned_df, _ = prepare_data()
    run_statistical_analysis(cleaned_df)
