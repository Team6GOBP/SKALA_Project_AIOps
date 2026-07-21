"""시각화 모듈.

Seaborn 정적 차트(연령 분포)와 Plotly 인터랙티브 차트(income 그룹별 근로시간 비교)를 생성한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEABORN_CHART_PATH, PLOTLY_CHART_PATH

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import plotly.express as px
import seaborn as sns

_KOREAN_FONT_CANDIDATES = [
    "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR",
]


def set_korean_font() -> None:
    """설치된 한글 폰트 후보 중 하나를 자동 탐색해 matplotlib에 적용한다.

    설치된 한글 폰트가 없으면 에러 없이 그냥 진행한다.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _KOREAN_FONT_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break

    plt.rcParams["axes.unicode_minus"] = False


def plot_age_distribution(df: pd.DataFrame, save_path=SEABORN_CHART_PATH) -> None:
    """[Seaborn 정적 차트] age 히스토그램 + KDE를 그려서 저장한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임 (age 컬럼 포함).
    save_path : str | Path
        저장 경로. 기본값은 config.SEABORN_CHART_PATH (프로젝트 루트).
    """
    set_korean_font()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="age", kde=True, ax=ax)
    ax.set_title("나이(age) 분포")
    ax.set_xlabel("나이")
    ax.set_ylabel("인원수")

    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Seaborn] 연령 분포 차트 저장 완료 : {save_path}")


def plot_income_group_comparison(df: pd.DataFrame, save_path=PLOTLY_CHART_PATH) -> None:
    """[Plotly 인터랙티브 차트] income 그룹별 평균 hours_per_week 막대 차트를 그려서 저장한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임 (income, hours_per_week 컬럼 포함).
    save_path : str | Path
        저장 경로. 기본값은 config.PLOTLY_CHART_PATH (프로젝트 루트).
    """
    grouped = (
        df.assign(income=df["income"].astype(str).str.strip())
        .groupby("income", as_index=False)["hours_per_week"]
        .mean()
    )

    fig = px.bar(
        grouped,
        x="income",
        y="hours_per_week",
        title="소득 그룹별 평균 주당 근로시간 비교",
        labels={"income": "소득 그룹", "hours_per_week": "평균 근로시간(시간/주)"},
    )

    fig.write_html(save_path)
    print(f"[Plotly] 소득 그룹 비교 차트 저장 완료 : {save_path}")


def run_visualization(df: pd.DataFrame) -> None:
    """Seaborn 정적 차트 -> Plotly 인터랙티브 차트 순서로 시각화를 실행한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임.
    """
    plot_age_distribution(df)
    plot_income_group_comparison(df)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_prep import prepare_data

    cleaned_df, _ = prepare_data()
    run_visualization(cleaned_df)
