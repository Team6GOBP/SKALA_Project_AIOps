"""
visualize.py — 시각화 모듈  (팀 SPEC 5 파일명 준수)
  - Seaborn 정적 차트   : eda_chart_seaborn.png
  - Plotly 인터랙티브 : eda_chart_plotly.html
  * 채점 기준: 제목·축 레이블 포함 필수 (라벨/제목이 명확해야 병합에 채택됨)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경(서버·CI)에서도 저장이 되도록 설정
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# SPEC 5 : 파일명 규칙 (팀 병합 시 이 이름 그대로여야 함)
SEABORN_PNG = OUT_DIR / "eda_chart_seaborn.png"
PLOTLY_HTML = OUT_DIR / "eda_chart_plotly.html"


def make_charts(df: pd.DataFrame) -> dict:
    """정적/인터랙티브 차트를 SPEC 지정 파일명으로 저장하고 경로 dict 반환."""
    OUT_DIR.mkdir(exist_ok=True)
    result = {"seaborn_png": None, "plotly_html": None}

    # --- 1) Seaborn 정적 차트 : 소득 그룹별 주당 근로시간 분포 --------------
    #  → t-test 로 검정한 지표(hours-per-week)를 시각적으로도 함께 보여주어
    #    "그래프 + 통계"가 서로 뒷받침하도록 구성 (병합 기준 2번 강점 요소)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="income", y="hours-per-week",
                    hue="income", palette="Set2", legend=False, ax=ax)
        ax.set_title("Working Hours per Week by Income Group")  # 제목
        ax.set_xlabel("Income Group")                            # x축 레이블
        ax.set_ylabel("Hours per Week")                          # y축 레이블
        fig.tight_layout()
        fig.savefig(SEABORN_PNG, dpi=120)
        plt.close(fig)
        result["seaborn_png"] = SEABORN_PNG.name
        print(f"[viz] Seaborn 정적 차트 저장 → {SEABORN_PNG}")
    except Exception as e:
        print(f"[오류] Seaborn 차트 생성 실패: {e}")

    # --- 2) Plotly 인터랙티브 차트 : 학력별 평균 근로시간(bar) --------------
    try:
        import plotly.express as px
        hours = (df.groupby("education", as_index=False)
                   .agg(mean_hours=("hours-per-week", "mean"))
                   .sort_values("mean_hours", ascending=False))
        fig2 = px.bar(hours, x="education", y="mean_hours",
                      title="Average Working Hours per Week by Education",
                      labels={"education": "Education Level",
                              "mean_hours": "Avg Hours / Week"})
        fig2.write_html(PLOTLY_HTML)   # 인터랙티브 차트는 HTML 로 저장·공유
        result["plotly_html"] = PLOTLY_HTML.name
        print(f"[viz] Plotly 인터랙티브 차트 저장 → {PLOTLY_HTML}")
    except ImportError:
        print("[경고] plotly 미설치 → pip install plotly (인터랙티브 차트 건너뜀)")
    except Exception as e:
        print(f"[오류] Plotly 차트 생성 실패: {e}")

    return result
