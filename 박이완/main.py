# -*- coding: utf-8 -*-
"""
===============================================================================
[Day 2] 종합 실습 — End2End 데이터 분석 프로젝트 (Adult Census Income)
                     — 팀 공통 SPEC(spec.md) 준수 버전
-------------------------------------------------------------------------------
프로그램 전체 설명 (실행: python main.py)
  ① 데이터 준비 (SPEC 1)  : Pandas·Polars 로딩 비교, 결측치는 최빈값(mode)
                             으로 대체, 중복 제거                (src/data_prep.py)
  ② 시각화     (SPEC 5)  : Seaborn 1개(eda_chart_seaborn.png)
                             + Plotly 1개(eda_chart_plotly.html) (src/visualize.py)
  ③ 통계 분석  (SPEC 2·4): 기술통계·상관계수(수치 5개, fnlwgt 제외)
                             + t-test(<=50K, >50K, equal_var=False)
                                                                 (src/stats_analysis.py)
  ④ ML Pipeline (SPEC 3·4·5): income {'<=50K':0,'>50K':1} 이진 분류,
                             수치 5개+범주 8개, test_size=0.2/random_state=42,
                             accuracy·F1 출력, model.joblib 저장 (src/ml_pipeline.py)
  ⑤ 자동화     (SPEC 5)  : 결과를 output/report.md 로 자동 생성  (src/report.py)

구조화 원칙 (강의 13장 반영)
  - 검증된 함수는 src/ 모듈로 분리, main.py 는 오케스트레이션만 담당
  - data/ 는 .gitignore 처리(원본은 URL 로 재현), requirements.txt + README 제공
===============================================================================
"""

import sys

from src.data_prep import prepare
from src.visualize import make_charts
from src.stats_analysis import run_stats
from src.ml_pipeline import train_and_save
from src.report import write_report


def main() -> None:
    # ① 데이터 준비 — 실패 시 이후 단계가 무의미하므로 즉시 종료
    try:
        df, data_summary = prepare()
    except RuntimeError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    # ② 시각화 (모듈 내부에서 차트별 예외 처리)
    viz = make_charts(df)

    # ③ 통계 분석
    try:
        stats_summary = run_stats(df)
    except (KeyError, ValueError) as e:
        print(f"[오류] 통계 분석 실패: {e}")
        sys.exit(1)

    # ④ ML Pipeline
    try:
        ml_summary = train_and_save(df)
    except Exception as e:
        print(f"[오류] ML 파이프라인 실패: {e}")
        sys.exit(1)

    # ⑤ report.md 자동 생성
    write_report(data_summary, viz, stats_summary, ml_summary)
    print("\n[완료] End2End 분석 종료 — output/ 폴더에서 결과를 확인하세요.")


if __name__ == "__main__":
    main()
