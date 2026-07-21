"""Adult Census Income End2End 분석 파이프라인 진입점.

목적
----
Pandas/Polars 로딩 비교 -> 결측치/중복 정제 -> 시각화(Seaborn+Plotly) ->
통계 분석(기술통계·상관계수·t-test) -> sklearn Pipeline 분류 모델 ->
report.md 자동 생성까지 이어지는 End2End 파이프라인을 한 번에 실행한다.

단계 요약
--------
1. src.data_prep.prepare_data() : 데이터 로딩 비교, EDA, 결측치/중복 정제
2. src.stats_analysis.run_statistical_analysis() : 기술통계, 상관계수, t-test
3. src.visualize.run_visualization() : Seaborn 정적 차트, Plotly 인터랙티브 차트 저장
4. src.ml_pipeline.train_and_evaluate() : Pipeline 학습/평가, model.joblib 저장
5. src.generate_report.generate_report() : report.md 자동 생성

실행 방법
--------
    python main.py

변경 내역
--------
- 2026-07-21 : 최초 구현 (PROJECT_BRIEF.md 팀 SPEC 기준).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_prep import prepare_data
from src.stats_analysis import run_statistical_analysis
from src.visualize import run_visualization
from src.ml_pipeline import train_and_evaluate
from src.generate_report import generate_report


def main() -> None:
    start = time.perf_counter()

    df_clean, rows_before = prepare_data()
    stats_result = run_statistical_analysis(df_clean)
    run_visualization(df_clean)
    ml_result = train_and_evaluate(df_clean)
    generate_report(
        rows_before=rows_before,
        rows_after=len(df_clean),
        stats_result=stats_result,
        ml_result=ml_result,
        missing_counts=df_clean.attrs.get("missing_counts"),
        ttest_result=stats_result.get("ttest"),
    )

    elapsed = time.perf_counter() - start
    print("=" * 60)
    print(f"전체 파이프라인 실행 완료 (총 소요시간 : {elapsed:.2f}초)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[오류] 파이프라인 실행 중 예외 발생 : {e}")
        sys.exit(1)
