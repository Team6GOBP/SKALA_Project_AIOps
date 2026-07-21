"""
stats_analysis.py — 통계 분석 모듈  (팀 SPEC 2·4 준수)

핵심 정책 (팀 SPEC 반영)
  - SPEC 4 : 상관계수 대상 수치형 5개 고정 (age, education-num,
              capital-gain, capital-loss, hours-per-week) — fnlwgt 제외
  - SPEC 2 : t-test 는 income 그룹의 hours-per-week 평균 차이를 검정
              scipy.stats.ttest_ind(group_50k_under, group_50k_over,
                                    equal_var=False)  ← 인자 순서 통일
              출력에 t-stat / p-value / p<0.05 해석 문장 필수
"""

import pandas as pd
from scipy import stats

# SPEC 4 : 상관계수·ML 수치 feature 로 사용할 5개 컬럼 (fnlwgt 제외)
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss",
            "hours-per-week"]


def run_stats(df: pd.DataFrame) -> dict:
    """기술통계·상관계수·t-test 를 수행하고 리포트용 요약 dict 를 반환한다."""
    print("\n" + "=" * 60)
    print("[stats] 통계 분석")
    print("=" * 60)

    # --- 1) 기술통계 (평균·표준편차·분위수) ---------------------------------
    desc = df[NUM_COLS].describe().round(2)
    print("■ 기술통계 (describe)")
    print(desc.to_string())

    # --- 2) 상관계수 (SPEC 4 : 5개 수치형만, fnlwgt 제외) ------------------
    corr = df[NUM_COLS].corr().round(3)
    print("\n■ 수치형 변수 상관행렬")
    print(corr.to_string())

    # --- 3) t-test (SPEC 2) -----------------------------------------------
    # SPEC 2 : ttest_ind(group_50k_under, group_50k_over, equal_var=False)
    #    인자 순서를 SPEC 에 맞춰 팀원 간 t 통계량 부호까지 동일하게 통일
    group_50k_under = df.loc[df["income"] == "<=50K", "hours-per-week"]
    group_50k_over  = df.loc[df["income"] == ">50K",  "hours-per-week"]
    t, p = stats.ttest_ind(group_50k_under, group_50k_over, equal_var=False)

    print("\n■ t-test: income 그룹별 주당 근로시간 평균 차이")
    print(f"  <=50K 평균={group_50k_under.mean():.2f}시간 "
          f"/ >50K 평균={group_50k_over.mean():.2f}시간")
    print(f"  t통계량={t:.3f}, p-value={p:.4g}")

    # SPEC 2 : p<0.05 해석 문장 필수
    if p < 0.05:
        interp = ("p < 0.05 이므로 유의미하다 — 두 소득 그룹의 주당 근로시간 "
                  "평균 차이는 통계적으로 유의미하다 (귀무가설 기각).")
    else:
        interp = ("p ≥ 0.05 이므로 유의미하지 않다 — 두 소득 그룹의 주당 근로시간 "
                  "평균 차이가 통계적으로 유의미하다고 볼 수 없다.")
    print(f"  → 해석: {interp}")

    return {
        "describe_md": desc.to_markdown(),
        "corr_md": corr.to_markdown(),
        "t_stat": round(float(t), 3),
        "p_value": f"{p:.4g}",
        "mean_under": round(float(group_50k_under.mean()), 2),
        "mean_over": round(float(group_50k_over.mean()), 2),
        "t_interpretation": interp,
    }
