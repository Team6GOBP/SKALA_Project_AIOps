import matplotlib
matplotlib.use("Agg")

import time

import pandas as pd
import polars as pl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "Adult Census Income.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
MISSING_COLS = ["workclass", "occupation", "native_country"]


def print_header(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# 1. 데이터 준비: 로드 + 중복 처리 + 결측치 처리 (mode 대체) - pandas 버전
def load_and_clean_pandas(path: str) -> tuple[pd.DataFrame, dict, float]:
    start = time.perf_counter()

    # "?"는 공백 없이 정확히 일치해야 결측으로 인식됨 (" ?"로 쓰면 결측치가 하나도 안 잡히는 버그 주의)
    df = pd.read_csv(path, na_values="?")
    rows_before = len(df)

    # 완전히 동일한 행(중복 레코드)은 삭제 - 결측치와 달리 정보 손실이 없음
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)

    # 결측 컬럼 3개는 행을 삭제하지 않고 최빈값(mode)으로 대체 -> 팀원 간 데이터 규모를 동일하게 유지
    missing_before = df[MISSING_COLS].isna().sum().to_dict()
    fill_values = {}
    for col in MISSING_COLS:
        mode_value = df[col].mode(dropna=True)[0]
        fill_values[col] = mode_value
        df[col] = df[col].fillna(mode_value)
    missing_after = df[MISSING_COLS].isna().sum().to_dict()

    # 원본 값에 공백이 섞여 있을 수 있어 strip 후 타깃을 0/1로 인코딩
    df["income"] = df["income"].str.strip()
    df["income_encoded"] = df["income"].map({"<=50K": 0, ">50K": 1})

    elapsed = time.perf_counter() - start

    report = {
        "rows_before": rows_before,
        "duplicates_removed": duplicates_removed,
        "rows_after_dedup": rows_after_dedup,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "fill_values": fill_values,
        "income_counts": df["income"].value_counts().to_dict(),
    }
    return df, report, elapsed


# 1-2. 데이터 준비: 로드 + 중복 처리 + 결측치 처리 (mode 대체) - polars 버전 (성능/결과 비교용)
def load_and_clean_polars(path: str) -> tuple[pl.DataFrame, dict, float]:
    start = time.perf_counter()

    # pandas 버전과 동일한 로직을 polars API로 재구현 (결과/성능 비교용)
    df = pl.read_csv(path, null_values="?")
    rows_before = df.height

    duplicates_removed = rows_before - df.unique().height
    df = df.unique(keep="first", maintain_order=True)
    rows_after_dedup = df.height

    missing_before = {col: df[col].null_count() for col in MISSING_COLS}
    fill_values = {}
    for col in MISSING_COLS:
        mode_value = df[col].mode()[0]
        fill_values[col] = mode_value
        df = df.with_columns(pl.col(col).fill_null(mode_value))
    missing_after = {col: df[col].null_count() for col in MISSING_COLS}

    df = df.with_columns(pl.col("income").str.strip_chars())

    elapsed = time.perf_counter() - start

    income_vc = df["income"].value_counts()
    income_counts = dict(zip(income_vc["income"], income_vc["count"]))
    report = {
        "rows_before": rows_before,
        "duplicates_removed": duplicates_removed,
        "rows_after_dedup": rows_after_dedup,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "fill_values": fill_values,
        "income_counts": income_counts,
    }
    return df, report, elapsed


# 1-3. pandas vs polars 데이터 준비 결과 비교 + 기본 EDA
def prepare_data(path: str) -> tuple[pd.DataFrame, dict, dict]:
    pdf, pandas_report, pandas_time = load_and_clean_pandas(path)
    pldf, polars_report, polars_time = load_and_clean_polars(path)

    # 두 라이브러리가 동일한 전처리 로직으로 같은 결과를 내는지 검증
    rows_match = pandas_report["rows_after_dedup"] == polars_report["rows_after_dedup"]
    income_counts_match = {str(k): v for k, v in pandas_report["income_counts"].items()} == {
        str(k): v for k, v in polars_report["income_counts"].items()
    }

    comparison = {
        "pandas": {**pandas_report, "elapsed_sec": pandas_time},
        "polars": {**polars_report, "elapsed_sec": polars_time},
        "rows_match": rows_match,
        "income_counts_match": income_counts_match,
    }

    print_header("1. 데이터 준비 (pandas vs polars 비교)")
    compare_rows = [
        ["pandas", pandas_report["rows_before"], pandas_report["duplicates_removed"],
         pandas_report["rows_after_dedup"], f"{pandas_time:.4f}"],
        ["polars", polars_report["rows_before"], polars_report["duplicates_removed"],
         polars_report["rows_after_dedup"], f"{polars_time:.4f}"],
    ]
    print(tabulate(compare_rows, headers=["라이브러리", "원본 행수", "중복 제거", "중복 제거 후", "소요시간(s)"], tablefmt="github"))
    print(f"\n결과 행 수 일치: {rows_match}  |  income 분포 일치: {income_counts_match}")

    print("\n[기본 EDA: 수치형 컬럼 요약 통계 (pandas 기준)]")
    desc = pdf[NUMERIC_COLS].describe().round(2)
    print(tabulate(desc, headers="keys", tablefmt="github"))

    print("\n[income 분포]")
    income_dist = pdf["income"].value_counts(normalize=True).round(4).reset_index()
    income_dist.columns = ["income", "비율"]
    print(tabulate(income_dist, headers="keys", tablefmt="github", showindex=False))

    print_header("2. 결측치 처리 (중복 제거 후, 최빈값 대체)")
    missing_rows = [
        [col, pandas_report["missing_before"][col], pandas_report["fill_values"][col], pandas_report["missing_after"][col]]
        for col in MISSING_COLS
    ]
    print(tabulate(missing_rows, headers=["컬럼", "처리 전 결측", "최빈값 대체", "처리 후 결측"], tablefmt="github"))

    # 이후 통계분석/ML 단계는 sklearn과의 호환성 때문에 pandas 결과(pdf)만 사용
    return pdf, pandas_report, comparison


# 2. 소득 그룹별 기초 통계: age / hours_per_week / education_num 평균·중앙값 비교 + capital_gain 쏠림 확인
def group_summary(df: pd.DataFrame) -> dict:
    cols = ["age", "hours_per_week", "education_num"]
    summary = df.groupby("income")[cols].agg(["mean", "median"]).round(2).reindex(["<=50K", ">50K"])

    print_header("3. 소득 그룹별 기초 통계")
    flat_summary = summary.copy()
    flat_summary.columns = [f"{a}_{b}" for a, b in flat_summary.columns]
    print(tabulate(flat_summary.reset_index(), headers="keys", tablefmt="github", showindex=False))

    # capital_gain은 대부분 0이고 소수만 큰 값을 가지는 분포라 평균과 중앙값 격차로 쏠림을 확인
    cg_by_income = df.groupby("income")["capital_gain"].agg(["mean", "median"]).round(2).reindex(["<=50K", ">50K"])
    overall_mean = df["capital_gain"].mean()
    overall_median = df["capital_gain"].median()
    skew_insight = (
        f"capital_gain 전체 평균은 {overall_mean:.2f}달러인데 중앙값은 {overall_median:.2f}달러입니다. "
        f"평균이 중앙값보다 훨씬 크다는 것은 대다수(중앙값 기준 절반 이상)는 capital_gain이 0이고, "
        f"소수의 인원이 매우 큰 자본 이득을 얻으면서 전체 평균을 끌어올리는 극단적인 쏠림(우측 꼬리 분포)이 "
        f"존재한다는 증거입니다."
    )

    print("\n[capital_gain 평균 vs 중앙값 (쏠림 확인)]")
    print(tabulate(cg_by_income.reset_index(), headers=["income", "mean", "median"], tablefmt="github", showindex=False))
    print(f"\n전체 평균: {overall_mean:.2f}  |  전체 중앙값: {overall_median:.2f}")
    print(f"-> {skew_insight}")

    return {
        "summary": summary,
        "capital_gain_by_income": cg_by_income,
        "capital_gain_overall_mean": overall_mean,
        "capital_gain_overall_median": overall_median,
        "skew_insight": skew_insight,
    }


# 3. t-test: income x hours_per_week, income x education_num (Welch's t-test)
def run_ttests(df: pd.DataFrame) -> dict:
    def welch_ttest(col: str, label: str) -> dict:
        group_under = df.loc[df["income"] == "<=50K", col]
        group_over = df.loc[df["income"] == ">50K", col]

        # equal_var=False -> Welch's t-test: 두 그룹의 분산이 같다고 가정하지 않음
        t_stat, p_value = stats.ttest_ind(group_under, group_over, equal_var=False)
        is_significant = p_value < 0.05
        # 표본 크기가 커서 p-value가 float64 최소 표현범위 아래로 언더플로우(0.0)될 수 있음
        p_display = "< 1e-300" if p_value == 0 else f"{p_value:.4g}"
        interpretation = (
            f"p-value({p_display}) < 0.05 이므로 귀무가설(H0)을 기각한다: "
            f"두 소득 그룹 간 {label} 평균 차이는 통계적으로 유의미하다."
            if is_significant else
            f"p-value({p_display}) >= 0.05 이므로 귀무가설(H0)을 기각할 수 없다: "
            f"두 소득 그룹 간 {label} 평균 차이는 통계적으로 유의미하지 않다."
        )
        return {
            "column": col,
            "label": label,
            "mean_under": group_under.mean(),
            "mean_over": group_over.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "p_display": p_display,
            "is_significant": bool(is_significant),
            "interpretation": interpretation,
        }

    hours_result = welch_ttest("hours_per_week", "주당 근무시간(hours_per_week)")
    edu_result = welch_ttest("education_num", "교육수준(education_num)")

    print_header("4. t-test (Welch's t-test)")
    ttest_rows = [
        ["hours_per_week", "<=50K 평균 = >50K 평균", f"{hours_result['mean_under']:.2f}", f"{hours_result['mean_over']:.2f}",
         f"{hours_result['t_stat']:.2f}", hours_result["p_display"], "유의미" if hours_result["is_significant"] else "유의미하지 않음"],
        ["education_num", "<=50K 평균 = >50K 평균", f"{edu_result['mean_under']:.2f}", f"{edu_result['mean_over']:.2f}",
         f"{edu_result['t_stat']:.2f}", edu_result["p_display"], "유의미" if edu_result["is_significant"] else "유의미하지 않음"],
    ]
    print(tabulate(
        ttest_rows,
        headers=["변수", "H0 (귀무가설)", "<=50K 평균", ">50K 평균", "t-statistic", "p-value", "판정"],
        tablefmt="github",
    ))
    print(f"\n- hours_per_week: {hours_result['interpretation']}")
    print(f"- education_num: {edu_result['interpretation']}")

    return {"hours_per_week": hours_result, "education_num": edu_result}


# 4. 상관계수 (수치형 5개, fnlwgt 제외) - 텍스트 랭킹 + 최고 상관 쌍
def compute_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    corr = df[NUMERIC_COLS].corr()

    # 대각선(자기 자신과의 상관계수=1)과 중복 쌍을 제외하고 변수 쌍별 상관계수를 절댓값 기준으로 정렬
    pairs = []
    for i, col_a in enumerate(NUMERIC_COLS):
        for col_b in NUMERIC_COLS[i + 1:]:
            pairs.append((col_a, col_b, corr.loc[col_a, col_b]))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    top_pair = pairs[0]

    print_header("5. 상관계수 (수치형 5개 컬럼)")
    print(tabulate(corr.round(4), headers="keys", tablefmt="github"))

    print("\n[상관계수 순위 (절댓값 기준 내림차순)]")
    rank_rows = [[col_a, col_b, f"{value:.4f}"] for col_a, col_b, value in pairs]
    print(tabulate(rank_rows, headers=["변수 A", "변수 B", "상관계수"], tablefmt="github"))
    print(f"\n-> 가장 연관성이 높은 변수 쌍: {top_pair[0]} - {top_pair[1]} (r={top_pair[2]:.4f})")

    return corr, {"pairs": pairs, "top_pair": top_pair}


# 4. EDA 차트 저장 (수치형 5개 컬럼 상관관계 히트맵)
def make_charts(corr: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("상관관계 히트맵 (수치형 변수)")
    plt.xlabel("변수")
    plt.ylabel("변수")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("eda_chart_seaborn.png", dpi=150)
    plt.close()

    fig = px.imshow(
        corr,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="상관관계 히트맵 (수치형 변수)",
    )
    fig.update_layout(xaxis_title="변수", yaxis_title="변수")
    fig.write_html("eda_chart_plotly.html")

    print_header("6. EDA 차트 저장 완료")
    print("- eda_chart_seaborn.png (정적 히트맵)")
    print("- eda_chart_plotly.html (인터랙티브 히트맵)")


# 5. ML Pipeline (수치형 5개 + 범주형 전체, fnlwgt/income 제외)
def train_model(df: pd.DataFrame) -> dict:
    # fnlwgt(인구총조사 가중치)와 income(타깃)은 feature에서 제외 -> 데이터 누수 방지
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df["income_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 수치형은 그대로 통과, 범주형은 원-핫 인코딩 (미학습 카테고리는 무시)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    # 전처리기 + 분류기를 하나의 Pipeline으로 묶어 fit/predict 시 전처리가 함께 적용되도록 구성
    # n_estimators/max_depth를 제한한 이유: 트리 수를 늘릴수록 model.joblib 용량이 커짐 (200그루 기준 149MB -> 100그루+depth 12로 7.8MB)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE, n_estimators=100, max_depth=12,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print_header("7. ML Pipeline 결과")
    print(tabulate(
        [["Train / Test 크기", f"{len(X_train)} / {len(X_test)}"],
         ["test_size / random_state", f"{TEST_SIZE} / {RANDOM_STATE}"],
         ["Accuracy", f"{accuracy:.4f}"],
         ["F1-score", f"{f1:.4f}"]],
        headers=["항목", "값"],
        tablefmt="github",
    ))

    joblib.dump(pipeline, "model.joblib")

    return {
        "feature_cols": feature_cols,
        "accuracy": accuracy,
        "f1_score": f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# 7. report.md 자동 생성
def write_report(
    comparison: dict, pandas_report: dict, group_stats: dict,
    ttest_results: dict, corr: pd.DataFrame, corr_summary: dict, model_result: dict,
) -> None:
    lines = []
    lines.append("# Day2 종합실습 리포트\n")

    lines.append("## 1. 데이터 준비 (pandas vs polars 비교)\n")
    lines.append("| 라이브러리 | 원본 행 수 | 중복 제거 건수 | 중복 제거 후 행 수 | 소요시간(s) |")
    lines.append("|---|---|---|---|---|")
    for lib in ("pandas", "polars"):
        r = comparison[lib]
        lines.append(
            f"| {lib} | {r['rows_before']} | {r['duplicates_removed']} | "
            f"{r['rows_after_dedup']} | {r['elapsed_sec']:.4f} |"
        )
    lines.append("")
    lines.append(f"- 두 라이브러리의 중복 제거 후 행 수 일치: {comparison['rows_match']}")
    lines.append(f"- 두 라이브러리의 income 분포 일치: {comparison['income_counts_match']}")
    lines.append("")

    lines.append("## 2. 결측치 처리 (중복 제거 후, mode 대체)\n")
    lines.append("| 컬럼 | 결측 개수(처리 전) | 최빈값 대체 | 결측 개수(처리 후) |")
    lines.append("|---|---|---|---|")
    for col in MISSING_COLS:
        lines.append(
            f"| {col} | {pandas_report['missing_before'][col]} | "
            f"{pandas_report['fill_values'][col]} | {pandas_report['missing_after'][col]} |"
        )
    lines.append("")

    lines.append("## 3. 소득 그룹별 기초 통계\n")
    flat_summary = group_stats["summary"].copy()
    flat_summary.columns = [f"{a}_{b}" for a, b in flat_summary.columns]
    lines.append(flat_summary.reset_index().to_markdown(index=False))
    lines.append("")
    lines.append("**capital_gain 평균 vs 중앙값 (쏠림 확인)**\n")
    lines.append(group_stats["capital_gain_by_income"].reset_index().to_markdown(index=False))
    lines.append("")
    lines.append(
        f"- 전체 평균: {group_stats['capital_gain_overall_mean']:.2f}, "
        f"전체 중앙값: {group_stats['capital_gain_overall_median']:.2f}"
    )
    lines.append(f"- 해석: {group_stats['skew_insight']}")
    lines.append("")

    lines.append("## 4. t-test\n")
    for key, title in [("hours_per_week", "주당 근무시간(hours_per_week)"), ("education_num", "교육수준(education_num)")]:
        r = ttest_results[key]
        lines.append(f"### income x {title}\n")
        lines.append(f"- H0: 두 소득 그룹의 평균 {title}는 같다 / H1: 다르다")
        lines.append(f"- `<=50K` 평균: {r['mean_under']:.4f}")
        lines.append(f"- `>50K` 평균: {r['mean_over']:.4f}")
        lines.append(f"- t-statistic: {r['t_stat']:.4f}")
        lines.append(f"- p-value: {r['p_display']}")
        lines.append(f"- 해석: {r['interpretation']}")
        lines.append("")

    lines.append("## 5. 상관계수 (수치형 5개 컬럼, fnlwgt 제외)\n")
    lines.append(corr.round(4).to_markdown())
    lines.append("")
    lines.append("**상관계수 순위 (절댓값 기준 내림차순)**\n")
    for col_a, col_b, value in corr_summary["pairs"]:
        lines.append(f"- {col_a} - {col_b}: {value:.4f}")
    top_a, top_b, top_v = corr_summary["top_pair"]
    lines.append(f"\n- 가장 연관성이 높은 변수 쌍: **{top_a} - {top_b}** (r={top_v:.4f})")
    lines.append("")

    lines.append("## 6. EDA 차트\n")
    lines.append("- `eda_chart_seaborn.png`: 수치형 5개 컬럼 상관관계 히트맵 (static heatmap)")
    lines.append("- `eda_chart_plotly.html`: 수치형 5개 컬럼 상관관계 히트맵 (interactive heatmap, hover로 정확한 값 확인)")
    lines.append("")

    lines.append("## 7. ML Pipeline 결과\n")
    lines.append(f"- Feature 컬럼: {', '.join(model_result['feature_cols'])}")
    lines.append(f"- Train / Test 크기: {model_result['n_train']} / {model_result['n_test']} "
                  f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    lines.append(f"- Accuracy: {model_result['accuracy']:.4f}")
    lines.append(f"- F1-score: {model_result['f1_score']:.4f}")
    lines.append(f"- 저장된 모델: `model.joblib`")
    lines.append("")

    with open("report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    df, pandas_report, comparison = prepare_data(DATA_PATH)
    group_stats = group_summary(df)
    ttest_results = run_ttests(df)
    corr, corr_summary = compute_correlation(df)
    make_charts(corr)
    model_result = train_model(df)
    write_report(comparison, pandas_report, group_stats, ttest_results, corr, corr_summary, model_result)

    print_header("완료")
    print("생성된 산출물: eda_chart_seaborn.png, eda_chart_plotly.html, model.joblib, report.md")


if __name__ == "__main__":
    main()
