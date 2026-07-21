"""
[Day2 종합실습] 광주_2반_강주현
Adult Census Income 데이터셋 End2End 분석.

진행 상황:
- Step 1 (완료) : Pandas·Polars 로딩 비교 (shape·dtype·결측치·중복행)
- Step 2~3 (완료) : 결측치 최빈값 대체, 중복행 제거, 기본 EDA
- Step 4 (완료) : Seaborn(상관관계·그룹비교)·Plotly(그룹비교·분포) 시각화 4종
- Step 5 (완료) : 통계 분석 (기술통계·상관계수·income 그룹 t-test)
- Step 6 (완료) : sklearn Pipeline (RandomForestClassifier, accuracy·F1, joblib 저장)
- Step 7 (완료) : report.md 자동 생성

SPEC.md 규칙 준수: 결측치는 최빈값 대체(삭제 X), na_values="?"(공백 없이),
상관계수/모델 feature는 fnlwgt 제외, t-test는 income×hours-per-week.
"""
import time
from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]

MISSING_COLS = ["workclass", "occupation", "native-country"]

# SPEC.md 4번 규칙: 상관계수·모델 feature는 fnlwgt 제외
CORR_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

OUT_DIR = Path(__file__).parent
SEABORN_CHART_PATH = OUT_DIR / "eda_chart_seaborn.png"
PLOTLY_CHART_PATH = OUT_DIR / "eda_chart_plotly.html"
# 아래 둘은 SPEC.md 필수 산출물이 아닌, 다양성을 위한 추가(선택) 시각화
SEABORN_EXTRA_CHART_PATH = OUT_DIR / "eda_chart_seaborn_groupcompare.png"
PLOTLY_EXTRA_CHART_PATH = OUT_DIR / "eda_chart_plotly_distribution.html"
MODEL_PATH = OUT_DIR / "model.joblib"
REPORT_PATH = OUT_DIR / "report.md"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 공통 유틸 - DataFrame -> 마크다운 표 변환 (tabulate 패키지 없이 직접 구현)
# ---------------------------------------------------------------------------
def _df_to_markdown(df: pd.DataFrame) -> str:
    """pandas.DataFrame.to_markdown()은 tabulate 패키지가 별도로 설치돼 있어야
    동작하는데, 이게 없는 환경에서 report.md 생성이 통째로 실패하는 걸 막기 위해
    외부 패키지 없이 직접 마크다운 표 문자열을 만든다."""
    headers = [df.index.name or ""] + [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


# ---------------------------------------------------------------------------
# 공통 유틸 - 필수 컬럼 존재 여부 검증
# ---------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """df에 columns가 전부 있는지 검사하고, 없으면 KeyError로 어떤 컬럼이 없는지 알려준다."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"[{context}] 필수 컬럼이 없습니다: {missing}")


# ---------------------------------------------------------------------------
# 공통 유틸 - 한글 폰트 설정 (OS별 자동 탐색)
# ---------------------------------------------------------------------------
def set_korean_font() -> None:
    """matplotlib/seaborn 차트에서 한글이 깨지지 않도록 OS별 기본 한글 폰트를 지정한다.
    (macOS: AppleGothic, Windows: Malgun Gothic, Linux: NanumGothic)
    지정 가능한 폰트가 하나도 없으면 경고만 남기고 기본 폰트로 진행한다."""
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 폰트를 찾지 못해 기본 폰트로 진행합니다. (라벨이 깨질 수 있음)")
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


# ---------------------------------------------------------------------------
# Step 1. 데이터 로딩 (Pandas vs Polars 비교)
# ---------------------------------------------------------------------------
def load_with_pandas(url: str) -> pd.DataFrame:
    """Pandas로 로딩. 결측치 표기(' ?')를 NaN으로 변환한다.

    주의: skipinitialspace=True는 필드 값의 앞 공백을 먼저 제거한 뒤
    na_values와 비교하므로, na_values는 공백 없이 "?"로 지정해야 한다.
    na_values=" ?"로 쓰면(자주 퍼진 스니펫의 실수) 매칭이 안 돼 결측치가
    하나도 안 잡히는 조용한 버그가 생긴다.
    """
    return pd.read_csv(
        url, header=None, names=COLS, na_values="?", skipinitialspace=True
    )


def load_with_polars(url: str) -> pl.DataFrame:
    """Polars로 로딩.

    주의 1: UCI 원본은 콤마 뒤에 공백이 섞여 있는데(예: "39, State-gov, 77516"),
    Polars에는 Pandas의 skipinitialspace 같은 옵션이 없다. 공백이 낀 채로 숫자
    컬럼을 인식하려 하면 파싱에 실패해 해당 컬럼 전체가 String으로 떨어진다.
    그래서 일단 전부 문자열(infer_schema_length=0)로 읽은 뒤, 모든 컬럼의 공백을
    strip_chars()로 제거하고, 수치형 컬럼만 명시적으로 cast한다.

    주의 2: UCI 원본 파일 끝에 빈 줄이 하나 있다. Pandas는 이를 자동으로 건너뛰지만
    Polars는 그대로 값이 전부 null인 한 행으로 읽어버려 행 수가 +1이 된다.
    그래서 모든 컬럼이 비어 있는 행을 마지막에 걸러낸다.
    """
    df = pl.read_csv(
        url, has_header=False, new_columns=COLS, infer_schema_length=0
    )
    df = df.with_columns([pl.col(c).str.strip_chars() for c in COLS])

    # 빈 줄(trailing blank line)로 생긴, 전 컬럼이 비어 있는 행 제거
    df = df.filter(
        ~pl.all_horizontal([pl.col(c).is_null() | (pl.col(c) == "") for c in COLS])
    )

    # 결측치 표기 "?" -> null (Pandas의 na_values="?"와 동일한 처리)
    df = df.with_columns([
        pl.when(pl.col(c) == "?").then(None).otherwise(pl.col(c)).alias(c)
        for c in COLS
    ])

    # 수치형 컬럼만 Int64로 형변환
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in NUMERIC_COLS])

    return df


def compare_loading(url: str) -> tuple[pd.DataFrame, dict]:
    """Pandas·Polars 로딩 시간과 결과(shape, dtype)를 비교해 출력하고,
    (재사용할 Pandas DataFrame, report.md용 통계 dict)를 반환한다."""
    t0 = time.perf_counter()
    df_pd = load_with_pandas(url)
    t_pandas = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_pl = load_with_polars(url)
    t_polars = time.perf_counter() - t0

    print("=== 로딩 시간 비교 ===")
    print(f"Pandas : {t_pandas:.4f}초 (shape={df_pd.shape})")
    print(f"Polars : {t_polars:.4f}초 (shape={df_pl.shape})")

    print("\n=== dtype 비교 ===")
    print("[Pandas]")
    print(df_pd.dtypes)
    print("\n[Polars]")
    print(df_pl.schema)

    missing_pandas = df_pd.isnull().sum()
    missing_pandas = missing_pandas[missing_pandas > 0]
    print("\n=== 결측치 개수 ===")
    print("[Pandas]")
    print(missing_pandas)
    print("[Polars]")
    polars_nulls = df_pl.null_count()
    print(polars_nulls.transpose(include_header=True, header_name="column", column_names=["null_count"]).filter(pl.col("null_count") > 0))

    # Polars의 is_duplicated()는 중복된 행 전부(원본 포함)를 True로 표시하는 반면,
    # Pandas의 duplicated()는 최초 등장 행은 False로 두고 "추가로 나온" 행만 센다.
    # 두 정의를 맞추기 위해 Polars는 (전체 행수 - 고유 행수)로 계산한다.
    dup_pandas = int(df_pd.duplicated().sum())
    dup_polars = len(df_pl) - df_pl.unique().height
    print("\n=== 중복행 개수 ===")
    print(f"Pandas : {dup_pandas}건")
    print(f"Polars : {dup_polars}건")

    stats = {
        "t_pandas": t_pandas,
        "t_polars": t_polars,
        "shape_pandas": df_pd.shape,
        "shape_polars": df_pl.shape,
        "missing_pandas": missing_pandas.to_dict(),
        "dup_pandas": dup_pandas,
        "dup_polars": dup_polars,
    }
    return df_pd, stats


# ---------------------------------------------------------------------------
# Step 2~3. 결측치·중복 처리 + 기본 EDA (SPEC.md 규칙 적용)
# ---------------------------------------------------------------------------
def handle_missing_and_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """SPEC.md 규칙: 결측치는 최빈값으로 대체(삭제 X), 중복행은 제거한다.
    (정제된 df, report.md용 통계 dict)를 반환한다."""
    df_clean = df.copy()

    before_missing = df_clean[MISSING_COLS].isnull().sum()
    mode_values = {}
    for col in MISSING_COLS:
        if col not in df_clean.columns:
            raise KeyError(f"'{col}' 컬럼이 존재하지 않습니다.")
        mode_value = df_clean[col].mode(dropna=True)
        if mode_value.empty:
            print(f"[경고] '{col}' 컬럼이 전부 결측치라 최빈값 대체를 건너뜁니다.")
            continue
        mode_values[col] = mode_value.iloc[0]
        df_clean[col] = df_clean[col].fillna(mode_value.iloc[0])

    print("=== 결측치 대체 결과 (최빈값) ===")
    for col in MISSING_COLS:
        print(f"{col:15s}: {before_missing[col]:>5,}건 -> 0건 (대체값: '{mode_values.get(col, 'N/A')}')")

    before_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed = before_rows - len(df_clean)
    print(f"\n=== 중복행 제거 ===")
    print(f"{before_rows:,}행 -> {len(df_clean):,}행 ({removed}건 제거)")

    stats = {
        "missing_before": before_missing.to_dict(),
        "mode_values": mode_values,
        "rows_before": before_rows,
        "rows_after": len(df_clean),
        "duplicates_removed": removed,
    }
    return df_clean, stats


def run_eda(df: pd.DataFrame) -> dict:
    """기본 EDA: describe, info, 범주형 value_counts, income 클래스 비율을 출력하고,
    report.md에 쓸 income 비율 dict를 반환한다."""
    print("\n" + "=" * 60)
    print("=== describe() : 수치형 컬럼 기술통계 ===")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("=== info() : 타입·결측치·메모리 ===")
    print("=" * 60)
    df.info()  # info()는 자체적으로 출력하고 None을 반환하므로 print()로 감싸지 않는다

    print("\n" + "=" * 60)
    print("=== 범주형 컬럼 값 분포 (상위 5개씩) ===")
    print("=" * 60)
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in categorical_cols:
        print(f"\n[{col}]")
        print(df[col].value_counts().head(5))

    print("\n" + "=" * 60)
    print("=== income 클래스 비율 (불균형 여부 확인용) ===")
    print("=" * 60)
    income_ratio = df["income"].value_counts(normalize=True) * 100
    print(income_ratio.round(2).astype(str) + "%")
    print("(참고: ML Pipeline에서 accuracy만 보면 이 불균형 때문에 착시가 생길 수 있어 F1도 같이 봐야 함)")

    return income_ratio.round(2).to_dict()


# ---------------------------------------------------------------------------
# Step 4. 시각화 (Seaborn 정적 1개 이상 + Plotly 인터랙티브 1개 이상)
# ---------------------------------------------------------------------------
def plot_eda_charts(df: pd.DataFrame) -> pd.DataFrame:
    """필수 차트(Seaborn=상관관계, Plotly=그룹비교) + 추가 차트(Seaborn=그룹비교,
    Plotly=분포)로 분포·상관관계·그룹비교 세 카테고리를 모두 다루도록 저장한다.
    report.md에서 재사용할 수 있도록 학력별 고소득 비율 표를 반환한다."""
    _require_columns(df, CORR_COLS + ["education", "income"], "plot_eda_charts")

    # --- Seaborn ① (필수, SPEC.md 산출물) : 상관관계 히트맵 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[CORR_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Adult Census Income - 수치형 변수 상관관계 히트맵")
    ax.set_xlabel("변수")
    ax.set_ylabel("변수")
    fig.tight_layout()
    fig.savefig(SEABORN_CHART_PATH, dpi=120)
    plt.close(fig)
    print(f"[Seaborn] 상관관계 히트맵 저장 완료: {SEABORN_CHART_PATH}")

    # --- Plotly ① (필수, SPEC.md 산출물) : 그룹비교 - 학력별 고소득(>50K) 비율 ---
    high_income_rate = (
        df.assign(is_high_income=(df["income"].str.strip() == ">50K"))
        .groupby("education", as_index=False)["is_high_income"]
        .mean()
        .assign(high_income_pct=lambda d: d["is_high_income"] * 100)
        .sort_values("high_income_pct", ascending=False)
    )
    fig_plotly = px.bar(
        high_income_rate,
        x="education",
        y="high_income_pct",
        title="학력별 고소득(income > 50K) 비율",
        labels={"education": "학력", "high_income_pct": "고소득 비율(%)"},
    )
    fig_plotly.write_html(PLOTLY_CHART_PATH)
    print(f"[Plotly] 학력별 고소득 비율 차트 저장 완료: {PLOTLY_CHART_PATH}")

    # --- Seaborn ② (추가/선택) : 그룹비교 - income별 hours-per-week 박스플롯 ---
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.boxplot(data=df, x="income", y="hours-per-week", ax=ax2)
    ax2.set_title("income 그룹별 주당 근무시간(hours-per-week) 분포")
    ax2.set_xlabel("income")
    ax2.set_ylabel("hours-per-week")
    fig2.tight_layout()
    fig2.savefig(SEABORN_EXTRA_CHART_PATH, dpi=120)
    plt.close(fig2)
    print(f"[Seaborn 추가] income별 근무시간 박스플롯 저장 완료: {SEABORN_EXTRA_CHART_PATH}")

    # --- Plotly ② (추가/선택) : 분포 - income별 색상 구분한 age 히스토그램 ---
    fig_plotly2 = px.histogram(
        df,
        x="age",
        color="income",
        barmode="overlay",
        opacity=0.6,
        nbins=40,
        title="income 그룹별 나이(age) 분포",
        labels={"age": "나이", "count": "인원수", "income": "income"},
    )
    fig_plotly2.write_html(PLOTLY_EXTRA_CHART_PATH)
    print(f"[Plotly 추가] income별 나이 분포 히스토그램 저장 완료: {PLOTLY_EXTRA_CHART_PATH}")

    return high_income_rate[["education", "high_income_pct"]]


# ---------------------------------------------------------------------------
# Step 5. 통계 분석 (기술통계 + 상관계수 + t-test)
# ---------------------------------------------------------------------------
def run_statistical_tests(df: pd.DataFrame) -> dict:
    """SPEC.md 규칙: 기술통계·상관계수 산출 후, income 그룹(<=50K vs >50K)의
    hours-per-week 평균 차이를 t-test로 검정하고 p-value를 해석한다.
    report.md용 결과 dict를 반환한다."""
    _require_columns(df, CORR_COLS + ["income"], "run_statistical_tests")

    describe_df = df[CORR_COLS].describe()
    print("=== 기술통계 (평균·표준편차·분위수) ===")
    print(describe_df)

    corr_df = df[CORR_COLS].corr().round(3)
    print("\n=== 상관계수 행렬 ===")
    print(corr_df)

    print("\n=== t-test : income(<=50K vs >50K) 그룹의 hours-per-week 평균 차이 ===")
    income_clean = df["income"].str.strip()
    under_50k = df.loc[income_clean == "<=50K", "hours-per-week"]
    over_50k = df.loc[income_clean == ">50K", "hours-per-week"]

    result = {"describe": describe_df, "corr": corr_df, "ttest": None}

    if under_50k.empty or over_50k.empty:
        print("[경고] 두 income 그룹 중 하나가 비어 있어 t-test를 건너뜁니다.")
        return result

    # equal_var=False: 두 그룹의 분산이 같다고 가정하지 않는 Welch's t-test
    t_stat, p_value = ttest_ind(under_50k, over_50k, equal_var=False)
    significant = p_value < 0.05
    print(f"<=50K 평균: {under_50k.mean():.2f} (n={len(under_50k):,})")
    print(f">50K  평균: {over_50k.mean():.2f} (n={len(over_50k):,})")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4g}")
    if significant:
        print("해석: p-value < 0.05 → 두 income 그룹의 평균 근무시간 차이는 통계적으로 유의미하다.")
    else:
        print("해석: p-value >= 0.05 → 두 income 그룹의 평균 근무시간 차이는 통계적으로 유의미하지 않다.")

    result["ttest"] = {
        "under_50k_mean": under_50k.mean(),
        "under_50k_n": len(under_50k),
        "over_50k_mean": over_50k.mean(),
        "over_50k_n": len(over_50k),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": significant,
    }
    return result


# ---------------------------------------------------------------------------
# Step 6. sklearn Pipeline (전처리 + 모델, 평가지표, joblib 저장)
# ---------------------------------------------------------------------------
def build_and_save_pipeline(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict:
    """SPEC.md 규칙: income을 0/1로 인코딩해 분류 문제로 풀고,
    수치형 5개+범주형 전체(fnlwgt 제외)를 feature로 사용한다.
    accuracy·F1을 출력하고 joblib으로 모델을 저장한 뒤, report.md용 결과 dict를 반환한다."""
    _require_columns(df, CORR_COLS + CATEGORICAL_FEATURES + ["income"], "build_and_save_pipeline")

    data = df.copy()
    data["income_binary"] = (data["income"].str.strip() == ">50K").astype(int)

    feature_cols = CORR_COLS + CATEGORICAL_FEATURES
    X = data[feature_cols]
    y = data["income_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CORR_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("=== sklearn Pipeline (RandomForestClassifier) 평가 ===")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1-score = {f1:.4f}")
    print("(참고: income 불균형(75:24) 때문에 accuracy만으로는 소수 클래스 예측력을 알 수 없어 F1을 같이 본다)")

    joblib.dump(pipeline, model_path)
    print(f"모델 저장 완료: {model_path}")

    return {
        "accuracy": acc,
        "f1": f1,
        "model_path": str(model_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": feature_cols,
    }


# ---------------------------------------------------------------------------
# Step 7. report.md 자동 생성
# ---------------------------------------------------------------------------
def generate_report(
    loading_stats: dict,
    clean_stats: dict,
    income_ratio: dict,
    stats_result: dict,
    model_result: dict,
    education_income_rate: pd.DataFrame,
    report_path: Path = REPORT_PATH,
) -> None:
    """Step 1~6 결과를 모아 report.md를 자동 생성한다.
    "핵심 인사이트" 섹션도 하드코딩된 문장이 아니라 실제 계산값에서 동적으로 만든다."""
    ttest = stats_result["ttest"]
    ttest_section = "t-test를 수행하지 못했습니다 (표본 부족)."
    ttest_insight = ""
    if ttest is not None:
        verdict = "통계적으로 유의미하다" if ttest["significant"] else "통계적으로 유의미하지 않다"
        ttest_section = (
            f"- `<=50K` 그룹 평균: {ttest['under_50k_mean']:.2f}시간 (n={ttest['under_50k_n']:,})\n"
            f"- `>50K` 그룹 평균: {ttest['over_50k_mean']:.2f}시간 (n={ttest['over_50k_n']:,})\n"
            f"- t-statistic = {ttest['t_stat']:.4f}, p-value = {ttest['p_value']:.4g}\n"
            f"- 해석: p-value {'< 0.05' if ttest['significant'] else '>= 0.05'} → "
            f"두 income 그룹의 평균 근무시간 차이는 **{verdict}**."
        )
        # 어느 그룹이 더 긴지는 실제 평균값을 비교해서 동적으로 판단한다 (하드코딩 금지)
        higher_group = ">50K" if ttest["over_50k_mean"] > ttest["under_50k_mean"] else "<=50K"
        diff = abs(ttest["over_50k_mean"] - ttest["under_50k_mean"])
        if ttest["significant"]:
            ttest_insight = (
                f"- `{higher_group}` 그룹의 평균 근무시간이 {diff:.1f}시간 더 길며, "
                f"이 차이는 t-test 결과 통계적으로 유의미하다 (p={ttest['p_value']:.4g})."
            )
        else:
            ttest_insight = (
                f"- 두 income 그룹 간 평균 근무시간 차이({diff:.1f}시간)는 t-test 결과 "
                f"통계적으로 유의미하지 않았다 (p={ttest['p_value']:.4g})."
            )

    missing_lines = "\n".join(
        f"- `{col}`: {before:,}건 → 0건 (대체값: `{clean_stats['mode_values'].get(col, 'N/A')}`)"
        for col, before in clean_stats["missing_before"].items()
    )

    income_lines = "\n".join(f"- `{label}`: {pct}%" for label, pct in income_ratio.items())

    # 상관계수 인사이트: 대각선(자기 자신=1.0)을 제외한 값 중 절댓값이 가장 큰 쌍을 동적으로 찾는다
    corr = stats_result["corr"]
    mask = ~np.eye(len(corr), dtype=bool)
    corr_pairs = corr.where(mask).stack()
    corr_insight = "상관계수를 계산할 수 있는 변수 쌍이 부족합니다."
    if not corr_pairs.empty:
        top_pair = corr_pairs.abs().idxmax()
        top_value = corr_pairs.loc[top_pair]
        corr_insight = (
            f"- 수치형 변수 중 `{top_pair[0]}`와(과) `{top_pair[1]}`의 상관계수가 {top_value:.3f}로 가장 크며, "
            f"나머지 변수쌍은 대체로 상관이 약해 서로 독립적인 정보를 담고 있는 것으로 보인다."
        )

    # 학력 인사이트: 실제 계산된 학력별 고소득 비율에서 최고/최저를 동적으로 찾는다
    edu_insight = "학력별 고소득 비율 데이터가 부족합니다."
    if not education_income_rate.empty:
        top_edu = education_income_rate.iloc[0]
        bottom_edu = education_income_rate.iloc[-1]
        edu_insight = (
            f"- 학력별 고소득(>50K) 비율은 `{top_edu['education']}`이 {top_edu['high_income_pct']:.1f}%로 가장 높고, "
            f"`{bottom_edu['education']}`이 {bottom_edu['high_income_pct']:.1f}%로 가장 낮아 "
            f"학력과 고소득 여부 사이에 뚜렷한 관계가 관찰된다."
        )

    content = f"""\
# Day2 종합실습 - Adult Census Income 분석 리포트

이 리포트는 `광주_2반_강주현.py` 실행 결과를 바탕으로 자동 생성되었습니다.
데이터셋: [Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

## 1. 데이터 준비

### 1-1. Pandas vs Polars 로딩 비교

| | Pandas | Polars |
|---|---|---|
| 로딩 시간 | {loading_stats['t_pandas']:.4f}초 | {loading_stats['t_polars']:.4f}초 |
| shape | {loading_stats['shape_pandas']} | {loading_stats['shape_polars']} |
| 중복행 | {loading_stats['dup_pandas']}건 | {loading_stats['dup_polars']}건 |

원본 결측치: {', '.join(f"`{c}` {n}건" for c, n in loading_stats['missing_pandas'].items())}

### 1-2. 결측치·중복 처리 (SPEC.md 규칙: 최빈값 대체, 중복행 제거)

{missing_lines}

- 중복행 제거: {clean_stats['rows_before']:,}행 → {clean_stats['rows_after']:,}행 ({clean_stats['duplicates_removed']}건 제거)

### 1-3. income 클래스 비율 (불균형 확인)

{income_lines}

> 참고: `<=50K`가 다수 클래스라 ML 평가 시 accuracy만으로는 부족해 F1도 함께 확인했습니다.

## 2. 시각화

| 카테고리 | 라이브러리 | 내용 | 파일 |
|---|---|---|---|
| 상관관계 (필수) | Seaborn | 수치형 5개 변수 상관 히트맵 | [{SEABORN_CHART_PATH.name}]({SEABORN_CHART_PATH.name}) |
| 그룹비교 (필수) | Plotly | 학력별 고소득(>50K) 비율 | [{PLOTLY_CHART_PATH.name}]({PLOTLY_CHART_PATH.name}) |
| 그룹비교 (추가) | Seaborn | income별 근무시간 박스플롯 | [{SEABORN_EXTRA_CHART_PATH.name}]({SEABORN_EXTRA_CHART_PATH.name}) |
| 분포 (추가) | Plotly | income별 나이 분포 히스토그램 | [{PLOTLY_EXTRA_CHART_PATH.name}]({PLOTLY_EXTRA_CHART_PATH.name}) |

![상관관계 히트맵]({SEABORN_CHART_PATH.name})

![income별 근무시간 박스플롯]({SEABORN_EXTRA_CHART_PATH.name})

> Plotly 인터랙티브 차트는 이미지로 표시되지 않으니 위 링크를 눌러 직접 열어서 확인하세요.

## 3. 통계 분석

### 3-1. 기술통계 (평균·표준편차·분위수)

{_df_to_markdown(stats_result['describe'])}

### 3-2. 상관계수 행렬

{_df_to_markdown(stats_result['corr'])}

### 3-3. t-test : income(<=50K vs >50K) 그룹의 hours-per-week 평균 차이

{ttest_section}

## 4. ML Pipeline

- 모델: `RandomForestClassifier` (ColumnTransformer + Pipeline)
- feature: {', '.join(f'`{c}`' for c in model_result['features'])}
- train/test: {model_result['n_train']:,}건 / {model_result['n_test']:,}건 (test_size=0.2, stratify=income)
- **Accuracy = {model_result['accuracy']:.4f}**
- **F1-score = {model_result['f1']:.4f}**
- 모델 저장 경로: `{model_result['model_path']}`

## 5. 핵심 인사이트

{edu_insight}
{ttest_insight}
{corr_insight}

---
*이 문서는 `generate_report()` 함수에 의해 자동 생성되었습니다. 수동 편집 없이 스크립트 재실행만으로 갱신됩니다.*
"""

    report_path.write_text(content, encoding="utf-8")
    print(f"\nreport.md 생성 완료: {report_path}")


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def main() -> None:
    set_korean_font()
    raw_df, loading_stats = compare_loading(URL)

    clean_df, clean_stats = handle_missing_and_duplicates(raw_df)
    income_ratio = run_eda(clean_df)

    plot_eda_charts_result = plot_eda_charts(clean_df)
    stats_result = run_statistical_tests(clean_df)
    model_result = build_and_save_pipeline(clean_df)

    generate_report(
        loading_stats, clean_stats, income_ratio, stats_result, model_result,
        education_income_rate=plot_eda_charts_result,
    )


if __name__ == "__main__":
    main()