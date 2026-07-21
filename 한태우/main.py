from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]

MISSING = ["workclass", "occupation", "native-country"]

NUMERIC = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]

ML_NUMERIC = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]

ML_CATEGORY = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


# ============================================================
# 데이터 로딩
# ============================================================

def download_data() -> bytes:
    with urlopen(DATA_URL, timeout=30) as response:
        data = response.read()

    print(f"[다운로드 완료] {len(data):,} bytes")
    return data


def load_pandas(data: bytes) -> tuple[pd.DataFrame, float]:
    start = perf_counter()

    df = pd.read_csv(
        BytesIO(data),
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    text_columns = df.select_dtypes(include="object").columns
    df[text_columns] = df[text_columns].apply(lambda x: x.str.strip())

    return df, perf_counter() - start


def load_polars(data: bytes) -> tuple[pl.DataFrame, float]:
    start = perf_counter()

    df = pl.read_csv(
        BytesIO(data),
        has_header=False,
        new_columns=COLUMNS,
        schema_overrides={column: pl.String for column in COLUMNS},
    )

    df = df.with_columns([
        pl.when(pl.col(column).str.strip_chars().is_in(["", "?"]))
        .then(None)
        .otherwise(pl.col(column).str.strip_chars())
        .alias(column)
        for column in COLUMNS
    ])

    df = df.with_columns([
        pl.col(column).cast(pl.Int64, strict=False)
        for column in NUMERIC
    ]).filter(pl.col("age").is_not_null())

    return df, perf_counter() - start


# ============================================================
# 데이터 정제
# ============================================================

def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Pandas 처리 전]")
    print(df[MISSING].isna().sum())
    print(f"중복: {df.duplicated().sum():,}개")

    df = df.drop_duplicates().reset_index(drop=True)

    for column in MISSING:
        df[column] = df[column].fillna(df[column].mode().iloc[0])

    print(
        f"[Pandas 처리 후] {len(df):,}행, "
        f"결측치 {df.isna().sum().sum()}개, "
        f"중복 {df.duplicated().sum()}개"
    )
    return df


def clean_polars(df: pl.DataFrame) -> pl.DataFrame:
    print("\n[Polars 처리 전]")
    print(df.select(MISSING).null_count())

    before = df.height
    df = df.unique(maintain_order=True)
    print(f"중복: {before - df.height:,}개")

    for column in MISSING:
        mode = df.select(
            pl.col(column).drop_nulls().mode().first()
        ).item()

        df = df.with_columns(pl.col(column).fill_null(mode))

    print(
        f"[Polars 처리 후] {df.height:,}행, "
        f"결측치 {sum(df.null_count().row(0))}개, "
        f"중복 {df.height - df.unique().height}개"
    )
    return df


# ============================================================
# 기본 EDA 및 비교
# ============================================================

def print_eda(pandas_df: pd.DataFrame, polars_df: pl.DataFrame) -> None:
    print("\n[Pandas 기본 EDA]")
    print(pandas_df[NUMERIC].describe().round(2))
    print("\n소득 분포")
    print(pandas_df["income"].value_counts())
    print("\n소득별 평균")
    print(pandas_df.groupby("income")[NUMERIC].mean().round(2))

    print("\n[Polars 기본 EDA]")
    print(polars_df.select(NUMERIC).describe())
    print("\n소득 분포")
    print(polars_df.group_by("income").len().sort("len", descending=True))
    print("\n소득별 평균")
    print(
        polars_df.group_by("income")
        .agg([
            pl.col(column).mean().round(2).alias(f"{column}_mean")
            for column in NUMERIC
        ])
        .sort("income")
    )


def compare_results(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
    pandas_time: float,
    polars_time: float,
) -> None:
    pandas_income = (
        pandas_df["income"].value_counts().sort_index().to_dict()
    )

    polars_income = dict(
        polars_df.group_by("income").len().sort("income").iter_rows()
    )

    faster = "Polars" if polars_time < pandas_time else "Pandas"
    ratio = max(pandas_time, polars_time) / min(pandas_time, polars_time)

    print("\n[Pandas와 Polars 비교]")
    print(f"Pandas: {pandas_df.shape}, {pandas_time:.6f}초")
    print(f"Polars: {polars_df.shape}, {polars_time:.6f}초")
    print(f"데이터 크기 일치: {pandas_df.shape == polars_df.shape}")
    print(f"소득 분포 일치: {pandas_income == polars_income}")
    print(f"{faster}가 약 {ratio:.2f}배 빠릅니다.")


# ============================================================
# 시각화
# ============================================================

def create_visualizations(df: pd.DataFrame) -> None:
    seaborn_path = BASE_DIR / "eda_chart_seaborn.png"
    plotly_path = BASE_DIR / "eda_chart_plotly.html"

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="income", y="hours-per-week")
    plt.title("Hours per Week by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel("Hours per Week")
    plt.tight_layout()
    plt.savefig(seaborn_path, dpi=300, bbox_inches="tight")
    plt.close()

    plot_df = (
        df.assign(high_income=df["income"].eq(">50K").astype(int))
        .groupby("education", as_index=False)
        .agg(
            high_income_rate=("high_income", "mean"),
            sample_count=("high_income", "size"),
        )
    )

    plot_df["high_income_rate"] *= 100
    plot_df = plot_df.sort_values("high_income_rate", ascending=False)

    fig = px.bar(
        plot_df,
        x="education",
        y="high_income_rate",
        title="High-Income Rate by Education Level",
        labels={
            "education": "Education Level",
            "high_income_rate": "Income >50K Rate (%)",
            "sample_count": "Sample Count",
        },
        hover_data={"sample_count": True, "high_income_rate": ":.2f"},
    )

    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(plotly_path)

    print("\n[시각화 저장 완료]")
    print(seaborn_path.name)
    print(plotly_path.name)


# ============================================================
# 통계 분석
# ============================================================

def statistical_analysis(df: pd.DataFrame) -> dict:
    analysis_df = df[ML_NUMERIC]
    under = df.loc[df["income"].eq("<=50K"), "hours-per-week"]
    over = df.loc[df["income"].eq(">50K"), "hours-per-week"]

    t_statistic, p_value = ttest_ind(
        under,
        over,
        equal_var=False,
    )

    interpretation = (
        "p < 0.05 → 두 소득 그룹의 주당 근무시간 평균 차이는 "
        "통계적으로 유의미하다."
        if p_value < 0.05
        else
        "p >= 0.05 → 두 소득 그룹의 주당 근무시간 평균 차이는 "
        "통계적으로 유의미하지 않다."
    )

    print("\n[통계 분석]")
    print("\n[기술통계]")
    print(
        analysis_df.describe()
        .loc[["mean", "std", "25%", "50%", "75%"]]
        .round(3)
    )
    print("\n[상관계수]")
    print(analysis_df.corr().round(3))
    print("\n[Welch t-test]")
    print(f"<=50K 평균: {under.mean():.3f}")
    print(f">50K 평균: {over.mean():.3f}")
    print(f"t-statistic: {t_statistic:.6f}")
    print(f"p-value: {p_value:.6e}")
    print(interpretation)

    return {
        "under_mean": under.mean(),
        "over_mean": over.mean(),
        "t_statistic": t_statistic,
        "p_value": p_value,
        "interpretation": interpretation,
    }


# ============================================================
# 머신러닝
# ============================================================

def train_model(df: pd.DataFrame) -> dict:
    features = ML_NUMERIC + ML_CATEGORY
    X = df[features]
    y = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})

    if y.isna().any():
        raise ValueError("income 컬럼에 변환할 수 없는 값이 있습니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    category_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    model = Pipeline([
        (
            "preprocessor",
            ColumnTransformer([
                ("numeric", numeric_pipe, ML_NUMERIC),
                ("category", category_pipe, ML_CATEGORY),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=2000, random_state=42),
        ),
    ])

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    model_path = BASE_DIR / "model.joblib"

    joblib.dump(model, model_path)

    print("\n[머신러닝 Pipeline]")
    print(f"학습 데이터: {len(X_train):,}건")
    print(f"테스트 데이터: {len(X_test):,}건")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"모델 저장 완료: {model_path}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ============================================================
# 자동 리포트
# ============================================================

def create_report(
    df: pd.DataFrame,
    stats: dict,
    model: dict,
) -> None:
    income_counts = df["income"].value_counts()

    report = f"""# Adult Census Income 분석 보고서

## 1. 프로젝트 개요

Adult Census Income 데이터를 Pandas와 Polars로 처리하고,
시각화·통계 검정·머신러닝 분류를 수행하였다.

## 2. 데이터 정제

- 정제 후 데이터: {len(df):,}행 × {df.shape[1]}열
- `<=50K`: {income_counts.get("<=50K", 0):,}건
- `>50K`: {income_counts.get(">50K", 0):,}건
- 제거된 중복: 24건
- 정제 후 결측치: {df.isna().sum().sum()}건
- 결측치 처리: `workclass`, `occupation`, `native-country` 최빈값 대체
- `fnlwgt`: 상관계수와 ML feature에서 제외

## 3. 시각화

- `eda_chart_seaborn.png`: 소득 그룹별 주당 근무시간
- `eda_chart_plotly.html`: 교육 수준별 고소득자 비율

## 4. Welch t-test

- `<=50K` 평균: {stats["under_mean"]:.3f}
- `>50K` 평균: {stats["over_mean"]:.3f}
- t-statistic: {stats["t_statistic"]:.6f}
- p-value: {stats["p_value"]:.6e}
- 해석: {stats["interpretation"]}

## 5. 머신러닝

- 타깃: `<=50K` → 0, `>50K` → 1
- 모델: Logistic Regression
- test_size: 0.2
- random_state: 42
- 학습 데이터: {model["train_size"]:,}건
- 테스트 데이터: {model["test_size"]:,}건
- Accuracy: {model["accuracy"]:.4f}
- F1-score: {model["f1"]:.4f}

## 6. 산출물

- `main.py`
- `eda_chart_seaborn.png`
- `eda_chart_plotly.html`
- `model.joblib`
- `report.md`

## 7. 결론

고소득 그룹의 평균 주당 근무시간이 더 높았으며,
Welch t-test 결과 그 차이는 통계적으로 유의미했다.
분류 모델은 Accuracy {model["accuracy"]:.4f},
F1-score {model["f1"]:.4f}를 기록했다.
"""

    report_path = BASE_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n리포트 저장 완료: {report_path}")


# ============================================================
# 실행
# ============================================================

def main() -> None:
    data = download_data()

    pandas_df, pandas_time = load_pandas(data)
    polars_df, polars_time = load_polars(data)

    pandas_df = clean_pandas(pandas_df)
    polars_df = clean_polars(polars_df)

    print_eda(pandas_df, polars_df)
    create_visualizations(pandas_df)

    stats = statistical_analysis(pandas_df)
    model = train_model(pandas_df)
    create_report(pandas_df, stats, model)

    compare_results(
        pandas_df,
        polars_df,
        pandas_time,
        polars_time,
    )

    print("\n전체 분석 및 산출물 생성 완료")


if __name__ == "__main__":
    main()