#!/usr/bin/env python3
"""SKALA Day 2 종합 실습 - Adult Census Income End-to-End 분석.

PDF p.119~123과 팀 공통 spec.md의 요구사항을 한 번의 실행으로 수행한다.

실행 예시
---------
    python main.py
    python main.py --data "/Users/lim/data-project/adult.data.txt"
    python main.py --output-dir ./result

기본 산출물은 이 스크립트와 같은 폴더에 생성된다.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any


# 패키지가 실제로 없는 경우에만 설치 메시지를 출력한다. ImportError 전체를
# 한꺼번에 잡으면 패키지 내부 오류도 "미설치"로 잘못 안내할 수 있다.
REQUIRED_MODULES = {
    "pandas": "pandas",
    "polars": "polars",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
}


def check_required_packages() -> None:
    """현재 Python 인터프리터에서 필요한 모듈을 찾을 수 있는지 확인한다."""
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_MODULES.items()
        if find_spec(module_name) is None
    ]
    if missing:
        command = f'"{sys.executable}" -m pip install ' + " ".join(missing)
        raise SystemExit(
            "필수 패키지가 없습니다: "
            + ", ".join(missing)
            + f"\n현재 Python: {sys.executable}\n설치 명령: {command}"
        )


check_required_packages()

# GUI가 없는 환경에서도 PNG를 저장하고, Matplotlib 캐시가 저장소에 생기지 않게 한다.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skala-matplotlib-cache")
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
SCRIPT_DIR = Path(__file__).resolve().parent

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_SOURCE_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# spec.md에서 고정한 상관분석 및 ML 수치형 feature. fnlwgt는 의도적으로 제외한다.
NUMERIC_FEATURES = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

MISSING_VALUE_COLUMNS = ["workclass", "occupation", "native-country"]

HEADER_ALIASES = {
    "education_num": "education-num",
    "marital_status": "marital-status",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country",
}


def parse_args() -> argparse.Namespace:
    """명령행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description="Adult Census Income 데이터의 End-to-End 분석을 수행합니다."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Adult 데이터 파일 경로. 생략하면 로컬 후보를 찾은 뒤 UCI에서 내려받습니다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="산출물 저장 폴더(기본값: 1.py가 있는 폴더)",
    )
    return parser.parse_args()


def section(title: str) -> None:
    """콘솔 출력의 구역을 명확하게 구분한다."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def format_p_value(p_value: float) -> str:
    """극소 확률이 부동소수점에서 0으로 보이는 오해를 피한다."""
    if p_value == 0.0:
        return "< 1e-300 (부동소수점 정밀도 한계)"
    if p_value < 1e-6:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


def is_headered_csv(path: Path) -> bool:
    """첫 번째 비어 있지 않은 줄로 헤더 포함 여부를 판별한다."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if line.strip():
                first_cell = line.split(",", maxsplit=1)[0].strip().lower()
                return first_cell == "age"
    raise ValueError(f"데이터 파일이 비어 있습니다: {path}")


def resolve_data_path(requested_path: Path | None) -> Path:
    """명시 경로, 로컬 후보, 사용자 캐시, UCI 다운로드 순으로 데이터를 찾는다."""
    if requested_path is not None:
        path = requested_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--data 파일을 찾을 수 없습니다: {path}")
        return path

    # 현재 학습 환경의 기존 파일도 찾되, 절대경로를 코드에 고정하지 않는다.
    candidates = [
        SCRIPT_DIR / "adult.data",
        SCRIPT_DIR / "adult.data.txt",
        SCRIPT_DIR.parent / "adult.data",
        SCRIPT_DIR.parent / "adult.data.txt",
        SCRIPT_DIR.parent.parent / "adult.data",
        SCRIPT_DIR.parent.parent / "adult.data.txt",
        SCRIPT_DIR.parent.parent / "Adult Census Income.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    cache_path = cache_root / "skala-aiops" / "adult.data"
    if cache_path.is_file():
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    print(f"로컬 데이터가 없어 UCI에서 다운로드합니다: {DATA_URL}")
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "Mozilla/5.0 SKALA-Day2-EDA/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        temporary_path.replace(cache_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Adult 데이터 다운로드에 실패했습니다. 인터넷 연결을 확인하거나 "
            "--data /path/to/adult.data 옵션을 사용하세요. "
            f"원인: {type(exc).__name__}: {exc}"
        ) from exc
    return cache_path.resolve()


def validate_columns(columns: list[str]) -> None:
    """데이터 스키마가 과제의 15개 컬럼과 일치하는지 확인한다."""
    missing = [column for column in COLUMNS if column not in columns]
    extra = [column for column in columns if column not in COLUMNS]
    if missing or extra:
        raise ValueError(
            "Adult 데이터 컬럼이 예상 스키마와 다릅니다. "
            f"누락={missing}, 추가={extra}"
        )


def load_with_pandas(path: Path, has_header: bool) -> pd.DataFrame:
    """Pandas로 Adult 데이터를 독립적으로 로딩하고 문자열을 정규화한다."""
    common_options: dict[str, Any] = {
        "na_values": "?",  # spec.md 요구: 공백 없는 '?' 사용
        "skipinitialspace": True,
        "skip_blank_lines": True,
    }
    if has_header:
        frame = pd.read_csv(path, **common_options).rename(columns=HEADER_ALIASES)
    else:
        frame = pd.read_csv(
            path, header=None, names=COLUMNS, **common_options
        )

    frame.columns = [str(column).strip() for column in frame.columns]
    validate_columns(frame.columns.tolist())
    frame = frame[COLUMNS].dropna(how="all").copy()

    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("?", pd.NA)
    for column in NUMERIC_SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def load_with_polars(path: Path, has_header: bool) -> pl.DataFrame:
    """Polars로 같은 원본을 독립적으로 로딩하고 동일한 스키마로 정규화한다."""
    common_options: dict[str, Any] = {
        "null_values": ["?", " ?", ""],
        "infer_schema_length": 10_000,
    }
    if has_header:
        frame = pl.read_csv(path, has_header=True, **common_options)
        rename_map = {
            old: new for old, new in HEADER_ALIASES.items() if old in frame.columns
        }
        frame = frame.rename(rename_map)
    else:
        frame = pl.read_csv(
            path,
            has_header=False,
            new_columns=COLUMNS,
            schema_overrides={column: pl.Int64 for column in NUMERIC_SOURCE_COLUMNS},
            **common_options,
        )

    frame = frame.rename({column: column.strip() for column in frame.columns})
    validate_columns(frame.columns)
    frame = frame.select(COLUMNS).filter(~pl.all_horizontal(pl.all().is_null()))
    frame = frame.with_columns(pl.col(pl.String).str.strip_chars())
    frame = frame.with_columns(
        [
            pl.when(pl.col(column) == "?")
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in frame.select(pl.col(pl.String)).columns
        ]
    )
    frame = frame.with_columns(
        [pl.col(column).cast(pl.Int64, strict=True) for column in NUMERIC_SOURCE_COLUMNS]
    )
    return frame


def pandas_missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Pandas 결측치 수를 일반 dict로 변환한다."""
    return {column: int(value) for column, value in frame.isna().sum().items()}


def polars_missing_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Polars 결측치 수를 일반 dict로 변환한다."""
    row = frame.select(pl.all().null_count()).row(0, named=True)
    return {column: int(value) for column, value in row.items()}


def clean_pandas(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, str]]:
    """중복 제거 후 세 범주형 결측 컬럼을 각각의 최빈값으로 대체한다."""
    clean = frame.drop_duplicates(keep="first").copy()
    removed_duplicates = len(frame) - len(clean)
    modes: dict[str, str] = {}
    for column in MISSING_VALUE_COLUMNS:
        mode_values = clean[column].mode(dropna=True)
        if mode_values.empty:
            raise ValueError(f"최빈값을 계산할 수 없는 컬럼입니다: {column}")
        mode_value = str(mode_values.iloc[0])
        modes[column] = mode_value
        clean[column] = clean[column].fillna(mode_value)
    return clean, removed_duplicates, modes


def clean_polars(frame: pl.DataFrame) -> tuple[pl.DataFrame, int, dict[str, str]]:
    """Polars에서도 Pandas와 같은 순서로 중복·결측치를 처리한다."""
    clean = frame.unique(maintain_order=True)
    removed_duplicates = frame.height - clean.height
    modes: dict[str, str] = {}
    for column in MISSING_VALUE_COLUMNS:
        mode_frame = clean.select(pl.col(column).drop_nulls().mode().sort().first())
        mode_value = mode_frame.item()
        if mode_value is None:
            raise ValueError(f"최빈값을 계산할 수 없는 컬럼입니다: {column}")
        modes[column] = str(mode_value)
        clean = clean.with_columns(pl.col(column).fill_null(mode_value))
    return clean, removed_duplicates, modes


def print_pandas_eda(frame: pd.DataFrame) -> None:
    """Pandas 기본 EDA 결과를 출력한다."""
    section("1-1. Pandas 기본 EDA")
    print(f"shape: {frame.shape}")
    print("\n[head]")
    print(frame.head())
    print("\n[info]")
    frame.info()
    print("\n[결측치 수]")
    print(frame.isna().sum())
    print(f"\n중복 행 수: {int(frame.duplicated().sum()):,}")
    print("\n[수치형 기술통계]")
    print(frame[NUMERIC_SOURCE_COLUMNS].describe().round(3))


def print_polars_eda(frame: pl.DataFrame) -> None:
    """Polars 기본 EDA 결과를 출력한다."""
    section("1-2. Polars 기본 EDA")
    print(f"shape: {frame.shape}")
    print("\n[head]")
    print(frame.head())
    print("\n[schema]")
    for column, dtype in frame.schema.items():
        print(f"{column:>16}: {dtype}")
    print("\n[결측치 수]")
    print(frame.select(pl.all().null_count()))
    # is_duplicated()는 최초 행까지 포함한 '중복 그룹 전체'를 표시한다.
    # Pandas duplicated().sum()과 동일하게 실제 제거 대상 행 수를 출력한다.
    duplicate_rows = frame.height - frame.unique().height
    print(f"\n중복 행 수: {duplicate_rows:,}")
    print("\n[수치형 기술통계]")
    print(frame.select(NUMERIC_SOURCE_COLUMNS).describe())


def compare_cleaning_results(
    pandas_raw: pd.DataFrame,
    pandas_clean: pd.DataFrame,
    pandas_duplicates: int,
    pandas_modes: dict[str, str],
    polars_raw: pl.DataFrame,
    polars_clean: pl.DataFrame,
    polars_duplicates: int,
    polars_modes: dict[str, str],
) -> pd.DataFrame:
    """Pandas와 Polars의 원본·정제 결과를 같은 표로 비교한다."""
    comparison = pd.DataFrame(
        {
            "Pandas": [
                len(pandas_raw),
                int(pandas_raw.isna().sum().sum()),
                pandas_duplicates,
                len(pandas_clean),
                int(pandas_clean.isna().sum().sum()),
            ],
            "Polars": [
                polars_raw.height,
                sum(polars_missing_counts(polars_raw).values()),
                polars_duplicates,
                polars_clean.height,
                sum(polars_missing_counts(polars_clean).values()),
            ],
        },
        index=[
            "원본 행 수",
            "원본 결측치 수",
            "제거한 중복 행 수",
            "정제 후 행 수",
            "정제 후 결측치 수",
        ],
    )
    if not comparison["Pandas"].equals(comparison["Polars"]):
        raise AssertionError("Pandas와 Polars의 정제 결과 요약이 서로 다릅니다.")
    if pandas_modes != polars_modes:
        raise AssertionError(
            f"Pandas와 Polars의 최빈값이 다릅니다: {pandas_modes} != {polars_modes}"
        )
    return comparison


def create_visualizations(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Seaborn 정적 차트와 Plotly 인터랙티브 차트를 각각 생성한다."""
    static_path = output_dir / "eda_chart_seaborn.png"
    interactive_path = output_dir / "eda_chart_plotly.html"

    sns.set_theme(style="whitegrid", context="notebook")
    figure, (box_axis, heatmap_axis) = plt.subplots(1, 2, figsize=(15, 6))

    # (좌) income 그룹별 주당 근로시간 분포 — t-test 대상 변수의 시각적 근거.
    sns.boxplot(
        data=frame,
        x="income",
        y="hours-per-week",
        hue="income",
        legend=False,
        palette={"<=50K": "#4C78A8", ">50K": "#F58518"},
        showfliers=False,
        ax=box_axis,
    )
    box_axis.set_title("Weekly Work Hours by Income Group", fontsize=14, pad=12)
    box_axis.set_xlabel("Income Group")
    box_axis.set_ylabel("Hours per Week")

    # (우) spec 고정 수치형 5개의 상관계수 히트맵 — 리포트 상관표의 시각화.
    correlation = frame[NUMERIC_FEATURES].corr(method="pearson")
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=heatmap_axis,
    )
    heatmap_axis.set_title("Correlation of Numeric Features", fontsize=14, pad=12)
    heatmap_axis.set_xlabel("Feature")
    heatmap_axis.set_ylabel("Feature")

    figure.tight_layout()
    figure.savefig(static_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    plot_data = (
        frame.groupby(["education", "income"], as_index=False, observed=True)
        .agg(
            mean_hours=("hours-per-week", "mean"),
            sample_count=("hours-per-week", "size"),
            education_num=("education-num", "median"),
        )
        .sort_values(["education_num", "income"])
    )
    interactive_figure = px.bar(
        plot_data,
        x="education",
        y="mean_hours",
        color="income",
        barmode="group",
        hover_data={"sample_count": True, "education_num": False},
        category_orders={
            "education": plot_data["education"].drop_duplicates().tolist(),
            "income": ["<=50K", ">50K"],
        },
        labels={
            "education": "Education Level",
            "mean_hours": "Average Hours per Week",
            "income": "Income Group",
            "sample_count": "Sample Count",
        },
        title="Average Weekly Work Hours by Education and Income Group",
    )
    interactive_figure.update_layout(
        template="plotly_white",
        xaxis_tickangle=-35,
        legend_title_text="Income Group",
        margin={"l": 60, "r": 30, "t": 80, "b": 120},
    )
    interactive_figure.write_html(
        interactive_path, include_plotlyjs=True, full_html=True
    )
    return static_path, interactive_path


def calculate_statistics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, str, dict[str, float]]:
    """기술통계·상관계수와 income 그룹 간 Welch t-test를 계산한다."""
    descriptive = (
        frame[NUMERIC_FEATURES]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["mean", "std", "25%", "50%", "75%"]]
    )
    correlation = frame[NUMERIC_FEATURES].corr(method="pearson")

    income = frame["income"].astype("string").str.strip()
    group_50k_under = frame.loc[income == "<=50K", "hours-per-week"].dropna()
    group_50k_over = frame.loc[income == ">50K", "hours-per-week"].dropna()
    if group_50k_under.empty or group_50k_over.empty:
        raise ValueError("t-test를 위한 두 income 그룹 중 하나가 비어 있습니다.")

    result = stats.ttest_ind(
        group_50k_under,
        group_50k_over,
        equal_var=False,  # spec.md 요구: Welch's t-test
        nan_policy="omit",
    )
    t_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if p_value < 0.05:
        interpretation = (
            "p < 0.05 → 두 income 그룹의 hours-per-week 평균 차이는 "
            "통계적으로 유의미하다."
        )
    else:
        interpretation = (
            "p >= 0.05 → 두 income 그룹의 hours-per-week 평균 차이는 "
            "통계적으로 유의미하지 않다."
        )
    group_means = {
        "<=50K": float(group_50k_under.mean()),
        ">50K": float(group_50k_over.mean()),
    }
    return descriptive, correlation, t_statistic, p_value, interpretation, group_means


def train_and_save_model(
    frame: pd.DataFrame, output_dir: Path
) -> tuple[Pipeline, float, float, str, Path, int, int]:
    """전처리와 모델을 하나의 Pipeline으로 학습·평가하고 joblib으로 저장한다."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    features = frame[feature_columns].copy()
    target = frame["income"].astype("string").str.strip().map({"<=50K": 0, ">50K": 1})
    if target.isna().any():
        unknown = sorted(frame.loc[target.isna(), "income"].astype(str).unique())
        raise ValueError(f"income 타깃에 알 수 없는 값이 있습니다: {unknown}")
    target = target.astype("int8")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, pos_label=1))
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        digits=4,
        zero_division=0,
    )

    model_path = output_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    return pipeline, accuracy, f1, report, model_path, len(x_train), len(x_test)


def markdown_table(frame: pd.DataFrame, index_label: str = "항목") -> str:
    """tabulate 추가 설치 없이 DataFrame을 Markdown 표로 변환한다."""
    display = frame.copy()
    display.insert(0, index_label, display.index.astype(str))

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def create_report(
    output_dir: Path,
    data_path: Path,
    comparison: pd.DataFrame,
    missing_before: dict[str, int],
    missing_after: dict[str, int],
    modes: dict[str, str],
    descriptive: pd.DataFrame,
    correlation: pd.DataFrame,
    t_statistic: float,
    p_value: float,
    ttest_interpretation: str,
    group_means: dict[str, float],
    accuracy: float,
    f1: float,
    train_rows: int,
    test_rows: int,
) -> Path:
    """실행 결과와 핵심 해석이 포함된 report.md를 자동 생성한다."""
    missing_table = pd.DataFrame(
        {
            "처리 전": [missing_before[column] for column in MISSING_VALUE_COLUMNS],
            "대체 최빈값": [modes[column] for column in MISSING_VALUE_COLUMNS],
            "처리 후": [missing_after[column] for column in MISSING_VALUE_COLUMNS],
        },
        index=MISSING_VALUE_COLUMNS,
    )
    report_path = output_dir / "report.md"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    report_text = f"""# Adult Census Income End-to-End 분석 보고서

- 자동 생성 시각: {generated_at}
- 데이터 출처: UCI Adult Census Income
- 원본 URL: {DATA_URL}
- 실행에 사용한 파일: `{data_path}`
- 타깃: `income` (`<=50K`=0, `>50K`=1)

## 1. 데이터 준비 및 Pandas·Polars 비교

Pandas는 `na_values="?"`, `skipinitialspace=True`로 로딩했다. Polars도 같은 원본을
독립적으로 로딩한 뒤 문자열 공백과 `?`를 결측치로 정규화했다. 중복 행은 첫 행을
남기고 제거했으며, `workclass`, `occupation`, `native-country`의 결측치는 행을
삭제하지 않고 각 컬럼의 최빈값으로 대체했다.

{markdown_table(comparison, "처리 항목")}

### 결측치 처리

{markdown_table(missing_table, "컬럼")}

## 2. 기술통계

`fnlwgt`는 인구총조사 가중치이므로 팀 SPEC에 따라 상관분석과 ML feature에서 제외했다.

{markdown_table(descriptive, "수치형 변수")}

## 3. 상관계수

{markdown_table(correlation, "변수")}

## 4. 시각화

### Seaborn 정적 차트

좌측은 income 그룹별 주당 근로시간 분포(t-test 대상 변수), 우측은 수치형 5개
변수의 상관계수 히트맵이다.

![Income 그룹별 주당 근로시간과 수치형 변수 상관관계](eda_chart_seaborn.png)

### Plotly 인터랙티브 차트

[교육수준·소득그룹별 평균 주당 근로시간 열기](eda_chart_plotly.html)

## 5. Welch's t-test

- 비교: `income <=50K`와 `income >50K` 그룹의 `hours-per-week` 평균
- `<=50K` 평균: {group_means['<=50K']:.4f}
- `>50K` 평균: {group_means['>50K']:.4f}
- t-statistic: {t_statistic:.6f}
- p-value: {format_p_value(p_value)}
- 해석: **{ttest_interpretation}**

## 6. ML Pipeline

- 전처리: 수치형 중앙값 대체 + 표준화, 범주형 최빈값 대체 + One-Hot Encoding
- 모델: Logistic Regression
- 분할: `test_size=0.2`, `random_state=42`, stratified split
- 학습 행 수: {train_rows:,}
- 평가 행 수: {test_rows:,}
- Accuracy: **{accuracy:.4f}**
- F1-score (`>50K`): **{f1:.4f}**
- 저장 모델: `model.joblib`

## 7. 분석 의견 및 개선 방향

- `>50K` 그룹의 평균 주당 근로시간은 {group_means['>50K']:.2f}시간으로,
  `<=50K` 그룹의 {group_means['<=50K']:.2f}시간보다 높다.
- t-test는 두 그룹 평균 차이의 통계적 유의성을 보여주지만 인과관계를 증명하지는 않는다.
- 본 모델은 해석과 재현성이 좋은 기준 모델이다. 향후 교차검증, 클래스 불균형 대응,
  트리 기반 모델과의 비교로 F1-score를 개선할 수 있다.
- 데이터셋에는 성별·인종 등 민감 특성이 포함되어 있으므로 서비스 적용 전 그룹별
  성능과 공정성 지표를 별도로 점검해야 한다.

## 8. 자동 생성 산출물

- `eda_chart_seaborn.png`
- `eda_chart_plotly.html`
- `model.joblib`
- `report.md`

이 문서는 `main.py` 실행 결과로 자동 생성되었으며 수동 계산값을 포함하지 않는다.
"""
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    """전체 분석 단계를 순서대로 실행한다."""
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_data_path(args.data)
    has_header = is_headered_csv(data_path)

    section("0. 실행 환경 및 데이터")
    print(f"Python 실행 파일: {sys.executable}")
    print(f"데이터 파일: {data_path}")
    print(f"헤더 포함 여부: {has_header}")
    print(f"산출물 폴더: {output_dir}")
    print(f"Pandas {pd.__version__} / Polars {pl.__version__}")

    pandas_raw = load_with_pandas(data_path, has_header)
    polars_raw = load_with_polars(data_path, has_header)
    print_pandas_eda(pandas_raw)
    print_polars_eda(polars_raw)

    pandas_missing_before = pandas_missing_counts(pandas_raw)
    pandas_clean, pandas_duplicates, pandas_modes = clean_pandas(pandas_raw)
    polars_clean, polars_duplicates, polars_modes = clean_polars(polars_raw)
    comparison = compare_cleaning_results(
        pandas_raw,
        pandas_clean,
        pandas_duplicates,
        pandas_modes,
        polars_raw,
        polars_clean,
        polars_duplicates,
        polars_modes,
    )
    pandas_missing_after = pandas_missing_counts(pandas_clean)

    section("2. 결측치·중복 처리 및 도구 비교")
    print(comparison)
    print(f"\nPandas 최빈값: {pandas_modes}")
    print(f"Polars 최빈값: {polars_modes}")
    print("\n[Pandas 정제 후 결측치]")
    print(pandas_clean.isna().sum())
    print("\n[Polars 정제 후 결측치]")
    print(polars_clean.select(pl.all().null_count()))

    section("3. 시각화")
    static_path, interactive_path = create_visualizations(pandas_clean, output_dir)
    print(f"Seaborn 정적 차트: {static_path}")
    print(f"Plotly 인터랙티브 차트: {interactive_path}")

    section("4. 기술통계·상관계수·Welch t-test")
    (
        descriptive,
        correlation,
        t_statistic,
        p_value,
        interpretation,
        group_means,
    ) = calculate_statistics(pandas_clean)
    print("[기술통계: 평균·표준편차·분위수]")
    print(descriptive.round(4))
    print("\n[상관계수]")
    print(correlation.round(4))
    print("\n[Welch's t-test]")
    print(f"<=50K 평균: {group_means['<=50K']:.4f}")
    print(f">50K 평균: {group_means['>50K']:.4f}")
    print(f"t-statistic: {t_statistic:.6f}")
    print(f"p-value: {format_p_value(p_value)}")
    print(interpretation)

    section("5. ML Pipeline 학습·평가")
    (
        _pipeline,
        accuracy,
        f1,
        classification_text,
        model_path,
        train_rows,
        test_rows,
    ) = train_and_save_model(pandas_clean, output_dir)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (>50K): {f1:.4f}")
    print("\n[classification report]")
    print(classification_text)
    print(f"모델 저장: {model_path}")

    section("6. report.md 자동 생성")
    report_path = create_report(
        output_dir=output_dir,
        data_path=data_path,
        comparison=comparison,
        missing_before=pandas_missing_before,
        missing_after=pandas_missing_after,
        modes=pandas_modes,
        descriptive=descriptive,
        correlation=correlation,
        t_statistic=t_statistic,
        p_value=p_value,
        ttest_interpretation=interpretation,
        group_means=group_means,
        accuracy=accuracy,
        f1=f1,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    print(f"보고서 저장: {report_path}")
    print("\n전체 분석이 정상적으로 완료되었습니다.")


if __name__ == "__main__":
    main()
