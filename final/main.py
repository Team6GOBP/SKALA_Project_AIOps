"""
[Day2 종합실습] 팀 통합본 (final/) - Adult Census Income End2End 분석

이 스크립트는 팀원 6명(강주현·김현수·박이완·이화목·임준혁·한태우)이 spec.md 규칙에 따라
각자 독립적으로 완성한 구현을 PM이 항목별로 비교해 가장 좋은 부분만 조합한 최종본이다.
각 함수 docstring에 어느 팀원의 구현을 기반으로 했는지 남겨두었다 (자세한 근거는
MERGE_NOTES.md 참고).

실행 순서:
  Step 1 : Pandas·Polars 로딩 비교 (shape·dtype·결측치·중복행)
  Step 2 : 결측치 최빈값 대체, 중복행 제거
  Step 3 : 기본 EDA (describe/info/범주형 분포/income 클래스 비율)
  Step 4 : 시각화 - Seaborn(그룹비교+상관관계 2-패널) · Plotly(그룹비교, 인터랙티브)
  Step 5 : 통계 분석 - 상관계수 + income 그룹 Welch's t-test
  Step 6 : sklearn ColumnTransformer + Pipeline (RandomForestClassifier), accuracy·F1, joblib 저장
  Step 7 : report.md 자동 생성

spec.md 준수 사항: na_values="?"(공백 없이), 결측 컬럼은 삭제 대신 최빈값 대체,
상관계수/모델 feature는 fnlwgt 제외, t-test는 income×hours-per-week(Welch),
ML feature=수치형 5개+범주형 8개, test_size=0.2/random_state=42/stratify=income.
"""
from __future__ import annotations

import time
import urllib.request
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 상수 (spec.md 4번 규칙 고정 컬럼)
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).parent
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_PATH = OUT_DIR / "data" / "adult.data"
SEABORN_CHART_PATH = OUT_DIR / "eda_chart_seaborn.png"
PLOTLY_CHART_PATH = OUT_DIR / "eda_chart_plotly.html"
MODEL_PATH = OUT_DIR / "model.joblib"
REPORT_PATH = OUT_DIR / "report.md"
RANDOM_STATE = 42

COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
MISSING_COLS = ["workclass", "occupation", "native-country"]
# fnlwgt는 인구총조사 가중치라 소득과 직접적 상관/인과 의미가 없어 상관계수·모델 feature에서 제외
CORR_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def set_korean_font() -> None:
    """[강주현 기반] OS별 한글 폰트를 자동 탐색해 차트 라벨이 깨지지 않게 한다."""
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 폰트를 찾지 못해 기본 폰트로 진행합니다. (라벨이 깨질 수 있음)")
    plt.rcParams["axes.unicode_minus"] = False


def _df_to_markdown(df: pd.DataFrame) -> str:
    """[강주현 기반] tabulate 패키지 없이 직접 마크다운 표를 만든다.
    (.to_markdown()이 요구하는 tabulate 의존성이 없는 환경에서 report.md 생성이
    통째로 실패하는 known pitfall을 원천적으로 피하기 위함 — 강주현·임준혁 둘 다
    이 방식을 썼고, 그중 재사용하기 쉬운 강주현 버전을 채택했다.)"""
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


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"[{context}] 필수 컬럼이 없습니다: {missing}")


# ---------------------------------------------------------------------------
# Step 1. 데이터 로딩 (Pandas vs Polars 비교) — [강주현 기반, 채택 사유는 MERGE_NOTES 1번]
# ---------------------------------------------------------------------------
def download_if_needed(path: Path = DATA_PATH) -> Path:
    """[박이완 기반] 원본 데이터는 .gitignore 정책상 저장소에 커밋하지 않으므로,
    로컬에 캐시된 파일이 있으면 재사용하고 없으면 UCI에서 내려받는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[data] 로컬 캐시 사용: {path}")
        return path
    print(f"[data] 로컬 파일이 없어 다운로드합니다: {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, path)
        print(f"[data] 다운로드 완료 → {path}")
    except Exception as exc:
        raise RuntimeError(
            f"데이터 다운로드 실패: {exc}\n"
            f"→ 네트워크를 확인하거나 {path} 위치에 adult.data 파일을 직접 두세요."
        ) from exc
    return path


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Pandas 로딩. skipinitialspace=True는 필드 값의 앞 공백을 먼저 제거한 뒤
    na_values와 비교하므로, na_values는 반드시 공백 없이 "?"로 지정해야 한다.
    na_values=" ?"로 쓰면 매칭이 안 돼 결측치가 하나도 안 잡히는 조용한 버그가 생긴다."""
    return pd.read_csv(
        path, header=None, names=COLS, na_values="?", skipinitialspace=True
    )


def load_with_polars(path: Path) -> pl.DataFrame:
    """Polars 로딩.

    주의 1: UCI 원본은 콤마 뒤에 공백이 섞여 있는데(예: "39, State-gov, 77516"),
    Polars에는 Pandas의 skipinitialspace 같은 옵션이 없다. 공백이 낀 채로 숫자
    컬럼을 인식하려 하면 파싱에 실패해 해당 컬럼 전체가 String으로 떨어진다.
    그래서 일단 전부 문자열(infer_schema_length=0)로 읽은 뒤 공백을 strip하고
    수치형 컬럼만 명시적으로 cast한다.

    주의 2: UCI 원본 파일 끝에 빈 줄이 있다. Pandas는 자동으로 건너뛰지만 Polars는
    값이 전부 null인 한 행으로 읽어 행 수가 +1이 된다. 그래서 모든 컬럼이 비어 있는
    행을 명시적으로 걸러낸다.
    """
    df = pl.read_csv(path, has_header=False, new_columns=COLS, infer_schema_length=0)
    df = df.with_columns([pl.col(c).str.strip_chars() for c in COLS])
    df = df.filter(
        ~pl.all_horizontal([pl.col(c).is_null() | (pl.col(c) == "") for c in COLS])
    )
    df = df.with_columns([
        pl.when(pl.col(c) == "?").then(None).otherwise(pl.col(c)).alias(c)
        for c in COLS
    ])
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in NUMERIC_COLS])
    return df


def compare_loading(path: Path) -> tuple[pd.DataFrame, dict]:
    """Pandas·Polars 로딩 결과를 비교하고, 두 라이브러리가 같은 shape·결측치·중복행
    수를 내는지 assert로 강하게 검증한다 (임준혁의 dual-library 교차검증 아이디어 채택)."""
    t0 = time.perf_counter()
    df_pd = load_with_pandas(path)
    t_pandas = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_pl = load_with_polars(path)
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
    polars_nulls = df_pl.null_count()
    missing_polars = {
        col: int(polars_nulls[col][0]) for col in COLS if int(polars_nulls[col][0]) > 0
    }
    print("\n=== 결측치 개수 ===")
    print("[Pandas]\n", missing_pandas)
    print("[Polars]\n", missing_polars)

    dup_pandas = int(df_pd.duplicated().sum())
    dup_polars = len(df_pl) - df_pl.unique().height
    print("\n=== 중복행 개수 ===")
    print(f"Pandas : {dup_pandas}건 / Polars : {dup_polars}건")

    # 트레일링 빈 줄·콤마+공백 파싱 문제를 제대로 처리했다면 두 라이브러리 결과가 일치해야 한다.
    assert df_pd.shape == df_pl.shape, (
        f"Pandas/Polars shape 불일치: {df_pd.shape} vs {df_pl.shape} "
        "(trailing blank line 처리 누락 가능성)"
    )
    assert dup_pandas == dup_polars, "Pandas/Polars 중복행 수 불일치"
    assert missing_pandas.to_dict() == missing_polars, "Pandas/Polars 결측치 수 불일치"

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
# Step 2. 결측치·중복 처리 (spec.md 규칙: 최빈값 대체, 삭제 X)
# ---------------------------------------------------------------------------
def handle_missing_and_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """[강주현 기반] 결측치는 최빈값 대체(삭제 X), 중복행은 제거한다."""
    df_clean = df.copy()

    before_missing = df_clean[MISSING_COLS].isnull().sum()
    mode_values = {}
    for col in MISSING_COLS:
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
    print(f"\n=== 중복행 제거 ===\n{before_rows:,}행 -> {len(df_clean):,}행 ({removed}건 제거)")

    # 결측치가 실제로 전부 채워졌는지, 삭제가 아닌 대체였는지 확실히 검증
    assert df_clean[MISSING_COLS].isnull().sum().sum() == 0, "결측치가 완전히 대체되지 않았습니다."

    stats = {
        "missing_before": before_missing.to_dict(),
        "mode_values": mode_values,
        "rows_before": before_rows,
        "rows_after": len(df_clean),
        "duplicates_removed": removed,
    }
    return df_clean, stats


def run_eda(df: pd.DataFrame) -> dict:
    """[강주현 기반] describe/info/범주형 분포/income 클래스 비율을 출력한다."""
    print("\n=== describe() : 수치형 컬럼 기술통계 ===")
    print(df.describe())

    print("\n=== info() : 타입·결측치·메모리 ===")
    df.info()  # info()는 자체 출력하고 None을 반환하므로 print()로 감싸지 않는다

    print("\n=== 범주형 컬럼 값 분포 (상위 5개씩) ===")
    for col in df.select_dtypes(include=["object"]).columns:
        print(f"\n[{col}]")
        print(df[col].value_counts().head(5))

    print("\n=== income 클래스 비율 (불균형 확인용) ===")
    income_ratio = df["income"].value_counts(normalize=True) * 100
    print(income_ratio.round(2).astype(str) + "%")
    print("(참고: 불균형 때문에 ML 평가에서 accuracy만 보면 착시가 생길 수 있어 F1도 같이 본다)")

    return income_ratio.round(2).to_dict()


# ---------------------------------------------------------------------------
# Step 4. 시각화 — [임준혁 기반, 채택 사유는 MERGE_NOTES 2번]
# ---------------------------------------------------------------------------
def plot_eda_charts(df: pd.DataFrame) -> None:
    """Seaborn 정적 차트(2-패널: income별 근무시간 박스플롯 + 수치형 상관관계
    히트맵)와 Plotly 인터랙티브 차트(학력×income별 평균 근무시간 그룹 막대)를 저장한다.
    두 차트 모두 income과 직접 연결된 인사이트를 보여주도록 설계했다."""
    _require_columns(df, CORR_COLS + ["education", "income"], "plot_eda_charts")

    sns.set_theme(style="whitegrid", context="notebook")
    figure, (box_axis, heatmap_axis) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(
        data=df, x="income", y="hours-per-week", hue="income", legend=False,
        palette={"<=50K": "#4C78A8", ">50K": "#F58518"}, showfliers=False, ax=box_axis,
    )
    box_axis.set_title("Weekly Work Hours by Income Group", fontsize=14, pad=12)
    box_axis.set_xlabel("Income Group")
    box_axis.set_ylabel("Hours per Week")

    correlation = df[CORR_COLS].corr(method="pearson")
    sns.heatmap(
        correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
        square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=heatmap_axis,
    )
    heatmap_axis.set_title("Correlation of Numeric Features", fontsize=14, pad=12)
    heatmap_axis.set_xlabel("Feature")
    heatmap_axis.set_ylabel("Feature")

    figure.tight_layout()
    figure.savefig(SEABORN_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"[Seaborn] 저장 완료: {SEABORN_CHART_PATH}")

    plot_data = (
        df.groupby(["education", "income"], as_index=False, observed=True)
        .agg(mean_hours=("hours-per-week", "mean"),
             sample_count=("hours-per-week", "size"),
             education_num=("education-num", "median"))
        .sort_values(["education_num", "income"])
    )
    fig_plotly = px.bar(
        plot_data, x="education", y="mean_hours", color="income", barmode="group",
        hover_data={"sample_count": True, "education_num": False},
        category_orders={
            "education": plot_data["education"].drop_duplicates().tolist(),
            "income": ["<=50K", ">50K"],
        },
        labels={"education": "Education Level", "mean_hours": "Average Hours per Week",
                "income": "Income Group", "sample_count": "Sample Count"},
        title="Average Weekly Work Hours by Education and Income Group",
    )
    fig_plotly.update_layout(
        template="plotly_white", xaxis_tickangle=-35, legend_title_text="Income Group",
        margin={"l": 60, "r": 30, "t": 80, "b": 120},
    )
    fig_plotly.write_html(PLOTLY_CHART_PATH, include_plotlyjs=True, full_html=True)
    print(f"[Plotly] 저장 완료: {PLOTLY_CHART_PATH}")


# ---------------------------------------------------------------------------
# Step 5. 통계 분석 — [강주현 기반 report 문구, 채택 사유는 MERGE_NOTES 3번]
# ---------------------------------------------------------------------------
def run_statistical_tests(df: pd.DataFrame) -> dict:
    """기술통계·상관계수 산출 후, income 그룹(<=50K vs >50K)의 hours-per-week
    평균 차이를 Welch's t-test로 검정하고 p-value를 해석한다."""
    _require_columns(df, CORR_COLS + ["income"], "run_statistical_tests")

    describe_df = df[CORR_COLS].describe()
    print("=== 기술통계 ===")
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

    # equal_var=False: 두 그룹의 분산이 같다고 가정하지 않는 Welch's t-test (spec.md 2번)
    t_stat, p_value = ttest_ind(under_50k, over_50k, equal_var=False)
    significant = p_value < 0.05
    print(f"<=50K 평균: {under_50k.mean():.2f} (n={len(under_50k):,})")
    print(f">50K  평균: {over_50k.mean():.2f} (n={len(over_50k):,})")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4g}")
    verdict = "통계적으로 유의미하다" if significant else "통계적으로 유의미하지 않다"
    print(f"해석: p-value {'< 0.05' if significant else '>= 0.05'} → 두 income 그룹의 평균 근무시간 차이는 {verdict}.")

    result["ttest"] = {
        "under_50k_mean": under_50k.mean(), "under_50k_n": len(under_50k),
        "over_50k_mean": over_50k.mean(), "over_50k_n": len(over_50k),
        "t_stat": t_stat, "p_value": p_value, "significant": significant,
    }
    return result


# ---------------------------------------------------------------------------
# Step 6. sklearn Pipeline — [강주현 모델·전처리 골격 + 임준혁 결측 방어 로직 결합,
# 채택 사유는 MERGE_NOTES 4번]
# ---------------------------------------------------------------------------
def build_and_save_pipeline(df: pd.DataFrame) -> dict:
    """income을 0/1로 인코딩해 분류 문제로 풀고, 수치형 5개+범주형 8개(fnlwgt 제외)를
    feature로 사용한다. accuracy·F1을 출력하고 joblib으로 저장한 뒤, 저장된 모델을
    다시 불러와 예측 확률이 np.allclose로 일치하는지 검증한다 (n_jobs=-1 병렬 학습은
    재로딩 후 부동소수점 미세 오차가 생길 수 있어 `==` 대신 `np.allclose`를 쓴다)."""
    _require_columns(df, CORR_COLS + CATEGORICAL_FEATURES + ["income"], "build_and_save_pipeline")

    data = df.copy()
    # spec.md 3번: 매핑 전 .str.strip() 먼저 적용
    income_stripped = data["income"].astype(str).str.strip()
    data["income_binary"] = income_stripped.map({"<=50K": 0, ">50K": 1})
    if data["income_binary"].isna().any():
        unknown = sorted(income_stripped[data["income_binary"].isna()].unique())
        raise ValueError(f"income 타깃에 알 수 없는 값이 있습니다: {unknown}")

    feature_cols = CORR_COLS + CATEGORICAL_FEATURES
    X = data[feature_cols]
    y = data["income_binary"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 결측치는 Step 2에서 이미 대체됐지만, feature 서브셋 재사용 시의 방어적 안전장치로
    # SimpleImputer를 파이프라인 안에 포함시켰다 (임준혁 아이디어 채택).
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, CORR_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("=== sklearn Pipeline (RandomForestClassifier) 평가 ===")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1-score = {f1:.4f}")
    print("(참고: income 불균형(약 75:25) 때문에 accuracy만으로는 소수 클래스 예측력을 알 수 없어 F1도 같이 본다)")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

    # joblib 재로딩 검증: n_jobs=-1 병렬 학습 결과는 재로딩 후 예측 확률에 부동소수점
    # 미세 오차가 생길 수 있으므로 `==` 대신 `np.allclose`로 비교한다.
    reloaded = joblib.load(MODEL_PATH)
    proba_original = pipeline.predict_proba(X_test)
    proba_reloaded = reloaded.predict_proba(X_test)
    assert np.allclose(proba_original, proba_reloaded), "재로딩된 모델의 예측 확률이 원본과 다릅니다."
    print("[검증] 저장 후 재로딩한 모델의 예측 확률이 np.allclose 기준으로 원본과 일치합니다.")

    return {
        "accuracy": acc, "f1": f1, "model_path": str(MODEL_PATH),
        "n_train": len(X_train), "n_test": len(X_test), "features": feature_cols,
    }


# ---------------------------------------------------------------------------
# Step 7. report.md 자동 생성 — [강주현 기반, 채택 사유는 MERGE_NOTES 5번]
# ---------------------------------------------------------------------------
def generate_report(
    loading_stats: dict, clean_stats: dict, income_ratio: dict,
    stats_result: dict, model_result: dict,
) -> None:
    """Step 1~6 결과를 모아 report.md를 자동 생성한다. "핵심 인사이트" 섹션은
    하드코딩된 문장이 아니라 실제 계산값에서 동적으로 만든다 (예: 어느 income
    그룹의 근무시간이 더 긴지는 실제 평균을 비교해서 판단하지, 미리 문장에
    박아 넣지 않는다)."""
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

    content = f"""\
# Day2 종합실습 - Adult Census Income 분석 리포트 (팀 통합본)

이 리포트는 `final/main.py` 실행 결과를 바탕으로 자동 생성되었습니다.
팀원 6명의 구현을 spec.md 기준으로 비교해 항목별 최우수 구현을 조합한 최종본입니다
(세부 채택 근거는 `MERGE_NOTES.md` 참고).
데이터셋: [Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

## 1. 데이터 준비

### 1-1. Pandas vs Polars 로딩 비교

| | Pandas | Polars |
|---|---|---|
| 로딩 시간 | {loading_stats['t_pandas']:.4f}초 | {loading_stats['t_polars']:.4f}초 |
| shape | {loading_stats['shape_pandas']} | {loading_stats['shape_polars']} |
| 중복행 | {loading_stats['dup_pandas']}건 | {loading_stats['dup_polars']}건 |

원본 결측치: {', '.join(f"`{c}` {n}건" for c, n in loading_stats['missing_pandas'].items())}

(Pandas·Polars 결과가 shape·결측치·중복행 수까지 완전히 일치함을 코드 내 assert로 검증했습니다 —
콤마+공백으로 인한 Polars의 숫자 컬럼 String 오분류, UCI 원본 파일 끝 trailing blank line으로 인한
행 수 불일치를 모두 로딩 단계에서 직접 처리했기 때문입니다.)

### 1-2. 결측치·중복 처리 (spec.md 규칙: 최빈값 대체, 중복행 제거)

{missing_lines}

- 중복행 제거: {clean_stats['rows_before']:,}행 → {clean_stats['rows_after']:,}행 ({clean_stats['duplicates_removed']}건 제거)

### 1-3. income 클래스 비율 (불균형 확인)

{income_lines}

> `<=50K`가 다수 클래스라 ML 평가 시 accuracy만으로는 부족해 F1도 함께 확인했고,
> train_test_split에도 stratify=income을 적용했습니다.

## 2. 시각화

| 카테고리 | 라이브러리 | 내용 | 파일 |
|---|---|---|---|
| 그룹비교+상관관계 (필수) | Seaborn | (좌) income별 근무시간 박스플롯 (우) 수치형 5개 상관 히트맵 | [{SEABORN_CHART_PATH.name}]({SEABORN_CHART_PATH.name}) |
| 그룹비교 (필수) | Plotly | 학력×income별 평균 근무시간 그룹 막대 | [{PLOTLY_CHART_PATH.name}]({PLOTLY_CHART_PATH.name}) |

![income별 근무시간 박스플롯 + 상관관계 히트맵]({SEABORN_CHART_PATH.name})

> Plotly 인터랙티브 차트는 이미지로 표시되지 않으니 위 링크를 눌러 직접 열어서 확인하세요.

## 3. 통계 분석

### 3-1. 기술통계 (평균·표준편차·분위수)

{_df_to_markdown(stats_result['describe'])}

### 3-2. 상관계수 행렬

{_df_to_markdown(stats_result['corr'])}

### 3-3. t-test : income(<=50K vs >50K) 그룹의 hours-per-week 평균 차이 (Welch's t-test, equal_var=False)

{ttest_section}

## 4. ML Pipeline

- 모델: `RandomForestClassifier` (ColumnTransformer + Pipeline, 결측 방어용 SimpleImputer 포함)
- feature: {', '.join(f'`{c}`' for c in model_result['features'])} (fnlwgt·income 제외 — 데이터 누수 없음)
- train/test: {model_result['n_train']:,}건 / {model_result['n_test']:,}건 (test_size=0.2, random_state=42, stratify=income)
- **Accuracy = {model_result['accuracy']:.4f}**
- **F1-score = {model_result['f1']:.4f}**
- 모델 저장 경로: `{model_result['model_path']}`
- joblib 재로딩 검증: 저장 후 다시 불러온 모델의 예측 확률이 `np.allclose` 기준으로 원본과 일치함을 확인했습니다.

## 5. 핵심 인사이트

{ttest_insight}
{corr_insight}

---
*이 문서는 `generate_report()` 함수에 의해 자동 생성되었습니다. 수동 편집 없이 스크립트 재실행만으로 갱신됩니다.*
"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"\nreport.md 생성 완료: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def main() -> None:
    set_korean_font()
    data_path = download_if_needed()
    raw_df, loading_stats = compare_loading(data_path)

    clean_df, clean_stats = handle_missing_and_duplicates(raw_df)
    income_ratio = run_eda(clean_df)

    plot_eda_charts(clean_df)
    stats_result = run_statistical_tests(clean_df)
    model_result = build_and_save_pipeline(clean_df)

    generate_report(loading_stats, clean_stats, income_ratio, stats_result, model_result)


if __name__ == "__main__":
    main()
