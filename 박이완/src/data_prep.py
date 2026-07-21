"""
data_prep.py — 데이터 준비 모듈  (팀 SPEC §1·§3 준수)

핵심 정책 (팀 SPEC 반영)
  - SPEC §1 : na_values='?' (공백 없이!). 결측 컬럼(workclass·occupation·
              native-country)은 삭제하지 않고 컬럼별 최빈값(mode)으로 대체
  - SPEC §3 : income 값에 앞뒤 공백이 섞여 있을 수 있으므로 매핑 전에 strip
  - Pandas / Polars 로딩 시간 비교 후 이후 분석은 Pandas DataFrame 사용
"""

import time
import urllib.request
from pathlib import Path

import pandas as pd

# 강의 자료(p119)에 안내된 데이터셋과 컬럼 정의
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"]

# SPEC §1 : 결측이 발생하는 3개 컬럼 (전체의 5~7%). 최빈값으로 대체
NA_COLS = ["workclass", "occupation", "native-country"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "adult.data"


def download_if_needed() -> Path:
    """data/adult.data 가 없으면 URL 에서 내려받는다 (있으면 캐시 재사용)."""
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PATH.exists():
        print(f"[data] 캐시 사용: {RAW_PATH}")
        return RAW_PATH
    try:
        print(f"[data] 다운로드 중... {URL}")
        urllib.request.urlretrieve(URL, RAW_PATH)
        print(f"[data] 다운로드 완료 → {RAW_PATH}")
    except Exception as e:
        # 네트워크 오류 시 원인을 명확히 알려주고 중단 (데이터 없이는 진행 불가)
        raise RuntimeError(
            f"데이터 다운로드 실패: {e}\n"
            f"→ 네트워크 확인 후 재실행하거나, {RAW_PATH} 위치에 파일을 직접 두세요."
        ) from e
    return RAW_PATH


def load_pandas(path: Path):
    """Pandas 로딩. (DataFrame, 소요 시간) 반환.
    SPEC §1 : na_values='?' 사용. skipinitialspace 로 원본 앞 공백을 먼저 제거해야
    ' ?' → '?' 가 되어 결측치가 정상 감지된다."""
    t0 = time.perf_counter()
    df = pd.read_csv(path, header=None, names=COLS,
                     na_values="?", skipinitialspace=True)
    return df, time.perf_counter() - t0


def load_polars(path: Path):
    """Polars 로딩. (DataFrame, 소요 시간) 반환. 미설치 시 (None, None)."""
    try:
        import polars as pl
    except ImportError:
        print("[경고] polars 미설치 → pip install polars (Polars 비교는 건너뜀)")
        return None, None
    t0 = time.perf_counter()
    # 원본 값 앞에 공백이 붙어 있어(예: " State-gov") 문자열 컬럼을 먼저 strip
    df = pl.read_csv(path, has_header=False, new_columns=COLS,
                     null_values=["?", " ?"])
    df = df.with_columns(pl.col(pl.Utf8).str.strip_chars())
    return df, time.perf_counter() - t0


def prepare():
    """전체 데이터 준비 절차 실행. (정제된 df, 리포트용 요약 dict) 반환."""
    path = download_if_needed()

    # --- 1) Pandas vs Polars 로딩 비교 -------------------------------------
    df, t_pd = load_pandas(path)
    pl_df, t_pl = load_polars(path)
    print("\n[data] 로딩 결과 비교")
    print(f"  Pandas : shape={df.shape}, {t_pd:.3f}초")
    if pl_df is not None:
        print(f"  Polars : shape={pl_df.shape}, {t_pl:.3f}초")
        same = (pl_df.shape == df.shape)
        print(f"  → 두 도구의 (행, 열) 일치 여부: {same}")

    # --- 2) 결측치·중복 처리 (SPEC §1) --------------------------------------
    na_before = df.isnull().sum()
    dup_cnt = int(df.duplicated().sum())
    print("\n[data] 컬럼별 결측치 (상위 5개)")
    print(na_before.sort_values(ascending=False).head())
    print(f"[data] 중복 행: {dup_cnt}건")

    # 중복 제거
    df = df.drop_duplicates().reset_index(drop=True)

    # SPEC §1 : 결측치는 삭제하지 않고 컬럼별 최빈값(mode)으로 대체
    #   → 팀원 간 학습 데이터 규모(행 수)를 동일하게 맞추기 위한 규칙
    mode_values = {}
    for col in NA_COLS:
        # mode() 는 여러 값을 반환할 수 있으므로 iloc[0] 로 첫 번째 값만 안전 사용
        mode_val = df[col].mode(dropna=True).iloc[0]
        df[col] = df[col].fillna(mode_val)
        mode_values[col] = str(mode_val)
    print(f"[data] 결측치 최빈값 대체 완료: {mode_values}")

    # --- 3) income 타깃 정리 (SPEC §3) --------------------------------------
    # 원본 값에 앞뒤 공백이 섞여 있어 " <=50K" 처럼 저장돼 있을 수 있음
    # → 매핑·비교 전에 반드시 strip() 을 걸어 두어야 t-test·모델이 정확히 동작
    df["income"] = df["income"].astype(str).str.strip()

    # --- 4) 기본 EDA --------------------------------------------------------
    print(f"\n[data] 정제 후 shape: {df.shape}")
    print("[data] 최종 결측 잔여:", int(df.isnull().sum().sum()), "건")
    print("[data] income 분포:")
    print(df["income"].value_counts())

    summary = {
        "shape_pandas": df.shape,
        "shape_polars": None if pl_df is None else tuple(pl_df.shape),
        "load_sec_pandas": round(t_pd, 3),
        "load_sec_polars": None if t_pl is None else round(t_pl, 3),
        "na_total_before": int(na_before.sum()),
        "na_by_col_before": {c: int(na_before[c]) for c in NA_COLS},
        "mode_values": mode_values,
        "dup_removed": dup_cnt,
        "income_counts": df["income"].value_counts().to_dict(),
    }
    return df, summary
