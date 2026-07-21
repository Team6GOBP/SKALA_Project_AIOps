"""데이터 로딩 및 정제 모듈.

Pandas/Polars 로딩 비교, 기초 EDA 출력, 결측치(최빈값 대체)·중복 제거를 담당한다.
단독 실행(`python src/data_prep.py`)과 패키지 import(`from src.data_prep import ...`)
둘 다 지원한다.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PATH, MISSING_VALUE_TOKEN, COLUMNS_WITH_MISSING

import pandas as pd
import polars as pl


def load_with_pandas(path=DATA_PATH) -> pd.DataFrame:
    """Pandas로 CSV를 로딩하고 소요 시간을 출력한다.

    Parameters
    ----------
    path : str | Path
        읽어들일 CSV 경로.

    Returns
    -------
    pd.DataFrame
        로딩된 원본 데이터프레임 (결측치는 아직 정제 전 상태, "?" 문자열 그대로).

    Raises
    ------
    FileNotFoundError
        path에 파일이 존재하지 않을 때.
    """
    start = time.perf_counter()
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    elapsed = time.perf_counter() - start
    print(f"[Pandas] 로딩 완료 : shape={df.shape}, 소요시간={elapsed:.4f}초")
    return df


def load_with_polars(path=DATA_PATH) -> pl.DataFrame:
    """Polars로 CSV를 로딩하고 소요 시간을 출력한다.

    Parameters
    ----------
    path : str | Path
        읽어들일 CSV 경로.

    Returns
    -------
    pl.DataFrame
        로딩된 원본 데이터프레임.

    Raises
    ------
    FileNotFoundError
        path에 파일이 존재하지 않을 때.
    """
    start = time.perf_counter()
    try:
        df = pl.read_csv(path)
    except FileNotFoundError:
        raise
    elapsed = time.perf_counter() - start
    print(f"[Polars] 로딩 완료 : shape={df.shape}, 소요시간={elapsed:.4f}초")
    return df


def compare_pandas_polars(df_pd: pd.DataFrame, df_pl: pl.DataFrame) -> None:
    """Pandas/Polars 로딩 결과를 비교해서 출력한다.

    shape가 동일한지, 대표 컬럼(`age`) 합계가 두 도구에서 일치하는지 검증한다.

    Parameters
    ----------
    df_pd : pd.DataFrame
        Pandas로 로딩한 데이터프레임.
    df_pl : pl.DataFrame
        Polars로 로딩한 데이터프레임.
    """
    print("=" * 60)
    print("Pandas vs Polars 로딩 결과 비교")
    print("=" * 60)
    print(f"Pandas shape : {df_pd.shape}")
    print(f"Polars shape : {df_pl.shape}")
    same_shape = df_pd.shape == df_pl.shape
    print(f"shape 일치 여부 : {same_shape}")

    pd_age_sum = df_pd["age"].sum()
    pl_age_sum = df_pl["age"].sum()
    print(f"age 합계 (Pandas) : {pd_age_sum}")
    print(f"age 합계 (Polars) : {pl_age_sum}")
    print(f"age 합계 일치 여부 : {pd_age_sum == pl_age_sum}")


def basic_eda(df: pd.DataFrame) -> None:
    """기초 EDA 정보를 출력한다.

    `df.info()`, 컬럼별 결측치 개수, 중복행 개수, "?" 로 표기된 결측치
    개수(컬럼별)를 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        EDA를 수행할 데이터프레임 (정제 전).
    """
    print("=" * 60)
    print("기초 EDA")
    print("=" * 60)
    df.info()

    print("-" * 60)
    print("컬럼별 결측치(NaN) 개수")
    print(df.isnull().sum())

    print("-" * 60)
    n_dup = df.duplicated().sum()
    print(f"중복행 개수 : {n_dup}")

    print("-" * 60)
    print(f"'{MISSING_VALUE_TOKEN}' 문자열로 표기된 결측치 개수 (컬럼별)")
    for col in COLUMNS_WITH_MISSING:
        count = (df[col].astype(str).str.strip() == MISSING_VALUE_TOKEN).sum()
        print(f"  {col} : {count}건")

    print("-" * 60)
    print("이상치 존재 여부 (참고용, 제거하지 않음)")
    n_age_outlier = df[(df["age"] < 17) | (df["age"] > 90)].shape[0]
    n_hours_outlier = df[(df["hours_per_week"] <= 0) | (df["hours_per_week"] > 99)].shape[0]
    print(f"  age 이상치(17미만/90초과) : {n_age_outlier}건")
    print(f"  hours_per_week 이상치(0이하/99초과) : {n_hours_outlier}건")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 대체 및 중복 제거를 수행한다.

    [팀 SPEC] 결측 표기 "?" 는 각 컬럼의 최빈값(mode)으로 대체한다 (행 삭제 금지).
    중복행은 `drop_duplicates()`로 제거한다. 비정상 값(이상치)은 팀 SPEC에
    명시되어 있지 않으므로 제거하지 않고 그대로 둔다 (basic_eda에서 개수만 출력).

    Parameters
    ----------
    df : pd.DataFrame
        정제 전 원본 데이터프레임.

    Returns
    -------
    pd.DataFrame
        정제된 데이터프레임 (결측치 최빈값 대체, 중복행 제거 완료).
    """
    print("=" * 60)
    print("데이터 정제 시작")
    print("=" * 60)

    df = df.copy()

    # ⚠️ 값에 섞인 공백을 먼저 제거한 뒤 "?" 매칭 (공백 포함 " ?" 매칭 버그 방지)
    for col in COLUMNS_WITH_MISSING:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace(MISSING_VALUE_TOKEN, pd.NA)

    print("결측치 최빈값 대체")
    missing_counts = {}
    for col in COLUMNS_WITH_MISSING:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_value = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_value)
            print(f"  {col} : {n_missing}건 -> 최빈값 '{mode_value}' 로 대체")
        else:
            print(f"  {col} : 결측치 없음")
        missing_counts[col] = int(n_missing)

    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    rows_after_dedup = len(df)
    print(f"중복 제거 : {rows_before_dedup}행 -> {rows_after_dedup}행 "
          f"({rows_before_dedup - rows_after_dedup}건 제거)")

    df.attrs["missing_counts"] = missing_counts

    print("정제 완료")
    return df


def prepare_data() -> tuple:
    """데이터 로딩 비교 -> EDA -> 정제 순서로 전체 준비 과정을 실행한다.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (정제된 데이터프레임, 정제 전 원본 행 수)

    Raises
    ------
    SystemExit
        데이터 파일이 존재하지 않을 때 안내 메시지를 출력하고 종료한다.
    """
    try:
        df_pd = load_with_pandas(DATA_PATH)
        df_pl = load_with_polars(DATA_PATH)
    except FileNotFoundError:
        print(f"[오류] 데이터 파일을 찾을 수 없습니다 : {DATA_PATH}")
        print("       data/raw/ 에 Adult_Census_Income.csv 를 배치하거나 "
              ".env의 CENSUS_DATA_PATH를 설정하세요.")
        sys.exit(1)

    compare_pandas_polars(df_pd, df_pl)
    basic_eda(df_pd)

    rows_before = len(df_pd)
    df_clean = clean_data(df_pd)

    return df_clean, rows_before


if __name__ == "__main__":
    cleaned_df, original_rows = prepare_data()
    print("=" * 60)
    print(f"prepare_data() 단독 실행 결과 : 원본 {original_rows}행 -> 정제 후 {len(cleaned_df)}행")
