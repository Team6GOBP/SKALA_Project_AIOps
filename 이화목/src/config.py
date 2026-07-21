"""프로젝트 전역 설정 모듈.

경로(데이터/산출물), 컬럼 상수, 재현성 고정값을 한 곳에서 관리한다.
다른 모듈은 매직 넘버/문자열을 직접 쓰지 말고 반드시 이 모듈의 상수를 import해서 쓴다.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Adult_Census_Income.csv"
DATA_PATH = Path(os.environ.get("CENSUS_DATA_PATH", _DEFAULT_DATA_PATH))

# [팀 SPEC] 산출물 경로는 프로젝트 루트 직속으로 고정
SEABORN_CHART_PATH = PROJECT_ROOT / "eda_chart_seaborn.png"
PLOTLY_CHART_PATH = PROJECT_ROOT / "eda_chart_plotly.html"
MODEL_PATH = PROJECT_ROOT / "model.joblib"
REPORT_PATH = PROJECT_ROOT / "report.md"

# ---------------------------------------------------------------------------
# 컬럼 상수
# ---------------------------------------------------------------------------
ALL_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

# 결측치가 "?" 로 표기되는 컬럼 (최빈값으로 대체 대상)
MISSING_VALUE_TOKEN = "?"
COLUMNS_WITH_MISSING = ["workclass", "occupation", "native_country"]

# [팀 SPEC] 상관계수 대상 수치형 컬럼 5개 고정
# fnlwgt는 인구총조사 표본 가중치라 소득과 직접적 인과·상관 의미가 없으므로 제외 (팀 SPEC)
NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# [팀 SPEC] ML Pipeline feature 고정
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
# fnlwgt는 feature에서도 제외 (팀 SPEC)
TARGET_COL = "income"

# ---------------------------------------------------------------------------
# 재현성 고정값 [팀 SPEC]
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
