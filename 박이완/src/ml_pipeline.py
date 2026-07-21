"""
ml_pipeline.py — ML 파이프라인 모듈  (팀 SPEC 3·4·5 준수)

핵심 정책 (팀 SPEC 반영)
  - SPEC 3 : income 매핑 {'<=50K': 0, '>50K': 1} (strip 은 data_prep 에서 완료)
              평가지표는 accuracy + F1-score 둘 다 출력
  - SPEC 4 : feature = 수치형 5개(NUM_COLS) + 범주형 8개(CAT_COLS)
              fnlwgt 는 상관·모델 양쪽 모두에서 제외
              공통 설정: test_size=0.2, random_state=42
  - SPEC 5 : 저장 파일명 model.joblib
  - 병합 기준 4번 (데이터 누수 없음) : income 은 y 로만 사용, X 에 포함하지 않음
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
MODEL_PATH = OUT_DIR / "model.joblib"   # SPEC 5 파일명

# SPEC 4 : 수치형 5개 (fnlwgt 제외)
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss",
            "hours-per-week"]
# SPEC 4 : 범주형 전체 (native-country 포함, fnlwgt·income 은 여기 없음)
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]


def train_and_save(df: pd.DataFrame) -> dict:
    """Pipeline 을 학습·평가·저장하고 리포트용 지표 dict 를 반환한다."""
    OUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print("[ml] sklearn Pipeline — 전처리 + 모델 학습")
    print("=" * 60)

    # SPEC 3 : 명시적 매핑으로 타깃 인코딩 (data_prep 에서 strip 완료 상태)
    #    dict.get 매핑을 쓰면 예상 밖 값은 NaN 이 되어 사전에 감지 가능
    y = df["income"].map({"<=50K": 0, ">50K": 1})
    if y.isnull().any():
        # SPEC 대로라면 이 지점에서 NaN 이 나오면 안 되므로, 조기 실패로 원인 노출
        raise ValueError(
            f"income 매핑에서 예상 밖 값 감지: {df.loc[y.isnull(),'income'].unique()}"
        )
    y = y.astype(int)

    # 병합 기준 4번 (데이터 누수 방지) :
    #   income 은 y 로만 사용하고 X 에 포함시키지 않는다.
    #   fnlwgt 는 SPEC 4 에 따라 X 에서 제외.
    X = df[NUM_COLS + CAT_COLS]

    # 전처리(수치 표준화 / 범주 원핫)와 모델을 하나의 Pipeline 객체로 통합
    preproc = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([
        ("prep", preproc),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # SPEC 4 : test_size=0.2, random_state=42 로 팀원 간 성능 비교 가능하게 통일
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # SPEC 3 : accuracy + F1 둘 다 출력
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    print(f"  정확도(accuracy) = {acc:.3f}")
    print(f"  F1-score         = {f1:.3f}")

    # joblib 으로 모델 저장 + 재로딩 후 예측 일치 여부 검증
    joblib.dump(model, MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    same = bool(np.array_equal(loaded.predict(X_te[:100]), pred[:100]))
    print(f"  모델 저장 → {MODEL_PATH.name} / 재로딩 예측 일치: {same}")

    return {"accuracy": round(acc, 3), "f1": round(f1, 3),
            "model_file": MODEL_PATH.name, "reload_ok": same,
            "n_train": len(X_tr), "n_test": len(X_te),
            "n_features_num": len(NUM_COLS),
            "n_features_cat": len(CAT_COLS)}
