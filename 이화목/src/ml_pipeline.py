"""ML Pipeline 모듈.

sklearn Pipeline으로 전처리+RandomForestClassifier를 구성해 income(<=50K vs >50K)을
분류한다. fnlwgt와 income 자체는 feature에서 제외해 데이터 누수를 방지한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_PATH,
)

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline() -> Pipeline:
    """전처리(ColumnTransformer) + RandomForestClassifier로 구성된 sklearn Pipeline을 만든다.

    Returns
    -------
    Pipeline
        수치형(StandardScaler) + 범주형(OneHotEncoder) 전처리와
        RandomForestClassifier(n_estimators=200, max_depth=12)를 묶은 파이프라인.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train_and_evaluate(df: pd.DataFrame, model_path=MODEL_PATH) -> dict:
    """Pipeline을 학습하고 테스트셋으로 평가한 뒤 모델을 저장한다.

    [팀 SPEC] X = NUMERIC_FEATURES + CATEGORICAL_FEATURES (fnlwgt, income 미포함
    -- 데이터 누수 방지), train_test_split(test_size=0.2, random_state=42)로 고정 분할.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 데이터프레임.
    model_path : str | Path
        학습된 모델을 저장할 경로. 기본값은 config.MODEL_PATH (프로젝트 루트).

    Returns
    -------
    dict
        {"accuracy":.., "f1":.., "n_train":.., "n_test":..}
    """
    print("=" * 60)
    print("ML Pipeline 학습 및 평가")
    print("=" * 60)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COL].astype(str).str.strip().map({"<=50K": 0, ">50K": 1})

    # 데이터 누수 방지 재확인 : income/fnlwgt가 feature에 섞이지 않았는지 검증
    assert TARGET_COL not in feature_cols, "income이 feature에 포함되면 안 됨 (데이터 누수)"
    assert "fnlwgt" not in feature_cols, "fnlwgt는 feature에서 제외해야 함 (팀 SPEC)"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"train : {len(X_train)}건, test : {len(X_test)}건")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"accuracy : {accuracy:.4f}")
    print(f"f1-score : {f1:.4f}")
    print("classification report :")
    print(classification_report(y_test, y_pred))

    try:
        joblib.dump(pipeline, model_path)
        print(f"모델 저장 완료 : {model_path}")

        reloaded = joblib.load(model_path)
        reloaded_pred = reloaded.predict(X_test)
        reloaded_accuracy = accuracy_score(y_test, reloaded_pred)
        print(f"재로딩 모델 accuracy : {reloaded_accuracy:.4f} "
              f"(원본과 동일 여부 : {reloaded_accuracy == accuracy})")
    except Exception as e:
        print(f"[오류] 모델 저장/재로딩 실패 : {e}")
        raise

    return {
        "accuracy": accuracy,
        "f1": f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_prep import prepare_data

    cleaned_df, _ = prepare_data()
    train_and_evaluate(cleaned_df)
