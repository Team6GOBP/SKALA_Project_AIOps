# 프로젝트 빌드 지시서 — Adult Census Income End2End 분석 프로젝트

이 문서는 CLI 코딩 에이전트(Claude Code 등)가 그대로 읽고 프로젝트 전체를
구현할 수 있도록 작성된 스펙 문서다. 아래 내용을 순서대로, 빠짐없이 구현할 것.

> **우선순위 안내** : 이 문서의 내용 중 "팀 공통 SPEC"으로 표시된 항목은
> 팀 전체가 합의한 확정 규칙이며, 다른 어떤 판단(일반적인 베스트 프랙티스,
> 이 문서의 다른 부분 등)보다 우선한다. 팀원 간 결과 병합·비교가 목적이므로
> 임의로 값을 바꾸지 말고 아래 수치·이름을 그대로 따를 것.

---

## 0. 프로젝트 개요

- **데이터셋** : Adult Census Income (`Adult_Census_Income.csv`, 32,561행 15컬럼)
- **원본 출처(참고용)** :
  ```python
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  cols = ["age","workclass","fnlwgt","education","education-num",
          "marital-status","occupation","relationship","race","sex",
          "capital-gain","capital-loss","hours-per-week","native-country","income"]
  ```
- **⚠️ 컬럼명 주의** : 이 프로젝트에서 실제로 쓰는 `Adult_Census_Income.csv`는
  컬럼명이 이미 **스네이크케이스(언더스코어)**로 정리되어 있다
  (`education_num`, `marital_status`, `capital_gain`, `capital_loss`,
  `hours_per_week`, `native_country`). 팀 SPEC 문서에 나오는 하이픈 표기
  (`education-num` 등)는 원본 UCI raw 데이터 기준 이름이며, **같은 컬럼을
  가리키는 것**이다. 코드에서는 반드시 언더스코어 버전을 쓴다.
- **목표** : Pandas·Polars 로딩 비교 → 결측치/중복 정제 → 시각화(Seaborn+Plotly)
  → 통계 분석(기술통계·상관계수·t-test) → sklearn Pipeline 분류 모델 →
  `report.md` 자동 생성까지 이어지는 End2End 파이프라인 구축.
- **채점 기준(참고용, 100점)**:
  - 데이터 준비 + 시각화 (35점) : Pandas·Polars 모두 사용, 결측치 처리, EDA / Seaborn 정적 + Plotly 인터랙티브 각 1개 이상, 제목·축 레이블 포함
  - 통계분석 + ML Pipeline (45점) : 기술통계·상관계수 출력, t-test 결과와 p-value 해석 포함 / Pipeline 객체로 전처리+모델 구성, 평가지표(정확도·F1 등) 출력, 모델 파일 저장
  - 자동화 + 발표 (20점) : report.md 자동 생성
  - 완성도 (10점) : 주석 처리 누락 시 감점
- **병합(PM) 기준(참고용)** : 병합 담당자가 팀원별 결과물을 아래 5개 기준으로
  비교해 항목별 최선을 골라 `final/`로 합친다. 따라서 아래 각 항목을 구현할 때
  이 기준을 염두에 두고 품질을 높일 것.
  1. 결측치·중복 처리 — 예외 처리가 가장 꼼꼼한 버전
  2. 시각화 — 라벨·제목이 명확하고 인사이트가 잘 드러나는 버전
  3. 통계 검정 — p-value 해석 문장이 정확한 버전
  4. ML Pipeline — **데이터 누수 없는 버전** (타깃 `income`과 직접 연산되는
     컬럼을 feature에 절대 넣지 않을 것 — 이 프로젝트는 애초에
     `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES`에만 feature를 한정하므로
     `income` 자체나 그로부터 파생된 컬럼이 섞이지 않는지 구현 후 반드시
     재확인)
  5. report.md — 자동화 완성도 (수동 편집 없이 `python main.py`만 돌려도
     완전한 리포트가 나와야 함)

---

## 1. 폴더 구조 (반드시 이 구조로 생성)

```
팀원이름/                              # 예: hemu/ (본인 이름으로 된 폴더가 프로젝트 루트)
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                             # 팀 SPEC 지정 파일명 — 이 위치 고정
├── eda_chart_seaborn.png                # 실행 후 생성됨 — 팀 SPEC 지정 파일명/위치 고정
├── eda_chart_plotly.html                 # 실행 후 생성됨 — 팀 SPEC 지정 파일명/위치 고정
├── model.joblib                           # 실행 후 생성됨 — 팀 SPEC 지정 파일명/위치 고정
├── report.md                               # 실행 후 생성됨 — 팀 SPEC 지정 파일명/위치 고정
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── stats_analysis.py
│   ├── visualize.py
│   ├── ml_pipeline.py
│   └── generate_report.py
└── data/
    └── raw/
        └── Adult_Census_Income.csv        (이미 존재한다고 가정, git에는 커밋하지 않음)
```

**⚠️ 팀 공통 SPEC — 산출물 위치** : `eda_chart_seaborn.png`, `eda_chart_plotly.html`,
`model.joblib`, `report.md` 4개 파일은 **`outputs/` 같은 하위 폴더가 아니라
프로젝트 루트(`팀원이름/` 바로 아래)에 직접 저장한다.** PM이 병합할 때 각
팀원 폴더를 동일한 상대경로로 찾기 때문에, 위치가 다르면 병합 스크립트가
못 찾는다. `src/config.py`에서 이 4개 경로를 정의할 때 `PROJECT_ROOT / 파일명`
(즉 프로젝트 루트 직속)으로 지정할 것.

**구조화 원칙 (반드시 준수)**
- 검증된 로직은 `src/`에 함수로 분리해 재사용 가능하게 만든다.
- `data/raw/`는 `.gitignore`에 포함해 git에 올리지 않는다. 대신 README에 데이터
  출처를 기록한다.
- `requirements.txt`(버전 고정) + `.env.example` + `README.md`의 실행 가이드,
  이 3가지만 있으면 누구나 동일한 결과를 재현할 수 있어야 한다.
- 각 `src/` 모듈은 (a) 단독 실행(`python src/data_prep.py`)과 (b) 패키지로
  import(`from src.data_prep import ...`) 둘 다 가능해야 한다. 이를 위해 각
  모듈 상단에 아래 패턴을 넣을 것:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from config import ...
  ```
- 처음부터 완벽한 구조를 만들 필요는 없다. `data_prep.py`, `stats_analysis.py`부터
  먼저 완성하고, 그다음 `visualize.py`, `ml_pipeline.py`, `generate_report.py`
  순서로 이어서 구현한다.

---

## 2. `src/config.py`

- 프로젝트 루트, 데이터 경로, 산출물 경로, 컬럼 상수를 한 곳에서 관리.
- 데이터 경로는 기본값(`data/raw/Adult_Census_Income.csv`)을 쓰되, 환경변수
  `CENSUS_DATA_PATH`가 설정되어 있으면 그 값을 우선 사용한다 (`.env`로 오버라이드
  가능하게 `python-dotenv`를 optional하게 불러온다 — 설치 안 되어 있어도 에러 나지
  않게 `try/except ImportError`로 감쌀 것).
- **[팀 SPEC] 산출물 경로는 프로젝트 루트 직속으로 고정** :
  ```python
  SEABORN_CHART_PATH = PROJECT_ROOT / "eda_chart_seaborn.png"
  PLOTLY_CHART_PATH  = PROJECT_ROOT / "eda_chart_plotly.html"
  MODEL_PATH          = PROJECT_ROOT / "model.joblib"
  REPORT_PATH           = PROJECT_ROOT / "report.md"
  ```
- **[팀 SPEC] 상관계수 대상 수치형 컬럼 5개 고정** :
  ```python
  NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
  # fnlwgt는 인구총조사 표본 가중치라 소득과 직접적 인과·상관 의미가 없으므로 제외 (팀 SPEC)
  ```
- **[팀 SPEC] ML Pipeline feature 고정** :
  ```python
  NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
  CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation",
                           "relationship", "race", "sex", "native_country"]
  # fnlwgt는 feature에서도 제외 (팀 SPEC)
  TARGET_COL = "income"
  ```
- **[팀 SPEC] 재현성 고정값** :
  ```python
  RANDOM_STATE = 42
  TEST_SIZE = 0.2
  ```

---

## 3. `src/data_prep.py`

### 함수 목록
1. `load_with_pandas(path) -> pd.DataFrame` : `pd.read_csv`로 로딩, 소요시간 출력.
2. `load_with_polars(path) -> pl.DataFrame` : `pl.read_polars`로 로딩, 소요시간 출력.
3. `compare_pandas_polars(df_pd, df_pl) -> None` : shape 비교, 대표 컬럼(`age`)
   합계가 두 도구에서 일치하는지 검증해서 출력.
4. `basic_eda(df) -> None` : `df.info()`, `df.isnull().sum()`, 중복행 개수,
   `"?"` 문자열로 표기된 결측치 개수(컬럼별)를 출력.
5. `clean_data(df) -> pd.DataFrame` :

   **[팀 SPEC] 결측치 처리 — 반드시 아래 방식을 따를 것 (삭제 금지)**
   - 결측 표기는 `"?"` (공백 없음). `df.read_csv(..., na_values="?")`로 로딩 시점에
     바로 처리하거나, 이미 로딩된 df라면 `df.replace("?", pd.NA)`로 결측 처리한다.
     - ⚠️ `" ?"`(공백 포함)로 잘못 쓰면 매칭이 안 돼서 결측치가 하나도 안 잡히는
       버그가 생기니 주의. 실제 CSV에서 값에 공백이 섞여 있는지 먼저
       `df["workclass"].unique()`로 확인하고 처리할 것 (있으면 `.str.strip()`
       먼저 적용 후 비교).
     - 결측 컬럼 : `workclass`, `occupation`, `native_country` (전체의 약 5~7%).
   - **삭제하지 말고 각 컬럼의 최빈값(mode)으로 대체한다.**
     ```python
     for col in ["workclass", "occupation", "native_country"]:
         mode_value = df[col].mode(dropna=True)[0]
         df[col] = df[col].fillna(mode_value)
     ```
     이유(팀 SPEC 근거) : 결측 비율이 크지 않고, 행을 삭제하면 ML 학습 데이터가
     줄어들어 팀원 간 성능 비교가 불공정해지기 때문. **절대 `dropna()`로
     행을 통째로 지우지 말 것.**
   - `fnlwgt`는 결측이 없으므로 별도 처리 불필요 (그대로 두되, 상관계수/ML
     feature에서는 config.py 규칙대로 제외).

   **중복 제거** : `df.drop_duplicates()` (약 24건 존재). 이건 팀 SPEC에
   명시되어 있지 않지만 일반적인 정제 절차이므로 수행하고, 제거 전/후 건수를
   로그로 남긴다.

   **비정상 값 방어 코드(선택)** : `age`가 17 미만/90 초과, `hours_per_week`가
   0 이하/99 초과인 행이 있으면 로그만 남기고 제거 여부는 신중히 결정
   (팀 SPEC에 명시 없음 — 제거하면 팀원 간 표본 수가 달라질 수 있으니, **제거하지
   않고 그대로 두는 것을 기본값으로 한다.** 이상치 존재 여부만 EDA 단계에서
   출력해 보여준다).

   - 정제 전/후 행 수, 결측 대체 건수(컬럼별)를 출력한다.
6. `prepare_data() -> tuple[pd.DataFrame, int]` :
   위 함수들을 순서대로 호출(로딩 비교 → EDA → 정제)하고
   `(정제된 df, 정제 전 원본 행 수)`를 반환. 파일이 없으면 `FileNotFoundError`를
   잡아서 안내 메시지 출력 후 `sys.exit(1)`.
7. `if __name__ == "__main__":` 블록에서 `prepare_data()` 단독 실행 테스트.

전체 함수에 docstring(설명, Parameters, Returns, Raises)을 작성하고, 각 단계에
`print("="*60)` 구분선과 함께 사람이 읽기 쉬운 로그를 남긴다. 모든 외부 입력(파일
로딩 등)은 `try/except`로 감싼다.

---

## 4. `src/stats_analysis.py`

### 함수 목록
1. `descriptive_stats(df) -> pd.DataFrame` :
   `config.NUMERIC_COLS`(age, education_num, capital_gain, capital_loss,
   hours_per_week)에 대해 `df[cols].describe()`로 평균·표준편차·분위수 출력.
2. `correlation_matrix(df) -> pd.DataFrame` :
   같은 컬럼들의 `.corr()` 상관계수 행렬을 출력(반올림 3자리).
3. `run_ttest_income_group(df) -> None` :

   **[팀 SPEC] t-test 사양 확정**
   - 그룹 : `income` 컬럼 (`<=50K` vs `>50K`)
   - 비교 변수 : `hours_per_week`
   - 방법 : `scipy.stats.ttest_ind(group_50k_under, group_50k_over, equal_var=False)`
     (Welch's t-test — 두 그룹의 분산이 같다고 가정하지 않음)
   - 출력에 **t-statistic, p-value, 그리고 "p < 0.05 → 유의미하다/않다" 해석
     문장을 반드시 포함**한다. (다음 형식 권장:
     `"p-value < 0.05 이므로 두 소득 그룹의 평균 근로시간 차이는 통계적으로 유의미하다."`)
   - 각 그룹의 평균값도 함께 출력해서 해석에 참고 자료를 준다.
4. `run_statistical_analysis(df) -> dict` :
   위 3개 함수를 순서대로 호출하고 `{"describe": ..., "corr": ...}` 반환
   (report.md 생성에 사용).

---

## 5. `src/visualize.py`

- `set_korean_font()` : matplotlib에 한글 폰트(AppleGothic/Malgun Gothic/NanumGothic/
  Noto Sans CJK KR 등)를 후보로 두고 설치된 것을 자동 탐색해 적용, 없으면 그냥
  진행(에러 아님). `axes.unicode_minus = False`도 설정.
- `plot_age_distribution(df, save_path=SEABORN_CHART_PATH)` :
  **[Seaborn 정적 차트 - 분포]** `age` 히스토그램 + KDE. 제목 "나이(age) 분포",
  x축 "나이", y축 "인원수" 라벨 반드시 포함(채점 기준 명시 항목). `fig.savefig()`로
  저장(`fig.show()` 금지). **저장 경로는 `config.SEABORN_CHART_PATH`
  (= `eda_chart_seaborn.png`, 프로젝트 루트) 고정.**
- `plot_income_group_comparison(df, save_path=PLOTLY_CHART_PATH)` :
  **[Plotly 인터랙티브 차트 - 그룹비교]** `income` 그룹(`<=50K` vs `>50K`)별
  평균 `hours_per_week`를 막대 차트로 비교. `plotly.express.bar` 사용, 제목과
  축 라벨 명시. `fig.write_html(save_path)`로 저장(`fig.show()` 금지). **저장
  경로는 `config.PLOTLY_CHART_PATH` (= `eda_chart_plotly.html`, 프로젝트 루트)
  고정.**
- `run_visualization(df)` : 위 두 함수를 순서대로 호출.

---

## 6. `src/ml_pipeline.py`

- **문제 유형 : 분류(classification)**.

  **[팀 SPEC] 타깃 인코딩 확정**
  ```python
  y = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})
  ```
  `.str.strip()`을 매핑 전에 반드시 먼저 적용한다 (원본 값에 앞뒤 공백이
  섞여 있을 수 있음 — 이 CSV는 공백이 없는 것으로 확인되었지만, 다른 팀원의
  raw UCI 로딩 방식에서는 공백이 있을 수 있으므로 방어적으로 항상 적용).

- `build_pipeline() -> Pipeline` :
  - `ColumnTransformer` : 수치형(`NUMERIC_FEATURES`)은 `StandardScaler`,
    범주형(`CATEGORICAL_FEATURES`)은 `OneHotEncoder(handle_unknown="ignore")`.
  - 모델 : `RandomForestClassifier(n_estimators=200, max_depth=12,
    random_state=RANDOM_STATE, n_jobs=-1)`.
  - 위 둘을 `Pipeline(steps=[("preprocessor", ...), ("model", ...)])`으로 묶는다.
- `train_and_evaluate(df, model_path=MODEL_PATH) -> dict` :
  - `X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]` (`fnlwgt`, `income` 미포함
    — 데이터 누수 방지 재확인 필수), `y`는 위 인코딩 규칙대로 생성.
  - **[팀 SPEC] 분할 설정 고정 — 추가 파라미터 넣지 말 것**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ```
    `stratify=y` 등 스펙에 없는 파라미터를 임의로 추가하지 않는다. 추가하면
    같은 `random_state`를 써도 실제 분할되는 행이 달라져서 팀원 간 성능
    비교가 불공정해진다.
  - `pipeline.fit(X_train, y_train)` → `pipeline.predict(X_test)`
  - **[팀 SPEC] 평가지표 — accuracy와 F1-score 둘 다 출력**:
    `accuracy_score(y_test, y_pred)`, `f1_score(y_test, y_pred)` (이진 인코딩
    0/1이므로 `pos_label` 지정 불필요, 기본값이 `1`(`>50K`)을 양성으로 취급).
    `classification_report`도 함께 출력해서 참고 자료로 남긴다.
  - `joblib.dump(pipeline, model_path)`로 저장 후, `joblib.load()`로 다시 불러와
    재로딩한 모델의 accuracy가 원본과 동일한지 확인 출력. **저장 경로는
    `config.MODEL_PATH` (= `model.joblib`, 프로젝트 루트) 고정.**
  - `dict`로 `{"accuracy":.., "f1":.., "n_train":.., "n_test":..}` 반환.

---

## 7. `src/generate_report.py`

- `generate_report(rows_before, rows_after, stats_result, ml_result, report_path=REPORT_PATH)` :
  아래 섹션을 포함한 `report.md`를 자동 생성한다 (f-string으로 값 채워 넣기,
  `DataFrame.to_markdown()` 사용 — `tabulate` 패키지 필요). **저장 경로는
  `config.REPORT_PATH` (= `report.md`, 프로젝트 루트) 고정.**
  1. 제목 + 생성 시각(`datetime.now()`)
  2. 데이터 준비 : 원본/정제 후 행 수, 정제 기준 요약 (결측 `"?"` → 최빈값 대체,
     대체된 컬럼과 건수, 중복 제거 건수)
  3. 통계 분석 : 기술통계 표, 상관계수 표, t-test 결과(그룹=income,
     대상=hours_per_week)의 t통계량·p-value·해석 한 줄
  4. 시각화 : `eda_chart_seaborn.png`, `eda_chart_plotly.html` 경로 안내
  5. ML Pipeline : 모델 구성, 입력 변수(수치형 5개+범주형 8개, fnlwgt 제외 명시),
     `test_size=0.2`/`random_state=42`, train/test 건수, **accuracy·F1**
  6. "본인 의견 / 개선 사항" 섹션은 **빈 템플릿으로 남겨둔다** (팀원이 직접 작성할
     부분이므로 자동 생성하지 않음).
  - **[병합 기준 대응]** report.md는 `python main.py` 실행만으로 끝까지 자동
    생성되어야 하며, 수동 편집이 필요한 부분이 있으면 안 된다 (5번 항목 제외).

---

## 8. `main.py`

- `src.data_prep.prepare_data()` → `src.stats_analysis.run_statistical_analysis()`
  → `src.visualize.run_visualization()` → `src.ml_pipeline.train_and_evaluate()`
  → `src.generate_report.generate_report()` 순서로 호출하는 단일 진입점.
- 전체 실행 시간을 측정해 마지막에 출력.
- 최상위에서 `try/except Exception`으로 감싸 실패 시 원인 출력 후 `sys.exit(1)`.
- 파일 최상단에 프로그램 설명 docstring(목적, 각 단계 요약, 실행 방법, 변경내역)을
  작성한다.
- **[팀 SPEC] 파일 위치 고정** : 프로젝트 루트 바로 아래 `main.py`로 저장
  (다른 이름/위치 금지).

---

## 9. `README.md`

다음 내용을 포함:
- 프로젝트 한 줄 소개
- 폴더 구조 트리 (산출물 4종이 프로젝트 루트에 위치한다는 점 명시)
- 데이터 출처(URL) 및 `data/raw/`에 배치하는 방법
- 실행 방법(venv 생성 → `pip install -r requirements.txt` → `.env` 설정(선택) →
  `python main.py`)
- 실행 후 생성되는 파일 목록과 위치 설명 (`eda_chart_seaborn.png`,
  `eda_chart_plotly.html`, `model.joblib`, `report.md` — 모두 프로젝트 루트)
- 개별 모듈 단독 실행법
- **팀 공통 SPEC을 따랐다는 점**과, 팀 SPEC 문서 링크/파일명을 각주로 남긴다
  (병합 담당자가 규칙 준수 여부를 빠르게 확인할 수 있도록).

## 10. `requirements.txt`

버전을 고정해서 명시 (`pandas`, `polars`, `scipy`, `scikit-learn`, `joblib`,
`matplotlib`, `seaborn`, `plotly`, `tabulate`, 선택적으로 `python-dotenv`).
설치된 각 패키지의 실제 버전을 `pip freeze`로 확인해서 채워 넣을 것 — 임의 버전
번호를 지어내지 말 것.

## 11. `.gitignore`

`data/raw/`, `venv/`, `.venv/`, `__pycache__/`, `*.pyc`, `.env`,
`.DS_Store`, `.vscode/`, `.idea/` 포함.
**주의** : 팀 SPEC상 `eda_chart_seaborn.png`, `eda_chart_plotly.html`,
`model.joblib`, `report.md`는 병합 담당자가 비교해야 하는 결과물이므로
`.gitignore`에 넣지 말고 git에 커밋되도록 둔다 (기존에 `outputs/`를 통째로
무시하던 방식과 다름 — 반드시 반영할 것).

## 12. `.env.example`

```
# cp .env.example .env 로 복사해서 사용
# 데이터를 기본 위치(data/raw/Adult_Census_Income.csv)가 아닌 다른 경로에 두었을 때만 채운다.
CENSUS_DATA_PATH=
```

---

## 13. 구현 후 검증 절차 (필수)

1. 새 가상환경을 만들고 `requirements.txt`만으로 설치가 되는지 확인한다.
2. `python main.py`를 루트에서 실행해 에러 없이 끝까지 도는지 확인하고, 총
   소요시간을 기록한다.
3. `python src/data_prep.py`처럼 각 모듈을 `src/` 안에서 단독 실행도 되는지
   확인한다.
4. 프로젝트 **루트**에 다음 4개 파일이 실제로 생성됐는지 확인한다(경로 오류
   여부를 반드시 확인 — `outputs/` 같은 하위 폴더에 잘못 생성되면 안 됨):
   `eda_chart_seaborn.png`, `eda_chart_plotly.html`, `model.joblib`, `report.md`.
5. `report.md`를 열어 결측치 대체 방식(최빈값), t-test 그룹(income), ML 평가지표
   (accuracy+F1)가 팀 SPEC과 일치하는 값으로 채워졌는지 최종 확인한다.
6. 위 검증이 모두 끝난 뒤, 테스트로 만든 가상환경/캐시(`__pycache__` 등)는
   정리한다.

## 14. 코드 스타일 / 완성도 체크리스트

- 모든 함수에 docstring(역할, Parameters, Returns 필요 시 Raises) 작성.
- 파일 상단에 모듈 설명 주석(무엇을 하는 파일인지).
- 외부 입력(파일 읽기, 모델 학습 등)은 전부 `try/except`로 감싸고, 에러 메시지에
  원인을 알 수 있는 문구를 포함.
- 매직 넘버(컬럼명, 경로, 임계값 등)는 `config.py`에 상수로 모아두고 하드코딩하지
  않는다.
- 코드 중복을 피한다(예: 한글 폰트 설정처럼 여러 곳에서 쓰이는 로직은 함수 하나로
  통일).

## 15. 팀 SPEC 준수 최종 체크리스트 (구현 완료 후 반드시 자가 점검)

- [ ] 결측치를 삭제하지 않고 최빈값으로 대체했는가?
- [ ] `na_values`/치환 시 `"?"`를 썼는가 (`" ?"` 아님)?
- [ ] t-test 그룹이 `income`(`<=50K` vs `>50K`), 비교 변수가 `hours_per_week`인가?
- [ ] t-test 결과에 p-value 해석 문장이 포함되어 있는가?
- [ ] `income`을 `.str.strip()` 후 0/1로 인코딩했는가?
- [ ] 평가지표로 accuracy와 F1을 모두 출력하는가?
- [ ] 상관계수·ML feature 모두에서 `fnlwgt`를 제외했는가?
- [ ] `train_test_split`이 정확히 `test_size=0.2, random_state=42`이고, 그 외
      파라미터(`stratify` 등)를 임의로 추가하지 않았는가?
- [ ] 산출물 4종(`eda_chart_seaborn.png`, `eda_chart_plotly.html`, `model.joblib`,
      `report.md`)이 프로젝트 루트에, 지정된 파일명 그대로 생성되는가?
- [ ] `main.py`가 프로젝트 루트에 있는가?
