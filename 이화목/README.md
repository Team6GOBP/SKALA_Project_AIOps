# Adult Census Income End2End 분석 프로젝트

Adult Census Income 데이터셋을 대상으로 Pandas·Polars 로딩 비교, 결측치/중복 정제,
시각화(Seaborn+Plotly), 통계 분석(기술통계·상관계수·t-test), sklearn Pipeline 분류
모델 학습, `report.md` 자동 생성까지 이어지는 End2End 파이프라인.

## 폴더 구조

```
이화목/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                      # 진입점
├── eda_chart_seaborn.png        # 실행 후 생성 (프로젝트 루트 고정)
├── eda_chart_plotly.html        # 실행 후 생성 (프로젝트 루트 고정)
├── model.joblib                 # 실행 후 생성 (프로젝트 루트 고정)
├── report.md                    # 실행 후 생성 (프로젝트 루트 고정)
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
        └── Adult_Census_Income.csv   (직접 배치, git에는 커밋하지 않음)
```

> 산출물 4종(`eda_chart_seaborn.png`, `eda_chart_plotly.html`, `model.joblib`,
> `report.md`)은 `outputs/` 같은 하위 폴더가 아니라 **프로젝트 루트에 직접**
> 생성/저장된다 (팀 SPEC).

## 데이터 출처

[UCI Adult / Census Income dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

원본 컬럼명은 하이픈 표기(`education-num` 등)이지만, 이 프로젝트가 사용하는
`Adult_Census_Income.csv`는 스네이크케이스(`education_num` 등)로 정리된 버전이다.
파일을 내려받아 `data/raw/Adult_Census_Income.csv` 경로에 배치한다 (`.gitignore`에
포함되어 git에는 올라가지 않음).

## 실행 방법

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # 선택 사항 (데이터 경로를 바꾸고 싶을 때만 편집)
python main.py
```

## 실행 후 생성되는 파일 (모두 프로젝트 루트)

| 파일 | 설명 |
| --- | --- |
| `eda_chart_seaborn.png` | Seaborn 정적 차트 — 나이(age) 분포 히스토그램 + KDE |
| `eda_chart_plotly.html` | Plotly 인터랙티브 차트 — 소득 그룹별 평균 근로시간 비교 |
| `model.joblib` | 학습된 sklearn Pipeline(전처리 + RandomForestClassifier) |
| `report.md` | 데이터 준비·통계 분석·시각화·ML 결과를 종합한 자동 리포트 |

## 개별 모듈 단독 실행

프로젝트 루트에서 각 모듈을 개별적으로 실행할 수도 있다.

```bash
python src/data_prep.py
python src/stats_analysis.py
python src/visualize.py
python src/ml_pipeline.py
```

## 팀 공통 SPEC 준수

이 프로젝트는 `PROJECT_BRIEF.md`에 명시된 팀 공통 SPEC(결측치 최빈값 대체,
`fnlwgt` 제외, t-test 사양, `train_test_split(test_size=0.2, random_state=42)`,
산출물 4종의 루트 고정 경로 등)을 그대로 따랐다. 규칙 준수 여부는
[`PROJECT_BRIEF.md`](./PROJECT_BRIEF.md)의 "팀 SPEC 준수 최종 체크리스트"
(15번 항목)로 확인할 수 있다.
