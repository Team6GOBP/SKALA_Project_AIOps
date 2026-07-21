# [Day 2] 종합 실습 — End2End 데이터 분석 프로젝트

Adult Census Income 데이터셋으로 **데이터 준비 → 시각화 → 통계 분석 → ML Pipeline → 리포트 자동 생성**을
한 번에 수행하는 End2End 분석 프로젝트입니다. **팀 공통 SPEC(spec.md) 준수 버전**.

## 실행 방법 (재현 가이드)

```bash
pip install -r requirements.txt
python main.py
```

- 최초 실행 시 데이터가 자동으로 다운로드되어 `data/adult.data` 에 캐시됩니다.
  - 출처: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- 모든 결과물은 `output/` 폴더에 생성됩니다.

## 팀 SPEC 준수 사항

| SPEC | 내용 | 반영 위치 |
|---|---|---|
| 1 결측 표기 | `na_values="?"` (공백 없이) | `src/data_prep.py :: load_pandas` |
| 1 결측 처리 | 삭제 없이 컬럼별 **최빈값(mode)** 으로 대체 | `src/data_prep.py :: prepare` |
| 2 t-test | `ttest_ind(<=50K, >50K, equal_var=False)` + 해석 문장 | `src/stats_analysis.py :: run_stats` |
| 3 income 매핑 | `.str.strip()` 후 `{"<=50K":0, ">50K":1}` | `data_prep.py` + `ml_pipeline.py` |
| 3 평가지표 | accuracy + F1-score 둘 다 출력 | `src/ml_pipeline.py` |
| 4 상관계수 대상 | 수치 5개 (fnlwgt 제외) | `src/stats_analysis.py :: NUM_COLS` |
| 4 ML feature | 수치 5개 + 범주 8개 (fnlwgt 제외, native-country 포함) | `src/ml_pipeline.py :: NUM_COLS/CAT_COLS` |
| 4 공통 설정 | `test_size=0.2`, `random_state=42` | `src/ml_pipeline.py` |
| 5 파일명 | `eda_chart_seaborn.png` / `eda_chart_plotly.html` / `model.joblib` / `report.md` / `main.py` | `src/visualize.py`, `src/ml_pipeline.py`, `src/report.py` |

## 폴더 구조

```
<팀원이름>/
├── main.py                         # 전체 실행 (오케스트레이션)
├── src/
│   ├── data_prep.py                # ① SPEC 1: na_values='?', 최빈값 대체
│   ├── visualize.py                # ② SPEC 5 파일명
│   ├── stats_analysis.py           # ③ SPEC 2·4
│   ├── ml_pipeline.py              # ④ SPEC 3·4·5
│   └── report.py                   # ⑤ report.md 자동 생성
├── templates/report_template.md
├── data/                           # 원본 데이터 캐시 (git 제외)
└── output/                         # SPEC 5 파일명 산출물
    ├── eda_chart_seaborn.png
    ├── eda_chart_plotly.html
    ├── model.joblib
    └── report.md
```

## 병합 담당자 체크포인트 (SPEC 6 매핑)

1. **결측치·중복 처리**: `data_prep.py` 의 예외 처리(다운로드 실패, mode 계산 실패)와
   SPEC 1 준수(`na_values="?"` + 최빈값 대체) 확인
2. **시각화**: `visualize.py` 각 차트에 제목·x/y 레이블 명시됨
3. **통계 검정**: `stats_analysis.py` 의 p<0.05 분기 해석 문장 (유의미/유의미하지 않음 양방향)
4. **ML Pipeline (데이터 누수 방지)**: `ml_pipeline.py` 에서 `income` 은 y 전용,
   X 에 포함하지 않으며 fnlwgt 도 제외
5. **report.md 자동화**: `main.py` 만 실행하면 Jinja2 템플릿으로 자동 생성 (수동 편집 없음)
