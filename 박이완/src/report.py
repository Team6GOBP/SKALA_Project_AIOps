"""
report.py — 분석 리포트 자동 생성 모듈  (팀 SPEC 5 파일명 준수)
  - 강의(12장. 분석 자동화)에서 다룬 Jinja2 템플릿 방식으로
    분석 결과를 output/report.md 로 자동 생성한다.
  - 병합 기준 5번(자동화 완성도) : 수동 편집 없이 main.py 만 돌면 나와야 함
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

BASE = Path(__file__).resolve().parent.parent
OUT_PATH = BASE / "output" / "report.md"   # SPEC 5 파일명


def write_report(data_summary: dict, viz: dict, stats: dict, ml: dict) -> Path:
    """각 단계의 요약 dict 를 템플릿에 주입해 report.md 를 생성한다."""
    try:
        env = Environment(loader=FileSystemLoader(BASE / "templates"))
        tmpl = env.get_template("report_template.md")
        md = tmpl.render(
            generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
            data=data_summary, viz=viz, stats=stats, ml=ml,
        )
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(md, encoding="utf-8")
        print(f"\n[report] 리포트 자동 생성 완료 → {OUT_PATH}")
    except Exception as e:
        # 리포트 생성 실패가 분석 자체를 무효화하지 않도록 원인만 출력
        print(f"[오류] report.md 생성 실패: {e}")
        raise
    return OUT_PATH
