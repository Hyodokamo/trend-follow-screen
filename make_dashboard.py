from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

OUT_DIR = Path("out")
DOCS_DIR = Path("docs")
HIST_SRC = OUT_DIR / "history"
HIST_DST = DOCS_DIR / "history"

DOCS_DIR.mkdir(exist_ok=True)
HIST_DST.mkdir(exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    return pd.read_csv(path)


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def build_history_links() -> str:
    files = sorted(HIST_DST.glob("picks_*.csv"), reverse=True)
    if not files:
        return "<p>(履歴なし)</p>"
    items = []
    for f in files:
        items.append(f'<li><a href="history/{f.name}">{f.name}</a></li>')
    return "<ul>" + "".join(items) + "</ul>"


def build_html(meta: pd.DataFrame, picks: pd.DataFrame, ranking: pd.DataFrame) -> str:
    asof = meta.loc[0, "asof_month_end"] if len(meta) else ""
    gen = meta.loc[0, "generated_at_utc"] if len(meta) else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    css = """
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:24px;}
h1,h2{margin:0.2em 0;}
.meta{color:#444; margin-bottom:16px;}
.box{border:1px solid #ddd; border-radius:10px; padding:12px; margin:12px 0;}
table{border-collapse:collapse; width:100%; font-size:14px;}
th,td{border:1px solid #ddd; padding:6px; text-align:left;}
th{background:#f6f6f6;}
.small{font-size:12px; color:#666;}
"""

    history_links = build_history_links()

    # NOTE: f-stringを避け、.format()で埋める（波括弧事故を根絶）
    html = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Trend Follow Screen - {asof}</title>
<style>{css}</style>
</head>
<body>
<h1>Trend Follow Screen</h1>

<div class="meta">
  <div>ASOF（月末）: <b>{asof}</b></div>
  <div>生成時刻（UTC）: {gen} / 表示更新: {now}</div>
  <div class="small">注意：これは売買の自動執行ではなく「月次ルールのスクリーニング結果の可視化」です。</div>
</div>

<div class="box">
  <h2>今月のPick</h2>
  <p class="small"><a href="picks.csv">picks.csv</a></p>
  {picks_table}
</div>

<div class="box">
  <h2>ランキング（Final Universeのみ）</h2>
  <p class="small"><a href="ranking.csv">ranking.csv</a></p>
  {ranking_table}
</div>

<div class="box">
  <h2>過去Picks（直近12回）</h2>
  {history_links}
</div>

<div class="box">
  <h2>メタ</h2>
  <p class="small"><a href="meta.csv">meta.csv</a></p>
  {meta_table}
</div>

</body>
</html>
""".format(
        css=css,
        asof=asof,
        gen=gen,
        now=now,
        picks_table=picks.to_html(index=False),
        ranking_table=ranking.to_html(index=False),
        history_links=history_links,
        meta_table=meta.to_html(index=False),
    )

    return html


def main():
    ranking_path = OUT_DIR / "ranking_latest.csv"
    picks_path = OUT_DIR / "picks_latest.csv"
    meta_path = OUT_DIR / "meta_latest.csv"

    ranking = read_csv(ranking_path)
    picks = read_csv(picks_path)
    meta = read_csv(meta_path)

    # docsに最新CSVコピー
    safe_copy(ranking_path, DOCS_DIR / "ranking.csv")
    safe_copy(picks_path, DOCS_DIR / "picks.csv")
    safe_copy(meta_path, DOCS_DIR / "meta.csv")

    # 履歴（picksのみ最新12回）をdocs/historyへコピー
    if HIST_SRC.exists():
        hist_files = sorted(HIST_SRC.glob("picks_*.csv"))[-12:]
        for f in hist_files:
            safe_copy(f, HIST_DST / f.name)

    # HTML生成
    html = build_html(meta, picks, ranking)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
