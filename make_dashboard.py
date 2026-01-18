from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("out")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_table_png(df: pd.DataFrame, path: Path, title: str):
    fig = plt.figure(figsize=(12, min(0.6 + 0.3 * len(df), 12)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    show_cols = [c for c in df.columns if c in [
        "Ticker", "close", "ma10", "mom12", "trend_ok", "score",
        "avg_traded_value_60d_native", "liq_ok", "dup_drop", "rank", "pick"
    ]]
    if not show_cols:
        show_cols = df.columns.tolist()

    table = ax.table(
        cellText=df[show_cols].round(4).astype(str).values,
        colLabels=show_cols,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_mom_bar(df: pd.DataFrame, mom_col: str, path: Path):
    d = df.copy()
    d = d.sort_values(mom_col, ascending=False)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.bar(d["Ticker"], d[mom_col])
    ax.set_title(f"{mom_col} (sorted)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

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
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Trend Follow Screen</title>
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
  <p class="small">CSV: docs/picks.csv</p>
  {picks.to_html(index=False, escape=False)}
</div>

<div class="box">
  <h2>ランキング（Final Universeのみ）</h2>
  <p class="small">CSV: docs/ranking.csv</p>
  {ranking.to_html(index=False, escape=False)}
</div>

<div class="box">
  <h2>モメンタム概観</h2>
  <img src="mom_bar.png" style="max-width:100%; height:auto;" />
</div>

<div class="box">
  <h2>画像（読みやすい版）</h2>
  <div class="small">picks_table.png / ranking_table.png</div>
  <img src="picks_table.png" style="max-width:100%; height:auto;" />
  <img src="ranking_table.png" style="max-width:100%; height:auto; margin-top:10px;" />
</div>

<div class="small">このページは /docs をGitHub Pagesのソースにして公開します。:contentReference[oaicite:9]{index=9}</div>
</body>
</html>
"""

def main():
    ranking_path = OUT_DIR / "ranking_latest.csv"
    picks_path = OUT_DIR / "picks_latest.csv"
    meta_path = OUT_DIR / "meta_latest.csv"

    ranking = read_csv(ranking_path)
    picks = read_csv(picks_path)
    meta = read_csv(meta_path)

    # docsにコピー
    (DOCS_DIR / "ranking.csv").write_bytes(ranking_path.read_bytes())
    (DOCS_DIR / "picks.csv").write_bytes(picks_path.read_bytes())
    (DOCS_DIR / "meta.csv").write_bytes(meta_path.read_bytes())

    # 可視化PNG
    save_table_png(picks, DOCS_DIR / "picks_table.png", "Picks")
    save_table_png(ranking.head(30), DOCS_DIR / "ranking_table.png", "Ranking (top 30 shown)")
    mom_col = [c for c in ranking.columns if c.startswith("mom")][0]
    save_mom_bar(ranking, mom_col, DOCS_DIR / "mom_bar.png")

    # HTML
    html = build_html(meta, picks, ranking)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    main()
