from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import re
import pandas as pd
import html as html_escape

DASHBOARD_VERSION = "v3-2026-01-18-no-style"

OUT_DIR = Path("out")
DOCS_DIR = Path("docs")
HIST_SRC = OUT_DIR / "history"
HIST_DST = DOCS_DIR / "history"

DOCS_DIR.mkdir(exist_ok=True)
HIST_DST.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    return pd.read_csv(path)


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def latest_history_files(pattern: str, k: int = 12) -> list[Path]:
    if not HIST_SRC.exists():
        return []
    files = sorted(HIST_SRC.glob(pattern))
    return files[-k:]


def copy_history_to_docs():
    patterns = ["picks_*.csv", "orders_*.csv", "screen_all_*.csv", "meta_*.csv", "asof_*.txt"]
    for pat in patterns:
        for f in latest_history_files(pat, k=12):
            safe_copy(f, HIST_DST / f.name)


def build_history_links(prefix: str, title: str) -> str:
    files = sorted(HIST_DST.glob(f"{prefix}_*.csv"), reverse=True)
    if not files:
        return f"<p>({title}: 履歴なし)</p>"
    items = [f'<li><a href="history/{f.name}">{f.name}</a></li>' for f in files]
    return "<div><div class='small'><b>{}</b></div><ul>{}</ul></div>".format(title, "".join(items))


def extract_date_from_name(stem: str, prefix: str) -> str:
    m = re.match(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})$", stem)
    return m.group(1) if m else ""


def load_latest_and_prev_from_history(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if not HIST_SRC.exists():
        return pd.DataFrame(), pd.DataFrame(), "(none)", "(none)"

    files = sorted(HIST_SRC.glob(f"{prefix}_*.csv"))
    dated: list[tuple[str, Path]] = []
    for f in files:
        d = extract_date_from_name(f.stem, prefix)
        if d:
            dated.append((d, f))
    dated.sort(key=lambda x: x[0])

    if not dated:
        return pd.DataFrame(), pd.DataFrame(), "(none)", "(none)"

    cur_asof, cur_file = dated[-1]
    cur = pd.read_csv(cur_file)

    if len(dated) >= 2:
        prev_asof, prev_file = dated[-2]
        prev = pd.read_csv(prev_file)
    else:
        prev_asof = "(none)"
        prev = pd.DataFrame(columns=cur.columns)

    return cur, prev, cur_asof, prev_asof


def diff_picks(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if cur is None or cur.empty or "Ticker" not in cur.columns:
        return pd.DataFrame(columns=["Ticker", "action"])
    if prev is None or prev.empty or "Ticker" not in prev.columns:
        prev = pd.DataFrame(columns=["Ticker"])

    cur_set = set(cur["Ticker"].astype(str))
    prev_set = set(prev["Ticker"].astype(str))

    keep = sorted(cur_set & prev_set)
    add = sorted(cur_set - prev_set)
    drop = sorted(prev_set - cur_set)

    rows = [{"Ticker": t, "action": "KEEP"} for t in keep] + \
           [{"Ticker": t, "action": "ADD"} for t in add] + \
           [{"Ticker": t, "action": "DROP"} for t in drop]

    out = pd.DataFrame(rows)
    order = {"KEEP": 0, "ADD": 1, "DROP": 2}
    if not out.empty:
        out["__o"] = out["action"].map(order)
        out = out.sort_values(["__o", "Ticker"]).drop(columns="__o")
    return out


def df_to_html_table_with_action_class(df: pd.DataFrame) -> str:
    """
    jinja2不要。action列がある場合は行に class を付与してCSSで色付け。
    """
    if df is None or df.empty:
        return "<p>(なし)</p>"

    cols = list(df.columns)
    thead = "<thead><tr>" + "".join(f"<th>{html_escape.escape(str(c))}</th>" for c in cols) + "</tr></thead>"

    body_rows = []
    for _, r in df.iterrows():
        action = str(r.get("action", "")).upper()
        cls = ""
        if action in ("ADD", "DROP", "KEEP"):
            cls = f' class="row-{action.lower()}"'
        tds = "".join(f"<td>{html_escape.escape(str(r.get(c, '')))}</td>" for c in cols)
        body_rows.append(f"<tr{cls}>{tds}</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def build_html(meta: pd.DataFrame, picks: pd.DataFrame, ranking: pd.DataFrame,
               orders: pd.DataFrame, screen_all: pd.DataFrame) -> str:
    asof = meta.loc[0, "asof_month_end"] if len(meta) else ""
    gen = meta.loc[0, "generated_at_utc"] if len(meta) else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    hist_picks = build_history_links("picks", "Picks（直近12回）")
    hist_orders = build_history_links("orders", "Orders（直近12回）")
    hist_screen = build_history_links("screen_all", "Screen All（直近12回）")

    cur_picks_h, prev_picks_h, cur_asof, prev_asof = load_latest_and_prev_from_history("picks")
    diff = diff_picks(cur_picks_h, prev_picks_h)
    diff_html = df_to_html_table_with_action_class(diff)

    picks_table = picks.to_html(index=False) if len(picks) else "<p>(picks.csv がありません)</p>"
    ranking_table = ranking.to_html(index=False) if len(ranking) else "<p>(ranking.csv がありません)</p>"
    meta_table = meta.to_html(index=False) if len(meta) else "<p>(meta.csv がありません)</p>"

    orders_table = orders.to_html(index=False) if len(orders) else "<p>(注文なし＝先月から変更なし)</p>"

    cur_html = cur_picks_h.to_html(index=False) if len(cur_picks_h) else "<p>(今月picks履歴がありません)</p>"
    prev_html = prev_picks_h.to_html(index=False) if len(prev_picks_h) else "<p>(先月なし)</p>"

    screen_all_table = screen_all.to_html(index=False) if len(screen_all) else "<p>(screen_all.csv がありません)</p>"

    css = """
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:24px;}
h1,h2{margin:0.2em 0;}
.meta{color:#444; margin-bottom:16px;}
.box{border:1px solid #ddd; border-radius:10px; padding:12px; margin:12px 0;}
table{border-collapse:collapse; width:100%; font-size:14px;}
th,td{border:1px solid #ddd; padding:6px; text-align:left;}
th{background:#f6f6f6;}
.small{font-size:12px; color:#666;}
.grid2{display:grid; grid-template-columns: 1fr 1fr; gap:12px;}
details > summary{cursor:pointer; padding:6px 0;}
footer{margin-top:18px; color:#777; font-size:12px;}
/* diff table row highlights */
.row-add{background:#eaffea;}
.row-drop{background:#ffecec;}
.row-keep{background:#f7f7f7;}
"""

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
  <h2>先月→今月の変更（KEEP/ADD/DROP）</h2>
  <div class="small">先月: {prev_asof} / 今月: {cur_asof}</div>
  {diff_html}
</div>

<div class="box">
  <h2>今月の注文（ADD/DROPのみ）</h2>
  <p class="small"><a href="orders.csv">orders.csv</a></p>
  {orders_table}
</div>

<div class="box">
  <h2>今月のPick（latest）</h2>
  <p class="small"><a href="picks.csv">picks.csv</a></p>
  {picks_table}
</div>

<div class="box">
  <h2>今月 vs 先月のPick（並べて確認）</h2>
  <div class="grid2">
    <div>
      <h3 style="margin-top:0;">今月（{cur_asof}）</h3>
      {cur_html}
    </div>
    <div>
      <h3 style="margin-top:0;">先月（{prev_asof}）</h3>
      {prev_html}
    </div>
  </div>
</div>

<div class="box">
  <h2>ランキング（Final Universeのみ）</h2>
  <p class="small"><a href="ranking.csv">ranking.csv</a></p>
  {ranking_table}
</div>

<div class="box">
  <h2>履歴</h2>
  {hist_picks}
  {hist_orders}
  {hist_screen}
</div>

<div class="box">
  <h2>メタ</h2>
  <p class="small"><a href="meta.csv">meta.csv</a></p>
  {meta_table}
</div>

<div class="box">
  <h2>スクリーニング全結果（参考：FALSE含む）</h2>
  <p class="small"><a href="screen_all.csv">screen_all.csv</a></p>
  <details>
    <summary>表示する（落選理由も含む）</summary>
    {screen_all_table}
  </details>
</div>

<footer>DASHBOARD_VERSION: {ver}</footer>
</body>
</html>
""".format(
        css=css,
        asof=asof,
        gen=gen,
        now=now,
        prev_asof=prev_asof,
        cur_asof=cur_asof,
        diff_html=diff_html,
        orders_table=orders_table,
        picks_table=picks_table,
        ranking_table=ranking_table,
        cur_html=cur_html,
        prev_html=prev_html,
        hist_picks=hist_picks,
        hist_orders=hist_orders,
        hist_screen=hist_screen,
        meta_table=meta_table,
        screen_all_table=screen_all_table,
        ver=DASHBOARD_VERSION,
    )
    return html


def main():
    ranking_path = OUT_DIR / "ranking_latest.csv"
    picks_path = OUT_DIR / "picks_latest.csv"
    meta_path = OUT_DIR / "meta_latest.csv"

    orders_path = OUT_DIR / "orders_latest.csv"
    screen_all_path = OUT_DIR / "screen_all_latest.csv"
    status_path = OUT_DIR / "status_latest.json"

    ranking = read_csv(ranking_path)
    picks = read_csv(picks_path)
    meta = read_csv(meta_path)

    orders = pd.read_csv(orders_path) if orders_path.exists() else pd.DataFrame()
    screen_all = pd.read_csv(screen_all_path) if screen_all_path.exists() else pd.DataFrame()

    # docsに最新コピー
    safe_copy(ranking_path, DOCS_DIR / "ranking.csv")
    safe_copy(picks_path, DOCS_DIR / "picks.csv")
    safe_copy(meta_path, DOCS_DIR / "meta.csv")

    if orders_path.exists():
        safe_copy(orders_path, DOCS_DIR / "orders.csv")
    else:
        (DOCS_DIR / "orders.csv").write_text("Ticker,action,side,qty,ref_price,note\n", encoding="utf-8")

    if screen_all_path.exists():
        safe_copy(screen_all_path, DOCS_DIR / "screen_all.csv")
    else:
        (DOCS_DIR / "screen_all.csv").write_text("", encoding="utf-8")

    if status_path.exists():
        safe_copy(status_path, DOCS_DIR / "status.json")

    # 履歴コピー
    copy_history_to_docs()

    # HTML生成
    html = build_html(meta, picks, ranking, orders, screen_all)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")

    print("make_dashboard.py:", DASHBOARD_VERSION)
    print("wrote:", DOCS_DIR / "index.html")


if __name__ == "__main__":
    main()
