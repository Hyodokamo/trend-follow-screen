from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import re
import pandas as pd

DASHBOARD_VERSION = "v2-2026-01-18"

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
    """
    docs/history に直近12回ぶんをコピー
    """
    patterns = ["picks_*.csv", "orders_*.csv", "screen_all_*.csv", "meta_*.csv"]
    for pat in patterns:
        for f in latest_history_files(pat, k=12):
            safe_copy(f, HIST_DST / f.name)


def build_history_links(prefix: str, title: str) -> str:
    files = sorted(HIST_DST.glob(f"{prefix}_*.csv"), reverse=True)
    if not files:
        return f"<p>({title}: 履歴なし)</p>"
    items = []
    for f in files:
        items.append(f'<li><a href="history/{f.name}">{f.name}</a></li>')
    return "<div><div class='small'><b>{}</b></div><ul>{}</ul></div>".format(title, "".join(items))


def extract_date_from_name(stem: str, prefix: str) -> str:
    m = re.match(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})$", stem)
    return m.group(1) if m else ""


def load_latest_and_prev_from_history(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    out/history/{prefix}_YYYY-MM-DD.csv から 最新と1つ前を読む
    """
    if not HIST_SRC.exists():
        return pd.DataFrame(), pd.DataFrame(), "(none)", "(none)"

    files = sorted(HIST_SRC.glob(f"{prefix}_*.csv"))
    dated = []
    for f in files:
        d = extract_date_from_name(f.stem, prefix)
        if d:
            dated.append((d, f))
    dated.sort(key=lambda x: x[0])

    if len(dated) == 0:
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

    rows = []
    for t in keep:
        rows.append({"Ticker": t, "action": "KEEP"})
    for t in add:
        rows.append({"Ticker": t, "action": "ADD"})
    for t in drop:
        rows.append({"Ticker": t, "action": "DROP"})

    out = pd.DataFrame(rows)
    order = {"KEEP": 0, "ADD": 1, "DROP": 2}
    if not out.empty:
        out["__o"] = out["action"].map(order)
        out = out.sort_values(["__o", "Ticker"]).drop(columns="__o")
    return out


def highlight_action_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>(比較対象が不足しています)</p>"

    def style_row(row):
        a = row.get("action", "")
        if a == "ADD":
            return ["background-color: #eaffea"] * len(row)
        if a == "DROP":
            return ["background-color: #ffecec"] * len(row)
        if a == "KEEP":
            return ["background-color: #f7f7f7"] * len(row)
        return [""] * len(row)

    return df.style.apply(style_row, axis=1).hide(axis="index").to_html()


def build_html(meta: pd.DataFrame, picks: pd.DataFrame, ranking: pd.DataFrame,
               orders: pd.DataFrame, screen_all: pd.DataFrame) -> str:
    asof = meta.loc[0, "asof_month_end"] if len(meta) else ""
    gen = meta.loc[0, "generated_at_utc"] if len(meta) else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # 履歴リンク
    hist_picks = build_history_links("picks", "Picks（直近12回）")
    hist_orders = build_history_links("orders", "Orders（直近12回）")
    hist_screen = build_history_links("screen_all", "Screen All（直近12回）")

    # 今月/先月 picks（historyから）
    cur_picks_h, prev_picks_h, cur_asof, prev_asof = load_latest_and_prev_from_history("picks")
    diff = diff_picks(cur_picks_h, prev_picks_h)
    diff_html = highlight_action_table(diff)

    # 表
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
        meta_table=meta_table,
        cur_html=cur_html,
        prev_html=prev_html,
        hist_picks=hist_picks,
        hist_orders=hist_orders,
        hist_screen=hist_screen,
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

    orders = pd.DataFrame()
    if orders_path.exists():
        orders = pd.read_csv(orders_path)

    screen_all = pd.DataFrame()
    if screen_all_path.exists():
        screen_all = pd.read_csv(screen_all_path)

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

    # どの版が動いたかログに残す（Actionsのログで確認できる）
    print("make_dashboard.py:", DASHBOARD_VERSION)
    print("wrote:", DOCS_DIR / "index.html")


if __name__ == "__main__":
    main()
