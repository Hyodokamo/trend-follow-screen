from __future__ import annotations

import time
import re
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_CSV = "universe.csv"

TOP_N = 3
MA_MONTHS = 10
MOM_MONTHS = 12
CORR_THRESHOLD = 0.95

LIQ_THRESHOLD_JPY = 50_000_000
LIQ_THRESHOLD_USD = 1_000_000

# yfinance取得期間
PERIOD = "15y"
RETRY = 3
SLEEP_SEC = 2

# “運用資金（注文数量計算の基準）”
# 本当は口座残高を入れるべきだが、自動化の第一段階では元本固定でよい（裁量排除）
PORTFOLIO_VALUE_NATIVE = 1_000_000.0

OUT_DIR = Path("out")
HIST_DIR = OUT_DIR / "history"
OUT_DIR.mkdir(exist_ok=True)
HIST_DIR.mkdir(exist_ok=True)

# =====================================================


def load_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} must have 'ticker' column")

    tickers = [str(t).strip() for t in df["ticker"].dropna().tolist()]
    seen = set()
    out: list[str] = []
    for t in tickers:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def download_prices(tickers: list[str]) -> pd.DataFrame:
    last_err = None
    for _ in range(RETRY):
        try:
            df = yf.download(
                tickers=tickers,
                period=PERIOD,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if df is None or df.empty:
                raise RuntimeError("yfinance returned empty dataframe")
            return df
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_SEC)
    raise RuntimeError(f"download failed after {RETRY} retries: {last_err}")


def get_field(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    yfinanceの返りが
      - MultiIndex: (ticker, field) か (field, ticker)
      - 単一ticker: 通常カラム
    のどちらでも動くように吸収
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 典型: (ticker, field)
        try:
            out = df.xs(field, axis=1, level=1, drop_level=True)
        except Exception:
            # 逆: (field, ticker)
            out = df.xs(field, axis=1, level=0, drop_level=True)

        cols = [t for t in tickers if t in out.columns]
        return out[cols]

    # 単一ticker
    if field not in df.columns:
        raise KeyError(f"field {field} not in columns: {df.columns}")
    out = df[[field]].copy()
    out.columns = [tickers[0]]
    return out


def liquidity_threshold(ticker: str) -> float:
    return LIQ_THRESHOLD_JPY if ticker.endswith(".T") else LIQ_THRESHOLD_USD


def load_prev_picks_from_history(hist_dir: Path, cur_asof: str) -> pd.DataFrame:
    """
    out/history/picks_YYYY-MM-DD.csv のうち、cur_asof の1つ前を返す。
    なければ空DF（Ticker列のみ）
    """
    files = sorted(hist_dir.glob("picks_*.csv"))
    if not files:
        return pd.DataFrame(columns=["Ticker"])

    def extract_date(p: Path) -> str:
        m = re.match(r"picks_(\d{4}-\d{2}-\d{2})$", p.stem)
        return m.group(1) if m else ""

    dated = [(extract_date(f), f) for f in files]
    dated = [(d, f) for d, f in dated if d]
    dated.sort(key=lambda x: x[0])

    prev_files = [f for d, f in dated if d < cur_asof]
    if not prev_files:
        return pd.DataFrame(columns=["Ticker"])

    prev_file = prev_files[-1]
    df = pd.read_csv(prev_file)
    if "Ticker" not in df.columns:
        raise ValueError(f"{prev_file} must have 'Ticker' column")
    return df


def build_orders(
    cur_picks: pd.DataFrame,
    prev_picks: pd.DataFrame,
    last_prices: pd.Series,
    portfolio_value_native: float,
    top_n: int,
) -> pd.DataFrame:
    """
    先月Pickと今月Pickを比較し、ADD/DROPだけをordersとして出力。
    KEEPは出力しない（＝発注不要の明確化）
    DROPの数量は保有数量が分からないため 'ALL' とする。
    """
    if "Ticker" not in cur_picks.columns:
        raise ValueError("cur_picks must have 'Ticker' column")
    if "Ticker" not in prev_picks.columns:
        # 空DFの想定
        prev_picks = pd.DataFrame(columns=["Ticker"])

    cur_set = set(cur_picks["Ticker"].astype(str))
    prev_set = set(prev_picks["Ticker"].astype(str))

    add = sorted(cur_set - prev_set)
    drop = sorted(prev_set - cur_set)

    orders: list[dict] = []

    # DROP（全売却）
    for t in drop:
        px = last_prices.get(t, np.nan)
        orders.append(
            {
                "Ticker": t,
                "action": "DROP",
                "side": "SELL",
                "qty": "ALL",
                "ref_price": float(px) if pd.notna(px) else "",
                "note": "Sell all (manual: use your current holdings qty)",
            }
        )

    # ADD（等金額配分でBUY数量計算）
    if len(add) > 0:
        target_per = np.floor((portfolio_value_native / top_n) / 1000.0) * 1000.0
        for t in add:
            px = last_prices.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                qty = ""
                note = "Price missing -> set qty manually"
            else:
                qty = int(np.floor(float(target_per) / float(px)))  # 日本ETFは通常1口
                note = f"Target ~{int(target_per)} (native) => qty={qty}"
            orders.append(
                {
                    "Ticker": t,
                    "action": "ADD",
                    "side": "BUY",
                    "qty": qty,
                    "ref_price": float(px) if pd.notna(px) else "",
                    "note": note,
                }
            )

    df = pd.DataFrame(orders)
    if df.empty:
        df = pd.DataFrame(columns=["Ticker", "action", "side", "qty", "ref_price", "note"])
    return df


def main():
    tickers = load_universe(UNIVERSE_CSV)
    if len(tickers) < 3:
        raise ValueError("universe must contain at least 3 tickers")

    raw = download_prices(tickers)
    close_d = get_field(raw, "Close", tickers)
    vol_d = get_field(raw, "Volume", tickers)

    # 流動性：直近60営業日の平均売買代金（終値×出来高）
    traded_value_60d = (close_d * vol_d).rolling(60).mean().iloc[-1]

    liq_ok = pd.Series(False, index=traded_value_60d.index)
    for t in liq_ok.index:
        thr = liquidity_threshold(t)
        val = traded_value_60d.get(t, np.nan)
        if pd.notna(val) and val >= thr:
            liq_ok.loc[t] = True

    # 月末終値（カレンダー月末で最後の取引日）
    close_m = close_d.resample("ME").last()

    # 当月途中は落とす（未確定月を混ぜない）
    last_month_start = close_m.index[-1].to_period("M").to_timestamp()
    this_month_start = pd.Timestamp.utcnow().to_period("M").to_timestamp()
    if last_month_start == this_month_start:
        close_m = close_m.iloc[:-1]

    if len(close_m) < (max(MA_MONTHS, MOM_MONTHS) + 2):
        raise ValueError("not enough monthly history for MA/Momentum")

    ma = close_m.rolling(MA_MONTHS).mean()
    mom = close_m / close_m.shift(MOM_MONTHS) - 1
    trend_ok = close_m > ma

    # 重複排除（相関）
    ret_m = close_m.pct_change().dropna(how="all")
    corr = ret_m.corr()

    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = corr.loc[a, b]
            if pd.notna(c) and c >= CORR_THRESHOLD:
                drop = a if traded_value_60d.get(a, -np.inf) < traded_value_60d.get(b, -np.inf) else b
                to_drop.add(drop)

    final = [t for t in tickers if bool(liq_ok.get(t, False)) and t not in to_drop]
    if len(final) == 0:
        raise RuntimeError("final universe became empty (liq/corr filters too strict)")

    latest = close_m.index[-1]
    asof = latest.strftime("%Y-%m-%d")

    # ASOF txt（latest + 履歴）
    (OUT_DIR / "asof_latest.txt").write_text(asof + "\n", encoding="utf-8")
    (HIST_DIR / f"asof_{asof}.txt").write_text(asof + "\n", encoding="utf-8")

    table = pd.DataFrame(
        {
            "close": close_m.loc[latest],
            f"ma{MA_MONTHS}": ma.loc[latest],
            f"mom{MOM_MONTHS}": mom.loc[latest],
            "trend_ok": trend_ok.loc[latest],
            "score": np.where(trend_ok.loc[latest], mom.loc[latest], -np.inf),
            "avg_traded_value_60d_native": traded_value_60d,
            "liq_ok": liq_ok,
            "dup_drop": pd.Series({t: (t in to_drop) for t in tickers}),
            "in_final": pd.Series({t: (t in final) for t in tickers}),
        }
    )

    ranked = table.loc[final].sort_values(["trend_ok", f"mom{MOM_MONTHS}"], ascending=[False, False]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    ranked["pick"] = ranked["rank"] <= TOP_N

    ranked_out = ranked.reset_index(names="Ticker")
    picks_out = ranked.loc[ranked["pick"]].reset_index(names="Ticker")

    # ---- 出力（latest + 履歴） ----
    ranked_latest = OUT_DIR / "ranking_latest.csv"
    picks_latest = OUT_DIR / "picks_latest.csv"
    meta_latest = OUT_DIR / "meta_latest.csv"

    ranked_hist = HIST_DIR / f"ranking_{asof}.csv"
    picks_hist = HIST_DIR / f"picks_{asof}.csv"
    meta_hist = HIST_DIR / f"meta_{asof}.csv"

    ranked_out.to_csv(ranked_latest, index=False, encoding="utf-8-sig")
    picks_out.to_csv(picks_latest, index=False, encoding="utf-8-sig")
    ranked_out.to_csv(ranked_hist, index=False, encoding="utf-8-sig")
    picks_out.to_csv(picks_hist, index=False, encoding="utf-8-sig")

    meta = pd.DataFrame(
        [
            {
                "asof_month_end": asof,
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "top_n": TOP_N,
                "ma_months": MA_MONTHS,
                "mom_months": MOM_MONTHS,
                "corr_threshold": CORR_THRESHOLD,
                "liq_threshold_jpy": LIQ_THRESHOLD_JPY,
                "liq_threshold_usd": LIQ_THRESHOLD_USD,
                "universe_count": len(tickers),
                "final_universe_count": len(final),
            }
        ]
    )
    meta.to_csv(meta_latest, index=False, encoding="utf-8-sig")
    meta.to_csv(meta_hist, index=False, encoding="utf-8-sig")

    # ---- status（運用に必要な状態をJSONで保存） ----
    status = {
        "asof_month_end": asof,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": TOP_N,
        "ma_months": MA_MONTHS,
        "mom_months": MOM_MONTHS,
        "corr_threshold": CORR_THRESHOLD,
        "liq_threshold_jpy": LIQ_THRESHOLD_JPY,
        "liq_threshold_usd": LIQ_THRESHOLD_USD,
        "universe_count": len(tickers),
        "final_universe": final,
        "picks": picks_out["Ticker"].astype(str).tolist(),
    }
    (OUT_DIR / "status_latest.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HIST_DIR / f"status_{asof}.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- orders（先月との差分からADD/DROPのみ） ----
    prev_picks = load_prev_picks_from_history(HIST_DIR, asof)
    last_prices = table["close"]  # asof月末の参照価格
    orders = build_orders(
        cur_picks=picks_out,
        prev_picks=prev_picks,
        last_prices=last_prices,
        portfolio_value_native=PORTFOLIO_VALUE_NATIVE,
        top_n=TOP_N,
    )

    orders_latest = OUT_DIR / "orders_latest.csv"
    orders_hist = HIST_DIR / f"orders_{asof}.csv"
    orders.to_csv(orders_latest, index=False, encoding="utf-8-sig")
    orders.to_csv(orders_hist, index=False, encoding="utf-8-sig")

    # ---- screen_all（FALSE含む全件） ----
    rp = ranked[["rank", "pick"]].copy().reset_index(names="Ticker")
    screen_all = table.copy().reset_index(names="Ticker").merge(rp, on="Ticker", how="left")

    screen_all["fail_liq"] = ~screen_all["liq_ok"].fillna(False)
    screen_all["fail_dup"] = screen_all["dup_drop"].fillna(False)
    screen_all["fail_trend"] = ~screen_all["trend_ok"].fillna(False)

    screen_all_latest = OUT_DIR / "screen_all_latest.csv"
    screen_all_hist = HIST_DIR / f"screen_all_{asof}.csv"
    screen_all.to_csv(screen_all_latest, index=False, encoding="utf-8-sig")
    screen_all.to_csv(screen_all_hist, index=False, encoding="utf-8-sig")

    print("ASOF:", asof)
    print("Final Universe:", final)
    print("Saved latest:", ranked_latest, picks_latest, meta_latest)
    print("Saved history:", ranked_hist, picks_hist, meta_hist)
    print("Saved extra:", orders_latest, screen_all_latest, (OUT_DIR / "status_latest.json"))


if __name__ == "__main__":
    main()
