from __future__ import annotations
import time
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
    out = []
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
    のどちらでも動くように最小限吸収
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
    
    asof_latest_txt = OUT_DIR / "asof_latest.txt"
    asof_hist_txt = HIST_DIR / f"asof_{asof}.txt"
    
    asof_latest_txt.write_text(asof + "\n", encoding="utf-8")
    asof_hist_txt.write_text(asof + "\n", encoding="utf-8")

    table = pd.DataFrame({
        "close": close_m.loc[latest],
        f"ma{MA_MONTHS}": ma.loc[latest],
        f"mom{MOM_MONTHS}": mom.loc[latest],
        "trend_ok": trend_ok.loc[latest],
        "score": np.where(trend_ok.loc[latest], mom.loc[latest], -np.inf),
        "avg_traded_value_60d_native": traded_value_60d,
        "liq_ok": liq_ok,
        "dup_drop": pd.Series({t: (t in to_drop) for t in tickers}),
        "in_final": pd.Series({t: (t in final) for t in tickers}),
    })

    ranked = table.loc[final].sort_values(["trend_ok", f"mom{MOM_MONTHS}"], ascending=[False, False]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    ranked["pick"] = ranked["rank"] <= TOP_N

    # ---- 出力（latest + 履歴） ----
    ranked_latest = OUT_DIR / "ranking_latest.csv"
    picks_latest = OUT_DIR / "picks_latest.csv"
    meta_latest = OUT_DIR / "meta_latest.csv"

    ranked_hist = HIST_DIR / f"ranking_{asof}.csv"
    picks_hist = HIST_DIR / f"picks_{asof}.csv"
    meta_hist = HIST_DIR / f"meta_{asof}.csv"

    ranked_out = ranked.reset_index(names="Ticker")
    picks_out = ranked.loc[ranked["pick"]].reset_index(names="Ticker")

    ranked_out.to_csv(ranked_latest, index=False, encoding="utf-8-sig")
    picks_out.to_csv(picks_latest, index=False, encoding="utf-8-sig")

    ranked_out.to_csv(ranked_hist, index=False, encoding="utf-8-sig")
    picks_out.to_csv(picks_hist, index=False, encoding="utf-8-sig")

    meta = pd.DataFrame([{
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
    }])
    meta.to_csv(meta_latest, index=False, encoding="utf-8-sig")
    meta.to_csv(meta_hist, index=False, encoding="utf-8-sig")

    print("ASOF:", asof)
    print("Final Universe:", final)
    print("Saved latest:", ranked_latest, picks_latest, meta_latest)
    print("Saved history:", ranked_hist, picks_hist, meta_hist)


if __name__ == "__main__":
    main()
