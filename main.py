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
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
def load_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    tickers = [t.strip() for t in df["ticker"].dropna().tolist()]
    seen = set()
    out = []
    for t in tickers:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def download_prices(tickers: list[str]) -> pd.DataFrame:
    last_err = None
    for k in range(RETRY):
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
    if isinstance(df.columns, pd.MultiIndex):
        out = df.xs(field, axis=1, level=1, drop_level=True)
        cols = [t for t in tickers if t in out.columns]
        return out[cols]
    else:
        if field not in df.columns:
            raise KeyError(f"field {field} not in columns: {df.columns}")
        out = df[[field]].copy()
        out.columns = [tickers[0]]
        return out

def month_end_index(close_d: pd.DataFrame) -> pd.DatetimeIndex:
    close_m = close_d.resample("M").last()
    now_utc = pd.Timestamp.now(tz="UTC").to_period("M")
    last_m = close_m.index[-1].to_period("M")
    if last_m == now_utc:
        close_m = close_m.iloc[:-1]

    return close_m.index

def liquidity_threshold(ticker: str) -> float:
    return LIQ_THRESHOLD_JPY if ticker.endswith(".T") else LIQ_THRESHOLD_USD

def main():
    tickers = load_universe(UNIVERSE_CSV)
    if len(tickers) < 3:
        raise ValueError("universe must contain at least 3 tickers")

    raw = download_prices(tickers)
    close_d = get_field(raw, "Close", tickers)
    vol_d = get_field(raw, "Volume", tickers)

    traded_value_60d = (close_d * vol_d).rolling(60).mean().iloc[-1]

    liq_ok = pd.Series(False, index=traded_value_60d.index)
    for t in liq_ok.index:
        thr = liquidity_threshold(t)
        val = traded_value_60d.get(t, np.nan)
        if pd.notna(val) and val >= thr:
            liq_ok.loc[t] = True

    close_m = close_d.resample("M").last()
    # 今月途中は落とす
    if close_m.index[-1].to_period("M") == pd.Timestamp.now(tz="UTC").to_period("M"):
        close_m = close_m.iloc[:-1]

    if len(close_m) < (max(MA_MONTHS, MOM_MONTHS) + 2):
        raise ValueError("not enough monthly history for MA/Momentum")

    ma = close_m.rolling(MA_MONTHS).mean()
    mom = close_m / close_m.shift(MOM_MONTHS) - 1
    trend_ok = close_m > ma

    ret_m = close_m.pct_change().dropna(how="all")
    corr = ret_m.corr()

    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = corr.loc[a, b]
            if pd.notna(c) and c >= CORR_THRESHOLD:
                # 流動性の低い方を落とす（ルール化）
                drop = a if traded_value_60d.get(a, -np.inf) < traded_value_60d.get(b, -np.inf) else b
                to_drop.add(drop)

    final = [t for t in tickers if bool(liq_ok.get(t, False)) and t not in to_drop]

    if len(final) == 0:
        raise RuntimeError("final universe became empty (liq/corr filters too strict)")

    asof = close_m.index[-1].strftime("%Y-%m-%d")

    latest = close_m.index[-1]
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

    ranked = table.loc[final].sort_values(["trend_ok", f"mom{MOM_MONTHS}"], ascending=[False, False])

    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    ranked["pick"] = ranked["rank"] <= TOP_N

    ranked_out = OUT_DIR / "ranking_latest.csv"
    picks_out = OUT_DIR / "picks_latest.csv"
    meta_out = OUT_DIR / "meta_latest.csv"

    ranked.reset_index(names="Ticker").to_csv(ranked_out, index=False, encoding="utf-8-sig")
    ranked.loc[ranked["pick"]].reset_index(names="Ticker").to_csv(picks_out, index=False, encoding="utf-8-sig")

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
    meta.to_csv(meta_out, index=False, encoding="utf-8-sig")

    print("ASOF:", asof)
    print("Final Universe:", final)
    print("Saved:", ranked_out, picks_out, meta_out)

if __name__ == "__main__":
    main()
