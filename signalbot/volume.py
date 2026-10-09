"""Volume for spot gold, from two sources.

Spot XAU/USD trades over the counter: there is no exchange volume for it, and
Twelve Data sends none. So signalbot uses

  * COMEX gold futures volume (GC=F, Yahoo Finance, free): real contracts
    traded in the futures market, which moves tick-for-tick with spot. It has
    history (5m / 15m ≈ 60 days, growing in the local cache the longer
    signalbot runs; 1h ≈ 2 years; daily back to 2000), so volume rules can be
    backtested. Yahoo delays it ~10 minutes, so the newest candles get their
    COMEX volume late.
  * Tick volume: the number of live price updates in each candle, counted from
    the Twelve Data stream (stream.py). Instant, but only exists for the time
    the stream has been running. This is the kind of "volume" TradingView
    shows on OANDA:XAUUSD.

Relative volume (RVOL) = this candle's volume ÷ the average of the previous
20 candles, from the same source, so the two are comparable as ratios.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("signalbot.volume")

TICKER = "GC=F"
YF = {"5m": ("5m", "60d"), "15m": ("15m", "60d"), "1h": ("60m", "730d"), "1d": ("1d", "max")}
MAX_AGE = {"5m": 120, "15m": 180, "1h": 600, "1d": 6 * 3600}
RVOL_N = 20


def _path(root: Path, tf: str, ticker: str = TICKER) -> Path:
    if ticker == TICKER:
        return root / "data" / f"COMEX_{tf}.csv"            # gold: keep the original cache name
    return root / "data" / f"YF_{''.join(c for c in ticker if c.isalnum())}_{tf}.csv"


def _download(tf: str, ticker: str = TICKER) -> pd.Series:
    import yfinance as yf
    interval, period = YF[tf]
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no {tf} data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    v = raw["Volume"].astype(float)
    idx = pd.to_datetime(raw.index)
    if tf == "1d":
        idx = pd.DatetimeIndex(idx.date)                     # trading date
    elif idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    v.index = idx
    v.index.name = "time"
    v.name = "volume"
    return v[~v.index.duplicated(keep="last")].sort_index()


def comex(root: Path, tf: str, refresh: bool = True, ticker: str = TICKER) -> pd.Series:
    """Futures volume per candle (COMEX gold by default), cached in data/ and growing over time."""
    p = _path(root, tf, ticker)
    cached = None
    if p.exists():
        try:
            cached = pd.read_csv(p, index_col="time", parse_dates=True)["volume"]
        except Exception:
            cached = None
        if cached is not None and (not refresh or time.time() - p.stat().st_mtime < MAX_AGE[tf]):
            return cached
    try:
        new = _download(tf) if ticker == TICKER else _download(tf, ticker)
    except Exception as e:
        if cached is not None:
            log.info("COMEX volume %s: %s — using cached", tf, e)
            return cached
        log.info("COMEX volume %s unavailable: %s", tf, e)
        return pd.Series(dtype=float, name="volume")
    s = new if cached is None else pd.concat([cached[cached.index < new.index.min()], new])
    p.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(p)
    return s


def attach(df: pd.DataFrame, root: Path, tf: str, refresh: bool = True, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Add `vol` (exchange or futures volume), `tvol` (tick volume) and their RVOLs to the candles.

    stocks / crypto: Twelve Data's own volume · gold, silver, platinum, palladium: their COMEX/NYMEX
    futures volume from Yahoo · forex: no exchange volume exists, tick volume only."""
    from .symbols import FUTURES_VOLUME, SYMBOLS
    meta = SYMBOLS.get(symbol, {"td": "XAU/USD", "kind": "metal"})
    d = df.copy()
    d.attrs["vol_label"] = "volume"
    own = d["volume"] if "volume" in d else pd.Series(np.nan, index=d.index)
    if meta.get("kind") in ("stock", "crypto") and (own > 0).mean() > 0.5:      # the feed really carries volume
        d["vol"] = own.where(own > 0)
        d["tvol"] = np.nan
        d["rvol"] = rvol(d["vol"])
        d["rvol_tick"] = np.nan
        d.attrs["vol_label"] = "exchange volume"
        return d
    ticker = FUTURES_VOLUME.get(meta.get("td"))
    v = comex(root, tf, refresh=refresh, ticker=ticker) if ticker else pd.Series(dtype=float)
    d.attrs["vol_label"] = f"{ticker} futures" if ticker else "tick volume"
    if tf == "1d":
        key = pd.DatetimeIndex(d.index.normalize())          # spot daily candle → its date
        d["vol"] = v.reindex(key).to_numpy() if len(v) else np.nan
    else:
        d["vol"] = v.reindex(d.index).to_numpy() if len(v) else np.nan
    d.loc[d["vol"] <= 0, "vol"] = np.nan                      # 0 = Yahoo has no print for that candle yet
    tv = d["volume"] if "volume" in d else pd.Series(0.0, index=d.index)
    d["tvol"] = tv.where(tv > 0)
    d["rvol"] = rvol(d["vol"])
    d["rvol_tick"] = rvol(d["tvol"])
    return d


def rvol(s: pd.Series, n: int = RVOL_N) -> pd.Series:
    """Volume ÷ mean of the previous n candles (needs ≥ n/2 of them to exist)."""
    base = s.shift(1).rolling(n, min_periods=n // 2).mean()
    return s / base


def latest(d: pd.DataFrame) -> dict:
    """The newest candle's volume reading, from whichever source has it."""
    if not len(d):
        return {}
    last = d.iloc[-1]
    out = {"comex": None, "rvol": None, "ticks": None, "rvol_tick": None, "source": None}
    if pd.notna(last.get("vol")):
        out |= {"comex": float(last["vol"]), "rvol": _f(last.get("rvol")), "source": "COMEX"}
    if pd.notna(last.get("tvol")):
        out |= {"ticks": float(last["tvol"]), "rvol_tick": _f(last.get("rvol_tick"))}
        if out["source"] is None:
            out["source"] = "ticks"
    filled = d["vol"].dropna()
    out["comex_until"] = str(filled.index[-1])[:16] if len(filled) else None
    out["label"] = d.attrs.get("vol_label", "COMEX")
    return out


def _f(x):
    return None if x is None or pd.isna(x) or not np.isfinite(x) else round(float(x), 2)
