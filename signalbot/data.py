"""Candles for every market on the watchlist, from Twelve Data, cached on disk in data/."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("signalbot.data")

from .symbols import SYMBOLS  # noqa: E402  (live registry: gold + whatever you add on the page)


def source_for(root: Path, symbol: str) -> str:
    """Every instrument here comes from Twelve Data."""
    return "twelvedata"


def available_symbols(root: Path) -> dict:
    return dict(SYMBOLS)

#: timeframes we support
TIMEFRAMES = {
    "1m":  {"label": "1 min",  "short": "1m"},
    "5m":  {"label": "5 min",  "short": "5m"},
    "15m": {"label": "15 min", "short": "15m"},
    "30m": {"label": "30 min", "short": "30m"},
    "1h":  {"label": "1 hour", "short": "1H"},
    "4h":  {"label": "4 hour", "short": "4H"},
    "1d":  {"label": "Daily",  "short": "1D"},
    "1w":  {"label": "Weekly", "short": "1W"},
}
#: timeframes built from a lower one instead of downloaded (no extra API credits)
DERIVED = {"30m": "15m", "4h": "1h", "1w": "1d"}
#: candles labelled by date, not time
DAILY = ("1d", "1w")


def is_intraday(tf: str) -> bool:
    return tf not in DAILY



def _cache_path(root: Path, symbol: str, interval: str) -> Path:
    return root / "data" / f"{symbol}_{interval}.csv"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index.name = "time"
    return df[~df.index.duplicated(keep="last")].sort_index()


BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def drop_incomplete(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None,
                    symbol: str = "XAUUSD") -> pd.DataFrame:
    """The feed's last bar is the one still forming. Rules must
    only ever see closed candles, so that bar is dropped until it completes."""
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    last = df.index[-1]
    if interval == "1d":
        closes_at = last + pd.Timedelta(days=1)        # a daily candle is labelled by its start
    elif interval == "1w":                             # Monday-labelled; done when the week's trading ends
        crypto = SYMBOLS.get(symbol, {}).get("kind") == "crypto"
        closes_at = last + pd.Timedelta(days=7 if crypto else 5)
    else:
        closes_at = last + pd.Timedelta(minutes=BAR_MINUTES[interval])
    return df.iloc[:-1] if now < closes_at else df


#: how long a cached file is trusted before topping up. Intraday top-ups are
#: built from the shared 1-minute cache (twelvedata.REFRESH), so they cost no
#: extra credits; the daily candle is asked for directly, once an hour.
MAX_AGE_LIVE = {"1d": 60 * 60, "1h": 30, "15m": 30, "5m": 30}
MINUTE_HISTORY = 8000          # 1-minute candles wanted for the 1 min chart and its backtest (~1 week)


def load(symbol: str, interval: str, root: Path, refresh: bool = True) -> pd.DataFrame:
    """Cached Twelve Data bars for `symbol`; topped up when the cache is stale.
    The bar still in progress is never returned (see drop_incomplete)."""
    meta = SYMBOLS[symbol]
    kind = meta.get("kind", "metal")
    from . import twelvedata
    if interval in DERIVED:
        base = load(symbol, DERIVED[interval], root, refresh=refresh)     # closed candles only
        out = twelvedata.resample_tf(base, interval, kind)
        return drop_incomplete(out, interval, symbol=symbol)
    if interval == "1m":                     # the shared 1-minute cache the stream and the other timeframes use
        m1 = twelvedata.minute_history(root, symbol=meta["td"], kind=kind, want=MINUTE_HISTORY, refresh=refresh)
        return drop_incomplete(twelvedata.market_hours_only(m1, "1m", kind), "1m", symbol=symbol)
    p = _cache_path(root, symbol, interval)
    cached = None
    if p.exists():
        try:
            cached = pd.read_csv(p, index_col="time", parse_dates=True)
        except Exception:
            cached = None
        fresh = time.time() - p.stat().st_mtime < MAX_AGE_LIVE[interval]
        from .twelvedata import market_closed
        if fresh and cached is not None and not cached.empty and not market_closed(kind=kind):
            # the cached file ends with the candle that was forming when it was written:
            # once that candle has closed, top up straight away so the rules see it
            step = pd.Timedelta(days=1) if interval == "1d" else pd.Timedelta(minutes=BAR_MINUTES[interval])
            fresh = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None) < cached.index[-1] + step
        if cached is not None:
            from .twelvedata import market_hours_only
            cached = market_hours_only(cached, interval, kind)
        if cached is not None and (fresh or not refresh):
            return drop_incomplete(cached, interval, symbol=symbol)
    try:
        from . import twelvedata
        # backfill years on first use; afterwards replace the last (possibly
        # forming) bar and append the new ones
        df = twelvedata.market_hours_only(twelvedata.update(root, interval, cached, symbol=meta["td"], kind=kind),
                                          interval, kind)
    except Exception as e:
        if cached is not None:
            log.warning("%s %s: Twelve Data top-up failed (%s) — using cached data", symbol, interval, e)
            return drop_incomplete(cached, interval, symbol=symbol)
        raise
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return drop_incomplete(df, interval, symbol=symbol)
