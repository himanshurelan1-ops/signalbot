"""The candle that is forming right now, so the chart moves between closes.

The rules only ever see closed candles (see data.drop_incomplete). This module
is for the eyes: it reads Twelve Data's 1-minute bars for the current session and
folds the ones since the last closed candle into one in-progress OHLC bar,
which the page redraws every few seconds — the way a charting app does.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pandas as pd

from .data import SYMBOLS, TIMEFRAMES, BAR_MINUTES, DAILY, load

log = logging.getLogger("signalbot.live")

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_root_for_live: str | None = None
_lock = threading.Lock()
TTL = 1.0          # seconds between re-reads of the shared 1-minute cache


def minute_bars(symbol: str) -> pd.DataFrame:
    with _lock:
        hit = _cache.get(symbol)
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
    from . import stream, twelvedata
    meta = SYMBOLS[symbol]
    df = twelvedata.minute_bars(Path(_root_for_live or "."), symbol=meta["td"],
                                kind=meta.get("kind", "metal"))[["open", "high", "low", "close", "volume"]]
    st = stream.get()
    cur = st.current_bar(symbol) if st else None
    if cur is not None and (df.empty or cur["time"] > df.index[-1]):   # the minute in progress, from ticks
        df = pd.concat([df, pd.DataFrame([{k: cur[k] for k in ("open", "high", "low", "close", "volume")}],
                                         index=pd.DatetimeIndex([cur["time"]], name="time"))])
    if df.empty:
        raise RuntimeError("no 1-minute data")
    with _lock:
        _cache[symbol] = (time.time(), df)
    return df


def forming_candle(root: Path, symbol: str, tf: str) -> dict:
    """{'time', 'open','high','low','close','last_price','last_time','lag_min'} for the bar in progress."""
    global _root_for_live
    _root_for_live = str(root)
    m1 = minute_bars(symbol)
    if "volume" not in m1:
        m1 = m1.assign(volume=0.0)
    closed = load(symbol, tf, root, refresh=False)
    last = closed.index[-1]
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    if tf in DAILY:
        span = pd.Timedelta(days=7 if tf == "1w" else 1)
        start = last + span                          # daily / weekly candles are labelled by their start
        label = str(start.date())
        seg = m1[m1.index > start]
        t_out = label
    else:
        # the bucket that holds the newest 1-minute bar — on a delayed feed that
        # is a few minutes behind the clock, and that is what we can honestly show
        step = pd.Timedelta(minutes=BAR_MINUTES[tf])
        newest = m1.index[-1]
        start = last + step
        while start + step <= newest:
            start += step
        seg = m1[(m1.index >= start) & (m1.index < start + step)]
        t_out = int(start.tz_localize("UTC").timestamp())
    last_px = float(m1["close"].iloc[-1])
    last_t = m1.index[-1]
    out = {"symbol": symbol, "tf": tf, "last_price": last_px, "last_time": str(last_t)[:16],
           "lag_min": int(max(0, (now - last_t).total_seconds() // 60)), "time": t_out,
           "closed_until": str(last)[:16], "bars": []}
    if not seg.empty:
        out |= {"open": float(seg["open"].iloc[0]), "high": float(seg["high"].max()),
                "low": float(seg["low"].min()), "close": float(seg["close"].iloc[-1]),
                "ticks": float(seg["volume"].sum())}
    # every bucket after the last cached closed candle that the day's 1-minute
    # bars can rebuild — so the chart has no hole while the main cache catches up
    if tf not in DAILY:
        step = pd.Timedelta(minutes=BAR_MINUTES[tf])
        b = last + step
        while b <= start:
            seg_b = m1[(m1.index >= b) & (m1.index < b + step)]
            if not seg_b.empty:
                out["bars"].append({"time": int(b.tz_localize("UTC").timestamp()),
                                    "open": float(seg_b["open"].iloc[0]), "high": float(seg_b["high"].max()),
                                    "low": float(seg_b["low"].min()), "close": float(seg_b["close"].iloc[-1]),
                                    "ticks": float(seg_b["volume"].sum())})
            b += step
    elif "open" in out:
        out["bars"].append({k: out[k] for k in ("time", "open", "high", "low", "close", "ticks")})
    return out
