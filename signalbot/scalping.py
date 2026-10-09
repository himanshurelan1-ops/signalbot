"""The most widely used scalping setups, as fixed rules the backtest can replay.

Every rule trades WITH the 200-EMA trend (the same regime filter as the
breakout and pullback rules) except the London breakout, which takes the
direction of the break. Entries, stops (2 ATR), targets (3 ATR), the time
stop and the spread cost all come from the same engine (strategy.backtest),
so the cards and the scoreboard compare like with like.

  ema 9/21 cross     EMA 9 crosses EMA 21 in the direction of the trend
  macd cross         MACD (12, 26, 9) crosses its signal line with the trend
  rsi bounce         RSI(14) dips under 30 and closes back above (mirror 70) with the trend
  bollinger squeeze  bands at their tightest 20% of the last 120 candles, then a close outside, with the trend
  supertrend flip    Supertrend (10, 3) flips with the trend
  vwap reclaim       price closes back above the day's VWAP after being below it, with the trend
  london breakout    first close beyond the Asian-session range (00:00–07:00 UTC) during 07:00–10:00 UTC
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema
from .symbols import pf

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

TEXT = {
    "ema 9/21 cross": "EMA 9/21 cross — in an uptrend (above the 200 EMA) buy when the 9 EMA crosses above the 21 EMA; "
                      "mirror for shorts in a downtrend.",
    "macd cross": "MACD cross — in an uptrend buy when MACD (12, 26, 9) crosses above its signal line; mirror for shorts.",
    "rsi bounce": "RSI bounce — in an uptrend buy when RSI(14) dips under 30 and closes back above it; "
                  "in a downtrend sell when it pokes over 70 and closes back below.",
    "bollinger squeeze": "Bollinger squeeze — when the 20-period bands are at their tightest 20% of the last 120 candles, "
                         "buy the first close above the upper band in an uptrend (below the lower band in a downtrend).",
    "supertrend flip": "Supertrend flip — Supertrend (10, 3) turning up in an uptrend is a buy; turning down in a downtrend a sell.",
    "vwap reclaim": "VWAP reclaim — in an uptrend buy when price closes back above the day's VWAP after a close below it; "
                    "mirror for shorts. VWAP resets 00:00 UTC (05:30 IST), weighted by COMEX volume when it is there.",
    "london breakout": "London breakout — the Asian session (00:00–07:00 UTC, 05:30–12:30 IST) sets a range; the first close "
                       "above it during 07:00–10:00 UTC (12:30–15:30 IST) is a buy, below it a sell. One trade a day.",
}
SETUPS = tuple(TEXT)
#: rules whose watch trigger is a price level that must be closed beyond
LEVEL_RULES = ("bollinger squeeze", "supertrend flip", "london breakout")


# ── indicators ──────────────────────────────────────────────────────────────

def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """(line, direction) — direction +1 up / -1 down."""
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()
    mid = (h + l) / 2
    up_b, dn_b = mid - mult * atr, mid + mult * atr
    line = np.full(len(c), np.nan)
    dirn = np.zeros(len(c), dtype=int)
    fu, fd = np.nan, np.nan
    for i in range(len(c)):
        if not np.isfinite(atr[i]):
            continue
        fu = up_b[i] if not np.isfinite(fu) or c[i - 1] < fu or up_b[i] > fu else fu
        fd = dn_b[i] if not np.isfinite(fd) or c[i - 1] > fd or dn_b[i] < fd else fd
        if dirn[i - 1] == 0:
            dirn[i] = 1 if c[i] >= mid[i] else -1
        elif dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < fu else 1
        else:
            dirn[i] = 1 if c[i] > fd else -1
        line[i] = fu if dirn[i] == 1 else fd
    return pd.Series(line, index=df.index), pd.Series(dirn, index=df.index)


def _utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx - IST_OFFSET


def vwap(d: pd.DataFrame) -> pd.Series:
    """Session VWAP, reset 00:00 UTC. Weighted by COMEX volume on days where it covers ≥80% of the
    candles (gaps filled with that day's median), otherwise equal weights (TWAP)."""
    tp = (d["high"] + d["low"] + d["close"]) / 3
    day = pd.Series(_utc(d.index).normalize(), index=d.index)
    v = d["vol"] if "vol" in d else pd.Series(np.nan, index=d.index)
    cover = v.notna().groupby(day).transform("mean")
    med = v.groupby(day).transform("median")
    w = v.fillna(med).where(cover >= 0.8, 1.0).fillna(1.0)
    return (tp * w).groupby(day).cumsum() / w.groupby(day).cumsum()


def add_indicators(d: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    c = d["close"]
    d["ema9"], d["ema21"] = ema(c, 9), ema(c, 21)
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    bw = (d["bb_up"] - d["bb_lo"]) / mid
    d["squeeze"] = bw <= bw.rolling(120, min_periods=60).quantile(0.2)
    d["st_line"], d["st_dir"] = supertrend(d)
    if intraday:
        d["vwap"] = vwap(d)
        utc = _utc(d.index)
        day = pd.Series(utc.normalize(), index=d.index)
        hr = pd.Series(utc.hour, index=d.index)
        asia = hr < 7
        d["asia_hi"] = d["high"].where(asia).groupby(day).transform("max").where(~asia)
        d["asia_lo"] = d["low"].where(asia).groupby(day).transform("min").where(~asia)
        d["london"] = (hr >= 7) & (hr < 10)
    return d


# ── signals ─────────────────────────────────────────────────────────────────

def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def signals(d: pd.DataFrame, setup: str) -> tuple[pd.Series, pd.Series]:
    up, dn = d["uptrend"], d["downtrend"]
    no = pd.Series(False, index=d.index)
    if setup == "ema 9/21 cross":
        return up & _cross_up(d["ema9"], d["ema21"]), dn & _cross_up(d["ema21"], d["ema9"])
    if setup == "macd cross":
        return up & _cross_up(d["macd"], d["macd_sig"]), dn & _cross_up(d["macd_sig"], d["macd"])
    if setup == "rsi bounce":
        r = d["rsi"]
        return up & (r.shift(1) < 30) & (r >= 30), dn & (r.shift(1) > 70) & (r <= 70)
    if setup == "bollinger squeeze":
        was = d["squeeze"].astype(float).shift(1).fillna(0).astype(bool)
        return (up & was & (d["close"] > d["bb_up"]) & (d["close"].shift(1) <= d["bb_up"].shift(1)),
                dn & was & (d["close"] < d["bb_lo"]) & (d["close"].shift(1) >= d["bb_lo"].shift(1)))
    if setup == "supertrend flip":
        s = d["st_dir"]
        return up & (s == 1) & (s.shift(1) == -1), dn & (s == -1) & (s.shift(1) == 1)
    if setup == "vwap reclaim":
        if "vwap" not in d:
            return no, no
        c, v = d["close"], d["vwap"]
        return up & (c.shift(1) < v.shift(1)) & (c > v), dn & (c.shift(1) > v.shift(1)) & (c < v)
    if setup == "london breakout":
        if "asia_hi" not in d:
            return no, no
        day = pd.Series(_utc(d.index).normalize(), index=d.index)
        brk_up = d["london"] & (d["close"] > d["asia_hi"])
        brk_dn = d["london"] & (d["close"] < d["asia_lo"])
        any_brk = (brk_up | brk_dn)
        first = any_brk & (any_brk.astype(int).groupby(day).cumsum() == 1)   # one trade a day
        return first & brk_up, first & brk_dn
    raise ValueError(setup)


# ── what would trigger next ─────────────────────────────────────────────────

def watch(setup: str, d: pd.DataFrame, p) -> dict:
    last = d.iloc[-1]
    a = float(last["atr"])
    upt = bool(last["uptrend"])
    side = "BUY" if upt else "SELL"
    w: dict = {"side": side}

    def lvl(x):
        x = float(x)
        return {"trigger": x, "stop": x - p.stop_atr * a if side == "BUY" else x + p.stop_atr * a,
                "target": x + p.target_atr * a if side == "BUY" else x - p.target_atr * a}

    f2 = lambda x: f"{pf(float(x))}"
    if setup == "ema 9/21 cross":
        above = last["ema9"] > last["ema21"]
        w["rule"] = (f"the 9 EMA crossing {'above' if upt else 'below'} the 21 EMA (now 9: {f2(last['ema9'])}, "
                     f"21: {f2(last['ema21'])}{' — already ' + ('above' if above else 'below') + ', waits for the next cross' if above == upt else ''})")
    elif setup == "macd cross":
        w["rule"] = (f"MACD crossing {'above' if upt else 'below'} its signal line "
                     f"(now {float(last['macd']):+.2f} vs {float(last['macd_sig']):+.2f})")
    elif setup == "rsi bounce":
        w["rule"] = (f"RSI(14) dipping under 30 and closing back above it" if upt else
                     f"RSI(14) poking over 70 and closing back below it") + f" (now {float(last['rsi']):.0f})"
    elif setup == "bollinger squeeze":
        if bool(last["squeeze"]):
            w |= lvl(last["bb_up"] if upt else last["bb_lo"])
            w["rule"] = f"a close {'above the upper' if upt else 'below the lower'} Bollinger band — the bands are squeezed now"
        else:
            w["rule"] = "a squeeze first (bands at their tightest 20% of the last 120 candles) — not squeezed now"
    elif setup == "supertrend flip":
        if int(last["st_dir"]) == (1 if upt else -1):
            w["rule"] = f"Supertrend is already {'up' if upt else 'down'} — waits for it to flip against and back"
        else:
            w |= lvl(last["st_line"])
            w["rule"] = f"a close {'above' if upt else 'below'} the Supertrend line (flips it {'up' if upt else 'down'})"
    elif setup == "vwap reclaim":
        if "vwap" in d and pd.notna(last.get("vwap")):
            w |= lvl(last["vwap"])
            w["rule"] = (f"a close {'below' if upt else 'above'} VWAP, then one back {'above' if upt else 'below'} it "
                         f"(VWAP now {f2(last['vwap'])})")
        else:
            w["rule"] = "intraday timeframes only"
    elif setup == "london breakout":
        if "asia_hi" in d and pd.notna(last.get("asia_hi")):
            hi, lo, c = float(last["asia_hi"]), float(last["asia_lo"]), float(last["close"])
            w["side"] = "BUY" if abs(hi - c) <= abs(c - lo) else "SELL"
            side = w["side"]
            w |= lvl(hi if side == "BUY" else lo)
            w["rule"] = (f"a close {'above' if side == 'BUY' else 'below'} the Asian range "
                         f"({f2(lo)} – {f2(hi)}) between 12:30 and 15:30 IST, first break of the day only")
        else:
            w["rule"] = "the Asian range (05:30–12:30 IST) is still forming, or this is not an intraday timeframe"
    return w
