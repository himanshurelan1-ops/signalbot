"""Rule-based setups, and the bar-by-bar backtest that measures them.

Nothing here predicts. Each setup is a fixed rule — "in an uptrend, buy a
close above the last 20 days' high" — and the backtest replays that rule
across every bar we have, filling at the next open, exiting at the stop,
the target, or after a time limit. What comes out is the setup's actual
record: how often the target was hit before the stop, and what the average
trade was worth in R (multiples of the initial risk). The current signal is
reported with exactly those numbers beside it, because they are the only
honest measure of how much to trust it.

Conservative conventions, so the numbers err low rather than high:
  * entries fill at the next bar's open, never at the signal close;
  * if a bar touches both stop and target, the stop is assumed first;
  * gaps through a level fill at the open, not at the level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .indicators import atr, donchian, ema, rsi

Side = Literal["BUY", "SELL"]


@dataclass
class Params:
    ema_trend: int = 200       # regime filter
    ema_pull: int = 20         # pullback reference
    donchian: int = 20         # breakout lookback
    atr_n: int = 14
    stop_atr: float = 2.0      # stop distance in ATRs
    target_atr: float = 3.0    # target distance in ATRs (1.5 R)
    max_bars: int = 20         # time stop
    rsi_n: int = 14
    intraday: bool = False     # square off at the day's last bar; no late entries
    last_entry: str = "14:45"  # intraday: no new positions after this time
    timed: bool = False        # intraday timeframe: label trades with time, not just the date
    vol_mult: float = 1.5      # volume breakout: signal candle's volume ≥ this × the 20-candle average
    cost: float = 0.30         # round-trip spread + commission, in price units ($ for gold), taken off every trade


@dataclass
class Trade:
    setup: str
    side: str
    signal_time: str
    entry_time: str
    entry: float
    stop: float
    target: float
    exit_time: str | None = None
    exit: float | None = None
    reason: str = "open"       # target | stop | time | open
    bars: int = 0
    r: float = 0.0             # result in R; for an open trade, mark-to-market
    pct: float = 0.0


@dataclass
class Stats:
    setup: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0        # share of closed trades that ended positive
    target_rate: float = 0.0     # share that reached the full target
    avg_r: float = 0.0
    profit_factor: float | None = 0.0
    total_r: float = 0.0
    max_dd_r: float = 0.0
    avg_bars: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    first: str = ""
    last: str = ""
    buys: int = 0
    sells: int = 0
    buy_win_rate: float = 0.0
    sell_win_rate: float = 0.0


@dataclass
class Signal:
    """What the rules say right now for one setup."""
    setup: str
    side: str | None            # BUY / SELL / None
    status: str                 # "signal" (fires on the last bar) | "open" | "none"
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    as_of: str = ""
    note: str = ""
    trade: Trade | None = None  # the live trade when status == "open"
    watch: dict = field(default_factory=dict)   # levels that WOULD trigger


# ── indicators & raw signals ────────────────────────────────────────────────

def enrich(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["ema_trend"] = ema(d["close"], p.ema_trend)
    d["ema_pull"] = ema(d["close"], p.ema_pull)
    d["atr"] = atr(d, p.atr_n)
    d["rsi"] = rsi(d["close"], p.rsi_n)
    d["dc_high"], d["dc_low"] = donchian(d, p.donchian)
    d.attrs["vol_mult"] = p.vol_mult
    from . import scalping
    scalping.add_indicators(d, intraday=p.intraday or p.timed)
    d["uptrend"] = d["close"] > d["ema_trend"]
    d["downtrend"] = d["close"] < d["ema_trend"]
    return d


def signals(d: pd.DataFrame, setup: str) -> pd.Series:
    """+1 for a BUY signal on that bar's close, -1 for SELL, 0 otherwise."""
    if setup == "breakout":
        buy = d["uptrend"] & (d["close"] > d["dc_high"])
        sell = d["downtrend"] & (d["close"] < d["dc_low"])
    elif setup == "pullback":
        prev_below = d["close"].shift(1) < d["ema_pull"].shift(1)
        prev_above = d["close"].shift(1) > d["ema_pull"].shift(1)
        buy = d["uptrend"] & prev_below & (d["close"] > d["ema_pull"]) & (d["rsi"] > 50)
        sell = d["downtrend"] & prev_above & (d["close"] < d["ema_pull"]) & (d["rsi"] < 50)
    elif setup == "volume breakout":
        rv = _rvol(d)
        hot = rv >= p_vol_mult(d)
        buy = d["uptrend"] & (d["close"] > d["dc_high"]) & hot
        sell = d["downtrend"] & (d["close"] < d["dc_low"]) & hot
    else:
        from . import scalping
        buy, sell = scalping.signals(d, setup)
    out = pd.Series(0, index=d.index, dtype=int)
    out[buy.fillna(False)] = 1
    out[sell.fillna(False)] = -1
    out[d["atr"].isna()] = 0
    return out


def _rvol(d: pd.DataFrame) -> pd.Series:
    """Relative volume: COMEX where it exists, else live tick volume (each vs its own 20-candle average)."""
    if "rvol" not in d:
        return pd.Series(np.nan, index=d.index)
    rv = d["rvol"]
    if "rvol_tick" in d:
        rv = rv.fillna(d["rvol_tick"])
    return rv


def p_vol_mult(d: pd.DataFrame) -> float:
    return float(d.attrs.get("vol_mult", 1.5))


from .scalping import SETUPS as _SCALP, TEXT as _SCALP_TEXT  # noqa: E402

SETUPS = ("breakout", "pullback", "volume breakout") + _SCALP
SETUP_TEXT = {
    "breakout": "Trend breakout — in an uptrend (close above the 200-bar EMA), buy a close "
                "above the previous 20 bars' high; mirror for shorts in a downtrend.",
    "pullback": "Trend pullback — in an uptrend, buy when price dips below the 20-bar EMA "
                "and closes back above it with RSI > 50; mirror for shorts.",
    "volume breakout": "Breakout with volume — the trend breakout, but only when the breakout candle trades "
                       "at least 1.5× its 20-candle average volume (COMEX gold futures; live tick volume "
                       "when COMEX hasn't printed yet). Mirror for shorts.",
    **_SCALP_TEXT,
}


# ── backtest ────────────────────────────────────────────────────────────────

def _ts(t, p: Params) -> str:
    return str(t)[:16] if (p.intraday or p.timed) else str(t.date())


def backtest(d: pd.DataFrame, setup: str, p: Params,
             sig_override: np.ndarray | None = None) -> list[Trade]:
    sig = signals(d, setup).to_numpy() if sig_override is None else np.asarray(sig_override)
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = d["atr"].to_numpy()
    idx = d.index
    n = len(d)
    if p.intraday:
        days = idx.normalize()
        eod = np.r_[(days[1:] != days[:-1]), True]          # last bar of each session
        hhmm = idx.strftime("%H:%M")
        late = np.array([t >= p.last_entry for t in hhmm])  # too late to open a position
    else:
        eod = np.zeros(n, dtype=bool)
        late = np.zeros(n, dtype=bool)
    trades: list[Trade] = []
    i = 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or (p.intraday and (eod[i] or late[i])):
            i += 1
            continue
        side = "BUY" if s > 0 else "SELL"
        e = i + 1
        entry = o[e]
        risk = p.stop_atr * a[i]
        if not np.isfinite(risk) or risk <= 0:
            i += 1
            continue
        stop = entry - risk if s > 0 else entry + risk
        target = entry + p.target_atr * a[i] if s > 0 else entry - p.target_atr * a[i]
        t = Trade(setup, side, _ts(idx[i], p), _ts(idx[e], p),
                  float(entry), float(stop), float(target))
        j = e
        while j < n:
            if s > 0:
                hit_stop = l[j] <= stop
                hit_tgt = h[j] >= target
                stop_px = min(o[j], stop) if hit_stop else None
                tgt_px = max(o[j], target) if hit_tgt else None
            else:
                hit_stop = h[j] >= stop
                hit_tgt = l[j] <= target
                stop_px = max(o[j], stop) if hit_stop else None
                tgt_px = min(o[j], target) if hit_tgt else None
            if hit_stop:                       # stop first when both touch
                t.exit, t.reason = float(stop_px), "stop"
            elif hit_tgt:
                t.exit, t.reason = float(tgt_px), "target"
            elif p.intraday and eod[j]:
                t.exit, t.reason = float(c[j]), "eod"     # flat by the close, always
            elif j - e + 1 >= p.max_bars:
                t.exit, t.reason = float(c[j]), "time"
            if t.reason != "open":
                t.exit_time = _ts(idx[j], p)
                t.bars = j - e + 1
                break
            j += 1
        if t.reason == "open":                 # still live on the last bar
            t.bars = n - e
            t.exit = float(c[-1])
        move = ((t.exit - entry) if s > 0 else (entry - t.exit)) - p.cost
        t.r = float(move / risk)
        t.pct = float(move / entry * 100)
        trades.append(t)
        i = (j if t.reason != "open" else n) + 1   # no overlapping positions
    return trades


def stats(trades: list[Trade], setup: str) -> Stats:
    closed = [t for t in trades if t.reason != "open"]
    st = Stats(setup=setup, trades=len(closed))
    if not closed:
        return st
    rs = np.array([t.r for t in closed])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    st.wins = int(len(wins))
    st.win_rate = float(len(wins) / len(rs))
    st.target_rate = float(sum(t.reason == "target" for t in closed) / len(closed))
    st.avg_r = float(rs.mean())
    st.profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None   # None = no losses
    st.total_r = float(rs.sum())
    eq = np.cumsum(rs)
    st.max_dd_r = float((np.maximum.accumulate(eq) - eq).max())
    st.avg_bars = float(np.mean([t.bars for t in closed]))
    st.best_r, st.worst_r = float(rs.max()), float(rs.min())
    st.first, st.last = closed[0].entry_time, closed[-1].exit_time or ""
    b = [t for t in closed if t.side == "BUY"]
    s = [t for t in closed if t.side == "SELL"]
    st.buys, st.sells = len(b), len(s)
    st.buy_win_rate = float(sum(t.r > 0 for t in b) / len(b)) if b else 0.0
    st.sell_win_rate = float(sum(t.r > 0 for t in s) / len(s)) if s else 0.0
    return st


def equity_curve(trades: list[Trade]) -> list[dict]:
    eq = 0.0
    out = []
    for t in trades:
        if t.reason == "open":
            continue
        eq += t.r
        out.append({"time": t.exit_time, "r": round(eq, 2)})
    return out


# ── what the rules say now ──────────────────────────────────────────────────

def current(d: pd.DataFrame, setup: str, p: Params, trades: list[Trade]) -> Signal:
    last = d.iloc[-1]
    as_of = _ts(d.index[-1], p)
    live = trades[-1] if trades and trades[-1].reason == "open" else None
    sig = int(signals(d, setup).iloc[-1])
    a = float(last["atr"])

    watch: dict = {}
    if setup in ("breakout", "volume breakout"):
        if bool(last["uptrend"]):
            lvl = float(last["dc_high"])
            watch = {"side": "BUY", "trigger": lvl, "rule": "a close above the 20-bar high" + (
                         f" on ≥{p.vol_mult:g}× average volume" if setup == "volume breakout" else ""),
                     "stop": lvl - p.stop_atr * a, "target": lvl + p.target_atr * a}
        else:
            lvl = float(last["dc_low"])
            watch = {"side": "SELL", "trigger": lvl, "rule": "a close below the 20-bar low" + (
                         f" on ≥{p.vol_mult:g}× average volume" if setup == "volume breakout" else ""),
                     "stop": lvl + p.stop_atr * a, "target": lvl - p.target_atr * a}
    elif setup != "pullback":
        from . import scalping
        watch = scalping.watch(setup, d, p)
    else:
        lvl = float(last["ema_pull"])
        if bool(last["uptrend"]):
            watch = {"side": "BUY", "trigger": lvl,
                     "rule": "a dip below the 20-bar EMA followed by a close back above it (RSI > 50)",
                     "stop": lvl - p.stop_atr * a, "target": lvl + p.target_atr * a}
        else:
            watch = {"side": "SELL", "trigger": lvl,
                     "rule": "a bounce above the 20-bar EMA followed by a close back below it (RSI < 50)",
                     "stop": lvl + p.stop_atr * a, "target": lvl - p.target_atr * a}
    watch["atr"] = a
    watch["trend"] = "up" if bool(last["uptrend"]) else "down"

    if live:
        return Signal(setup, live.side, "open", live.entry, live.stop, live.target, as_of,
                      f"live since {live.entry_time}: {live.r:+.2f}R so far", live, watch)
    if sig != 0 and p.intraday and d.index[-1].strftime("%H:%M") >= p.last_entry:
        return Signal(setup, None, "none", None, None, None, as_of,
                      f"rule fired at {d.index[-1].strftime('%H:%M')} but it's past the "
                      f"{p.last_entry} cutoff — no new positions this late in the session",
                      None, watch)
    if sig != 0:
        side = "BUY" if sig > 0 else "SELL"
        entry = float(last["close"])          # the next open is unknown; close is the estimate
        stop = entry - p.stop_atr * a if sig > 0 else entry + p.stop_atr * a
        target = entry + p.target_atr * a if sig > 0 else entry - p.target_atr * a
        return Signal(setup, side, "signal", entry, stop, target, as_of,
                      "fires on the latest close — enter at the next open; levels move with the fill",
                      None, watch)
    return Signal(setup, None, "none", None, None, None, as_of, "no trade", None, watch)


def to_dict(obj) -> dict:
    return asdict(obj)
