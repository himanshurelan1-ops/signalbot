"""Trendlines and channels, fitted to swing pivots, and the signals they give.

A resistance line is the straight line through two or more swing highs that
price has respected since (no close above it) and that is still within 3 ATR
of price (a steep line that a sideways market has left behind is retired); support is the mirror through
swing lows. Of all candidate pairs among the recent pivots, the line with the
most touches wins, most recent on ties. When the two lines run parallel the
structure is a channel.

Two rules are backtested on them with the same engine as everything else:

  * trendline break — a close through the line (after a close on the other
    side) in the direction of the break;
  * channel bounce  — inside a channel, a bar that touches one line and
    closes back inside, traded toward the other line.

Lines are recomputed only when a new pivot is confirmed, so the historical
replay sees exactly what a trader watching the chart would have seen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .indicators import atr as _atr
from .patterns import Pivot, pivots
from .strategy import Params, Signal, Trade, backtest, equity_curve, stats
from .symbols import pf

TOUCH_ATR = 0.4       # a pivot within this many ATRs of the line counts as a touch
BREAK_ATR = 0.1       # a close this far through the line is a break
MAX_PIVOTS = 7        # how many recent pivots of each kind to consider
MAX_DIST_ATR = 3.0    # a line further than this from price is retired: price has walked away from it


@dataclass
class Line:
    kind: str               # "R" resistance / "S" support
    slope: float            # price per bar
    intercept: float
    touches: int
    from_i: int             # first pivot on the line
    to_i: int               # last pivot on the line

    def at(self, i: int) -> float:
        return self.slope * i + self.intercept


def _best_line(pts: list[Pivot], closes: np.ndarray, now_i: int, atr_now: float, kind: str) -> Line | None:
    pts = pts[-MAX_PIVOTS:]
    if len(pts) < 2 or not np.isfinite(atr_now) or atr_now <= 0:
        return None
    tol = TOUCH_ATR * atr_now
    best: Line | None = None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            pa, pb = pts[a], pts[b]
            if pb.i == pa.i:
                continue
            slope = (pb.price - pa.price) / (pb.i - pa.i)
            icpt = pa.price - slope * pa.i
            xs = np.arange(pa.i, now_i + 1)
            line = slope * xs + icpt
            seg = closes[pa.i:now_i + 1]
            # respected since the first touch: no close through the line
            if kind == "R" and (seg > line + tol).any():
                continue
            if kind == "S" and (seg < line - tol).any():
                continue
            if abs(line[-1] - closes[now_i]) > MAX_DIST_ATR * atr_now:
                continue                             # already far from price — not a level anyone trades
            touches = sum(1 for p in pts if abs(p.price - (slope * p.i + icpt)) <= tol)
            cand = Line(kind, float(slope), float(icpt), touches, pa.i, pb.i)
            if best is None or (cand.touches, cand.to_i) > (best.touches, best.to_i):
                best = cand
    return best


def is_channel(r: Line | None, s: Line | None, atr_now: float) -> bool:
    if not r or not s:
        return False
    scale = max(abs(r.slope), abs(s.slope), 0.02 * atr_now)
    return abs(r.slope - s.slope) <= 0.3 * scale


def rolling_signals(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    """Replay the chart bar by bar: lines as they were visible, breaks and
    bounces as they happened. Returns the signal arrays plus the lines now."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = _atr(df, 14).to_numpy()
    n = len(df)
    pivs = pivots(df, left, right)
    by_conf: dict[int, list[Pivot]] = {}
    for p in pivs:
        by_conf.setdefault(p.confirmed, []).append(p)

    brk = np.zeros(n, dtype=int)
    bnc = np.zeros(n, dtype=int)
    res: Line | None = None
    sup: Line | None = None
    highs: list[Pivot] = []
    lows: list[Pivot] = []
    history: list[dict] = []          # when a line was set / broken, for the chart

    for i in range(1, n):
        if i in by_conf:
            for p in by_conf[i]:
                (highs if p.kind == "H" else lows).append(p)
            res = _best_line(highs, c, i, a[i], "R")
            sup = _best_line(lows, c, i, a[i], "S")
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        tol = BREAK_ATR * a[i]
        # retire a line once price has drifted far from it (a steep line left behind by a sideways market)
        if res is not None and abs(res.at(i) - c[i]) > MAX_DIST_ATR * a[i]:
            res = None
        if sup is not None and abs(sup.at(i) - c[i]) > MAX_DIST_ATR * a[i]:
            sup = None
        chan = is_channel(res, sup, a[i])
        if res is not None:
            if c[i] > res.at(i) + tol and c[i - 1] <= res.at(i - 1) + tol:
                brk[i] = 1
                history.append({"i": i, "what": "break up", "price": float(res.at(i))})
                res = None
            elif chan and h[i] >= res.at(i) - TOUCH_ATR * a[i] and c[i] < res.at(i) and c[i] < o[i]:
                bnc[i] = -1
        if sup is not None:
            if c[i] < sup.at(i) - tol and c[i - 1] >= sup.at(i - 1) - tol:
                brk[i] = -1
                history.append({"i": i, "what": "break down", "price": float(sup.at(i))})
                sup = None
            elif chan and l[i] <= sup.at(i) + TOUCH_ATR * a[i] and c[i] > sup.at(i) and c[i] > o[i]:
                bnc[i] = 1
    return {"break": brk, "bounce": bnc, "resistance": res, "support": sup,
            "channel": is_channel(res, sup, a[-1]) if n else False, "atr": float(a[-1]) if n else 0.0,
            "events": history[-20:]}


LINE_TEXT = {
    "trendline break": "a close through a trendline that price had respected (after a close on the other side) — "
                       "traded in the direction of the break",
    "channel bounce": "inside a parallel channel, a bar that touches one line and closes back inside — "
                      "traded toward the other line",
}


def _signal_now(name: str, sig: np.ndarray, d: pd.DataFrame, p: Params, trades: list[Trade],
                watch: dict) -> Signal:
    from .strategy import _ts
    last = d.iloc[-1]
    a = float(last["atr"])
    as_of = _ts(d.index[-1], p)
    live = trades[-1] if trades and trades[-1].reason == "open" else None
    if live:
        return Signal(name, live.side, "open", live.entry, live.stop, live.target, as_of,
                      f"live since {live.entry_time}: {live.r:+.2f}R so far", live, watch)
    s = int(sig[-1])
    if s != 0:
        side = "BUY" if s > 0 else "SELL"
        entry = float(last["close"])
        stop = entry - p.stop_atr * a if s > 0 else entry + p.stop_atr * a
        target = entry + p.target_atr * a if s > 0 else entry - p.target_atr * a
        return Signal(name, side, "signal", entry, stop, target, as_of,
                      "fired on the latest close — enter at the next candle", None, watch)
    return Signal(name, None, "none", None, None, None, as_of, "no trade", None, watch)


def analyse(d: pd.DataFrame, p: Params, extend: int = 5, long_only: bool = False) -> dict:
    """Lines drawn for the chart + the two line-based setups with their records.
    `d` must already be enriched (has atr / ema columns)."""
    rs = rolling_signals(d)
    n = len(d)
    a = rs["atr"]
    out_lines = []
    for ln in (rs["resistance"], rs["support"]):
        if ln is None:
            continue
        out_lines.append({
            "kind": "resistance" if ln.kind == "R" else "support",
            "touches": ln.touches,
            "from": {"i": ln.from_i, "price": ln.at(ln.from_i)},
            "to": {"i": n - 1 + extend, "price": ln.at(n - 1 + extend)},
            "now": ln.at(n - 1),
            "slope_atr_per_100": ln.slope * 100 / a if a else 0.0,
        })
    watch = {"trend": "up" if bool(d["uptrend"].iloc[-1]) else "down"}
    if rs["resistance"] is not None:
        lvl = rs["resistance"].at(n - 1)
        watch |= {"side": "BUY", "trigger": lvl, "rule": f"a close above the resistance line (now ≈ {pf(lvl)})",
                  "stop": lvl - p.stop_atr * a, "target": lvl + p.target_atr * a}
    if rs["support"] is not None:
        lvl = rs["support"].at(n - 1)
        watch_s = {"side": "SELL", "trigger": lvl, "rule": f"a close below the support line (now ≈ {pf(lvl)})",
                   "stop": lvl + p.stop_atr * a, "target": lvl - p.target_atr * a}
        # show the nearer line as the primary watch
        if "trigger" not in watch or abs(lvl - d["close"].iloc[-1]) < abs(watch["trigger"] - d["close"].iloc[-1]):
            watch |= watch_s

    setups = []
    for name, key in (("trendline break", "break"), ("channel bounce", "bounce")):
        sig_arr = rs[key].copy()
        if long_only:
            sig_arr[sig_arr < 0] = 0
        trades = backtest(d, name, p, sig_override=sig_arr)
        sig = _signal_now(name, sig_arr, d, p, trades, watch if key == "break" else {"trend": watch["trend"]})
        setups.append({"setup": name, "text": LINE_TEXT[name], "caveat": "",
                       "signal": asdict(sig), "stats": asdict(stats(trades, name)),
                       "equity": equity_curve(trades),
                       "trades": [asdict(t) for t in trades[-12:]][::-1],
                       "all_trades": [{"entry_time": t.entry_time, "r": t.r, "reason": t.reason} for t in trades]})
    return {"lines": out_lines, "channel": bool(rs["channel"]), "events": rs["events"], "setups": setups}
