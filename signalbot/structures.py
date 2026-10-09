"""Channels and wedges at three swing sizes, Fibonacci retracements, and the rules built on them.

Swing sizes (bars each side of a pivot):  minor 3 · intermediate 8 · major 21. On every timeframe each
size gets its best resistance line and best support line through recent pivots (the line price has
respected since its first touch, with the most touches). The pair is classified:

  * parallel                     → channel (rising / falling / sideways)
  * converging, both down        → falling wedge  (usually breaks up)
  * converging, both up          → rising wedge   (usually breaks down)
  * converging, opposite slopes  → triangle
  * diverging                    → broadening

Fibonacci: the last completed intermediate swing (low → high or high → low, at least 3 ATR) with the
23.6 / 38.2 / 50 / 61.8 / 78.6 % retracements and the 127.2 / 161.8 % extensions as targets. The
50–61.8 % band is the "golden zone".

Two rules, replayed bar by bar with only the pivots visible at the time and backtested with the same
engine as every other rule:

  * wedge breakout   — a close out of a wedge or triangle (intermediate swings), in the break's direction
  * fib golden zone  — after a swing of ≥ 3 ATR, the first candle that dips into its 50–78.6 % retracement
                       and closes back above 61.8 % in the swing's direction (mirror for down swings)
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .indicators import atr as _atr
from .lines import Line, _signal_now
from .patterns import Pivot, pivots
from .strategy import Params, backtest, equity_curve, stats

SCALES = (("minor", 3, 6, 3.0), ("intermediate", 8, 8, 6.0), ("major", 21, 8, 12.0))   # name, k, pivots kept, max ATR away
TOUCH_ATR = 0.4
FIB = (0.236, 0.382, 0.5, 0.618, 0.786)
EXT = (1.272, 1.618)
MIN_LEG_ATR = 3.0

TEXT = {
    "wedge breakout": "a close out of a wedge or triangle drawn through the intermediate swings (8 bars each side) — "
                      "traded in the direction of the break; falling wedges usually break up, rising wedges down",
    "fib golden zone": "after a swing of at least 3 ATR, the first candle that dips into the 50–78.6% retracement and "
                       "closes back beyond 61.8% in the swing's direction — the classic Fibonacci pullback entry",
}
SETUPS = tuple(TEXT)


def best_line(pts: list[Pivot], closes: np.ndarray, now_i: int, a: float, kind: str,
              keep: int, max_dist: float, grace: int = 0) -> Line | None:
    """The line through two of the last `keep` pivots that price has respected since, with the most touches.
    `grace`: closes through the line in the last few bars are allowed, so a line that just broke is still shown."""
    pts = pts[-keep:]
    if len(pts) < 2 or not np.isfinite(a) or a <= 0:
        return None
    tol = TOUCH_ATR * a
    best = None
    for x in range(len(pts)):
        for y in range(x + 1, len(pts)):
            pa, pb = pts[x], pts[y]
            slope = (pb.price - pa.price) / (pb.i - pa.i)
            icpt = pa.price - slope * pa.i
            line = slope * np.arange(pa.i, now_i + 1) + icpt
            m = max(1, len(line) - grace)
            seg = closes[pa.i:pa.i + m]
            if (kind == "R" and (seg > line[:m] + tol).any()) or (kind == "S" and (seg < line[:m] - tol).any()):
                continue
            if abs(line[-1] - closes[now_i]) > max_dist * a:
                continue
            touches = sum(1 for p in pts if p.i >= pa.i and abs(p.price - (slope * p.i + icpt)) <= tol)
            cand = Line(kind, float(slope), float(icpt), touches, pa.i, pb.i)
            if best is None or (cand.touches, cand.to_i - cand.from_i) > (best.touches, best.to_i - best.from_i):
                best = cand
    return best


def shape(r: Line | None, s: Line | None, a: float, now_i: int) -> str | None:
    """channel / wedge / triangle / broadening, from the two lines' slopes (ATR per bar)."""
    if not r or not s or a <= 0:
        return None
    rs, ss = r.slope / a, s.slope / a
    flat = 0.004                                    # < 0.4 ATR per 100 bars counts as flat
    width_now = r.at(now_i) - s.at(now_i)
    if width_now <= 0:
        return None
    scale = max(abs(rs), abs(ss), flat)
    if abs(rs - ss) <= 0.3 * scale:
        return "rising channel" if (rs + ss) / 2 > flat else "falling channel" if (rs + ss) / 2 < -flat else "sideways channel"
    if rs < ss:                                     # lines converge
        if rs < -flat and ss < -flat:
            return "falling wedge"
        if rs > flat and ss > flat:
            return "rising wedge"
        return "triangle"
    return "broadening"


def _scale_lines(d: pd.DataFrame, k: int, keep: int, max_dist: float):
    c = d["close"].to_numpy()
    a = float(d["atr"].iloc[-1])
    n = len(d)
    pv = [p for p in pivots(d, k, k) if p.confirmed <= n - 1]
    hs = [p for p in pv if p.kind == "H"]
    ls = [p for p in pv if p.kind == "L"]
    g = max(3, k)                                   # a line broken within the last k bars is still drawn, marked broken
    return (best_line(hs, c, n - 1, a, "R", keep, max_dist, grace=g),
            best_line(ls, c, n - 1, a, "S", keep, max_dist, grace=g), pv)


def channels(d: pd.DataFrame, extend: int = 3) -> list[dict]:
    """The channel / wedge at each swing size, for the chart (duplicates across sizes dropped)."""
    n, a = len(d), float(d["atr"].iloc[-1])
    c = float(d["close"].iloc[-1])
    out, seen = [], []
    for name, k, keep, dist in SCALES:
        r, s, _ = _scale_lines(d, k, keep, dist)
        sh = shape(r, s, a, n - 1)
        edges = []
        for ln, side in ((r, "upper"), (s, "lower")):
            if ln is None:
                continue
            gap = (c - ln.at(n - 1)) if side == "upper" else (ln.at(n - 1) - c)
            if gap > 1.5 * a:                         # broken and left well behind: no longer in play
                continue
            sig = (round(ln.at(n - 1) / a, 1), round(ln.slope / a * 100, 1))
            if sig in seen:                          # the same line already drawn at a smaller size
                continue
            seen.append(sig)
            edges.append({"side": side, "touches": ln.touches,
                          "from": {"i": ln.from_i, "price": ln.at(ln.from_i)},
                          "to": {"i": n - 1 + extend, "price": ln.at(n - 1 + extend)},
                          "now": ln.at(n - 1),
                          "broken": (c > ln.at(n - 1) + 0.1 * a) if side == "upper" else (c < ln.at(n - 1) - 0.1 * a)})
        if len(edges) < 2:
            sh = None
        if edges:
            out.append({"scale": name, "swing": k, "shape": sh, "edges": edges,
                        "upper_now": r.at(n - 1) if r else None, "lower_now": s.at(n - 1) if s else None})
    return out


def fib_levels(lo: float, hi: float, up: bool) -> list[dict]:
    rng = hi - lo
    lv = [{"r": 0.0, "price": hi if up else lo}, {"r": 1.0, "price": lo if up else hi}]
    for f in FIB:
        lv.append({"r": f, "price": hi - f * rng if up else lo + f * rng})
    for e in EXT:                                     # targets beyond the swing's end
        lv.append({"r": -(e - 1), "price": lo + e * rng if up else hi - e * rng, "ext": e})
    return sorted(lv, key=lambda x: -x["price"])


def fib_now(d: pd.DataFrame, k: int = 8) -> dict | None:
    """Retracement of the last completed swing at size k, if price is still inside it."""
    n = len(d)
    a = float(d["atr"].iloc[-1])
    pv = [p for p in pivots(d, k, k) if p.confirmed <= n - 1]
    if len(pv) < 2 or not np.isfinite(a):
        return None
    c = float(d["close"].iloc[-1])
    # the move running since the last confirmed pivot, once it has pulled back for a couple of candles
    last = pv[-1]
    seg = d.iloc[last.i + 1:]
    if len(seg) >= 3:
        up = last.kind == "L"
        e = int(np.argmax(seg["high"].to_numpy()) if up else np.argmin(seg["low"].to_numpy()))
        ext = float(seg["high"].iloc[e] if up else seg["low"].iloc[e])
        lo, hi = (last.price, ext) if up else (ext, last.price)
        if e <= len(seg) - 3 and hi - lo >= MIN_LEG_ATR * a and lo <= c <= hi:
            depth = (hi - c) / (hi - lo) if up else (c - lo) / (hi - lo)
            return {"dir": "up" if up else "down", "from_i": last.i, "to_i": last.i + 1 + e, "low": lo, "high": hi,
                    "depth": depth, "in_golden": 0.5 <= depth <= 0.786, "levels": fib_levels(lo, hi, up),
                    "provisional": True}
    for j in range(len(pv) - 1, 0, -1):               # newest valid swing first
        p0, p1 = pv[j - 1], pv[j]
        up = p1.kind == "H"
        lo, hi = (p0.price, p1.price) if up else (p1.price, p0.price)
        if hi - lo < MIN_LEG_ATR * a:
            continue
        after = d.iloc[p1.i + 1:]
        if len(after) and ((after["high"].max() > hi) if up else (after["low"].min() < lo)):
            return None                               # the swing has extended: no retracement to measure yet
        if len(after) and ((after["close"].min() < lo) if up else (after["close"].max() > hi)):
            return None                               # fully retraced: the swing is void
        depth = (hi - c) / (hi - lo) if up else (c - lo) / (hi - lo)
        return {"dir": "up" if up else "down", "from_i": p0.i, "to_i": p1.i, "low": lo, "high": hi,
                "depth": depth, "in_golden": 0.5 <= depth <= 0.786, "levels": fib_levels(lo, hi, up)}
    return None


def rolling(d: pd.DataFrame, k: int = 8, keep: int = 8, dist: float = 6.0) -> dict:
    """Replay bar by bar: wedge / triangle breakouts and golden-zone entries as they would have appeared."""
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = d["atr"].to_numpy() if "atr" in d else _atr(d, 14).to_numpy()
    n = len(d)
    by_conf: dict[int, list[Pivot]] = {}
    for p in pivots(d, k, k):
        by_conf.setdefault(p.confirmed, []).append(p)
    wedge = np.zeros(n, int)
    golden = np.zeros(n, int)
    highs, lows, allp = [], [], []
    r = s = None
    sh = None
    leg = None            # (up, lo, hi) of the last completed swing, None once used / void
    for i in range(1, n):
        if i in by_conf:
            for p in by_conf[i]:
                (highs if p.kind == "H" else lows).append(p)
                allp.append(p)
            if np.isfinite(a[i]) and a[i] > 0:
                r = best_line(highs, c, i, a[i], "R", keep, dist)
                s = best_line(lows, c, i, a[i], "S", keep, dist)
                sh = shape(r, s, a[i], i)
                if len(allp) >= 2:
                    p0, p1 = allp[-2], allp[-1]
                    if p0.kind != p1.kind:
                        up = p1.kind == "H"
                        lo, hi = (p0.price, p1.price) if up else (p1.price, p0.price)
                        leg = (up, lo, hi) if hi - lo >= MIN_LEG_ATR * a[i] else None
                        # price may already have moved past since the pivot: check the bars in between
                        if leg:
                            seg_h, seg_l = h[p1.i + 1:i + 1], l[p1.i + 1:i + 1]
                            if (up and len(seg_h) and seg_h.max() > hi) or (not up and len(seg_l) and seg_l.min() < lo):
                                leg = None
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        # wedge / triangle breakout
        if sh in ("falling wedge", "rising wedge", "triangle"):
            tol = 0.1 * a[i]
            if r is not None and c[i] > r.at(i) + tol and c[i - 1] <= r.at(i - 1) + tol and sh != "rising wedge":
                wedge[i] = 1
                r = s = sh = None
            elif s is not None and c[i] < s.at(i) - tol and c[i - 1] >= s.at(i - 1) - tol and sh != "falling wedge":
                wedge[i] = -1
                r = s = sh = None
        # golden zone
        if leg:
            up, lo, hi = leg
            rng = hi - lo
            if up:
                if h[i] > hi or c[i] < lo:
                    leg = None
                elif l[i] <= hi - 0.5 * rng and l[i] >= hi - 0.786 * rng - 0.1 * a[i] and c[i] > o[i] and c[i] >= hi - 0.618 * rng:
                    golden[i] = 1
                    leg = None
            else:
                if l[i] < lo or c[i] > hi:
                    leg = None
                elif h[i] >= lo + 0.5 * rng and h[i] <= lo + 0.786 * rng + 0.1 * a[i] and c[i] < o[i] and c[i] <= lo + 0.618 * rng:
                    golden[i] = -1
                    leg = None
    return {"wedge breakout": wedge, "fib golden zone": golden}


def analyse(d: pd.DataFrame, p: Params, long_only: bool = False) -> dict:
    """Channels for the chart, the current Fibonacci retracement, and the two rules with their records."""
    rs = rolling(d)
    out = {"channels": [], "fib": None, "setups": []}
    try:
        out["channels"] = channels(d)
    except Exception as e:
        out["channels_error"] = f"{type(e).__name__}: {e}"
    try:
        out["fib"] = fib_now(d)
    except Exception as e:
        out["fib_error"] = f"{type(e).__name__}: {e}"
    trend = "up" if bool(d["uptrend"].iloc[-1]) else "down"
    for name in SETUPS:
        sig_arr = rs[name].copy()
        if long_only:
            sig_arr[sig_arr < 0] = 0
        trades = backtest(d, name, p, sig_override=sig_arr)
        watch = {"trend": trend}
        f = out["fib"]
        if name == "fib golden zone" and f:
            side = "BUY" if f["dir"] == "up" else "SELL"
            g50 = next(x["price"] for x in f["levels"] if x["r"] == 0.5)
            g618 = next(x["price"] for x in f["levels"] if x["r"] == 0.618)
            watch |= {"side": side, "trigger": g618,
                      "rule": f"a {'bullish' if side == 'BUY' else 'bearish'} candle out of the golden zone "
                              f"{min(g50, g618):,.2f}–{max(g50, g618):,.2f}",
                      "stop": g618 - p.stop_atr * float(d['atr'].iloc[-1]) * (1 if side == 'BUY' else -1),
                      "target": f["high"] if side == "BUY" else f["low"]}
        sig = _signal_now(name, sig_arr, d, p, trades, watch)
        out["setups"].append({"setup": name, "text": TEXT[name], "caveat": "",
                              "signal": asdict(sig), "stats": asdict(stats(trades, name)),
                              "equity": equity_curve(trades),
                              "trades": [asdict(t) for t in trades[-12:]][::-1],
                              "all_trades": [{"entry_time": t.entry_time, "r": t.r, "reason": t.reason} for t in trades]})
    return out
