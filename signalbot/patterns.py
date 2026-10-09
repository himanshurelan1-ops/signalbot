"""Classical chart and candlestick patterns — detected mechanically, and judged
by their own record on *this* instrument and timeframe.

Pattern lore is mostly folklore: "a hammer is bullish" is true on some
markets and false on others. So every pattern here is detected by a fixed
rule, and then the same detector is run across the whole history to count
what actually happened next — did price travel the textbook way by 1.5 ATR
before going 1.5 ATR the other way? That ratio is shown beside each pattern.
Above ~55% with a decent sample means something; 50% means the pattern is
noise here, however famous it is.

Swing pivots (fractals) are the raw material for chart patterns: a pivot
high is a bar whose high is the highest of `left` bars before and `right`
bars after it. It is only *confirmed* `right` bars later — detection here
respects that, so nothing is ever "seen" before a trader could have seen it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr as _atr, ema
from .symbols import pf

# ── pivots ──────────────────────────────────────────────────────────────────

@dataclass
class Pivot:
    i: int          # bar index
    price: float
    kind: str       # "H" or "L"
    confirmed: int  # bar index at which it became visible


def pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out: list[Pivot] = []
    for i in range(left, n - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] >= win_h.max() and np.argmax(win_h) == left:
            out.append(Pivot(i, float(h[i]), "H", i + right))
        elif l[i] <= win_l.min() and np.argmin(win_l) == left:
            out.append(Pivot(i, float(l[i]), "L", i + right))
    # alternate H / L: of two consecutive highs keep the higher, of two lows the lower
    alt: list[Pivot] = []
    for p in out:
        if alt and alt[-1].kind == p.kind:
            keep_new = (p.price > alt[-1].price) if p.kind == "H" else (p.price < alt[-1].price)
            if keep_new:
                alt[-1] = p
            continue
        alt.append(p)
    return alt


# ── forward outcome: did it work? ───────────────────────────────────────────

def outcome(df_np: dict, i: int, direction: int, atr_i: float,
            move_atr: float = 1.5, max_bars: int = 20) -> float | None:
    """From the open after bar i, +1 if price travels `move_atr` ATRs in
    `direction` before travelling that far against it, -1 if the reverse,
    0 if neither within max_bars, None if there is no future data."""
    o, h, l = df_np["open"], df_np["high"], df_np["low"]
    n = len(o)
    e = i + 1
    if e >= n or not np.isfinite(atr_i) or atr_i <= 0:
        return None
    entry = o[e]
    up, dn = entry + move_atr * atr_i, entry - move_atr * atr_i
    for j in range(e, min(n, e + max_bars)):
        hit_up, hit_dn = h[j] >= up, l[j] <= dn
        if hit_up and hit_dn:
            return -1.0                                   # ambiguous bar → count against
        if hit_up:
            return 1.0 if direction > 0 else -1.0
        if hit_dn:
            return 1.0 if direction < 0 else -1.0
    return 0.0


@dataclass
class PatternStats:
    name: str
    n: int = 0
    worked: int = 0
    failed: int = 0
    undecided: int = 0
    work_rate: float | None = None   # worked / (worked + failed); None = no textbook direction

    def add(self, r: float | None) -> None:
        if r is None:
            return
        self.n += 1
        if r > 0:
            self.worked += 1
        elif r < 0:
            self.failed += 1
        else:
            self.undecided += 1
        dec = self.worked + self.failed
        self.work_rate = self.worked / dec if dec else None


# ── candlestick patterns ────────────────────────────────────────────────────

CANDLE_TEXT = {
    "bullish engulfing": "a green body that swallows the previous red body, after a dip — buyers took over",
    "bearish engulfing": "a red body that swallows the previous green body, after a rise — sellers took over",
    "hammer": "long lower wick (≥2× body), small upper wick, after a dip — a rejected sell-off",
    "shooting star": "long upper wick (≥2× body), small lower wick, after a rise — a rejected rally",
    "morning star": "red bar, small-bodied bar, then a green bar closing into the first — a 3-bar bottom",
    "evening star": "green bar, small-bodied bar, then a red bar closing into the first — a 3-bar top",
    "doji": "open ≈ close — indecision; no direction by itself",
    "inside bar": "range inside the previous bar's range — compression; trade the break of the mother bar",
}
CANDLE_DIR = {"bullish engulfing": 1, "bearish engulfing": -1, "hammer": 1, "shooting star": -1,
              "morning star": 1, "evening star": -1, "doji": 0, "inside bar": 0}


def candlestick_events(df: pd.DataFrame) -> list[tuple[int, str]]:
    """(bar index, pattern name) for every candlestick pattern in the data."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    e20 = ema(df["close"], 20).to_numpy()
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    out = []
    for i in range(2, len(df)):
        green, red = c[i] > o[i], c[i] < o[i]
        pgreen, pred = c[i - 1] > o[i - 1], c[i - 1] < o[i - 1]
        below, above = c[i] < e20[i], c[i] > e20[i]
        if body[i] <= 0.1 * rng[i]:
            out.append((i, "doji"))
        if h[i] < h[i - 1] and l[i] > l[i - 1]:
            out.append((i, "inside bar"))
        if green and pred and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] > body[i - 1] and below:
            out.append((i, "bullish engulfing"))
        if red and pgreen and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] > body[i - 1] and above:
            out.append((i, "bearish engulfing"))
        if body[i] > 0 and lower[i] >= 2 * body[i] and upper[i] <= body[i] and below:
            out.append((i, "hammer"))
        if body[i] > 0 and upper[i] >= 2 * body[i] and lower[i] <= body[i] and above:
            out.append((i, "shooting star"))
        small_mid = body[i - 1] <= 0.3 * rng[i - 1]
        if (c[i - 2] < o[i - 2] and small_mid and green
                and c[i] > (o[i - 2] + c[i - 2]) / 2 and c[i - 2] < e20[i - 2]):
            out.append((i, "morning star"))
        if (c[i - 2] > o[i - 2] and small_mid and red
                and c[i] < (o[i - 2] + c[i - 2]) / 2 and c[i - 2] > e20[i - 2]):
            out.append((i, "evening star"))
    return out


# ── chart patterns from pivots ──────────────────────────────────────────────

CHART_TEXT = {
    "double top": "two highs at the same level with a trough between — breaks down through the trough (neckline)",
    "double bottom": "two lows at the same level with a peak between — breaks up through the peak (neckline)",
    "head and shoulders": "three highs, the middle one highest — breaks down through the neckline joining the two troughs",
    "inverse head and shoulders": "three lows, the middle one lowest — breaks up through the neckline joining the two peaks",
    "ascending triangle": "flat highs, rising lows — pressure builds under a ceiling; textbook break is up",
    "descending triangle": "flat lows, falling highs — pressure builds on a floor; textbook break is down",
    "symmetrical triangle": "falling highs and rising lows converge — a coil; direction comes from the break",
    "rising wedge": "both lines rise but converge — momentum fading; textbook break is down",
    "falling wedge": "both lines fall but converge — selling fading; textbook break is up",
    "ascending channel": "parallel rising lines — trend intact; buy the lower line, sell the upper",
    "descending channel": "parallel falling lines — trend intact; sell the upper line, buy the lower",
    "rectangle": "flat parallel lines — a range; trade the edges or the eventual break",
}
#: textbook direction: +1 bullish, -1 bearish, 0 = whichever way it breaks
CHART_DIR = {"double top": -1, "double bottom": 1, "head and shoulders": -1,
             "inverse head and shoulders": 1, "ascending triangle": 1, "descending triangle": -1,
             "symmetrical triangle": 0, "rising wedge": -1, "falling wedge": 1,
             "ascending channel": 1, "descending channel": -1, "rectangle": 0}


@dataclass
class ChartPattern:
    name: str
    direction: int
    pivots: list[dict]              # the pivots that form it, for drawing
    level: float | None = None      # neckline / break level (textbook direction)
    upper: dict | None = None       # {"slope", "intercept"} in (bar index → price), for lines
    lower: dict | None = None
    note: str = ""


def _fit(points: list[Pivot]) -> tuple[float, float, float]:
    """Least-squares line through pivots: slope, intercept, max residual."""
    xs = np.array([p.i for p in points], dtype=float)
    ys = np.array([p.price for p in points])
    if len(xs) < 2:
        return 0.0, float(ys.mean()) if len(ys) else 0.0, 0.0
    slope, icpt = np.polyfit(xs, ys, 1)
    resid = np.abs(ys - (slope * xs + icpt)).max()
    return float(slope), float(icpt), float(resid)


def _two_line_pattern(recent: list[Pivot], highs: list[Pivot], lows: list[Pivot],
                      atr_now: float, now_i: int, tol: float, pd_) -> ChartPattern | None:
    if len(highs) < 2 or len(lows) < 2:
        return None
    sh, ih, rh = _fit(highs)
    sl, il, rl = _fit(lows)
    if rh > tol or rl > tol:
        return None
    span = max(recent[-1].i - recent[0].i, 1)
    # slopes as ATRs per 100 bars, so "flat" means the same thing on every timeframe
    nh, nl = sh * 100 / atr_now, sl * 100 / atr_now
    flat = 1.0
    up_now = sh * now_i + ih
    lo_now = sl * now_i + il
    if up_now <= lo_now:
        return None                                       # lines already crossed — stale
    lines = dict(upper={"slope": sh, "intercept": ih}, lower={"slope": sl, "intercept": il})
    width_note = f"upper ≈ {pf(up_now)}, lower ≈ {pf(lo_now)}"
    converging = (sh - sl) < 0
    if abs(nh) <= flat and abs(nl) <= flat:
        return ChartPattern("rectangle", 0, pd_(recent), None, note=width_note, **lines)
    if abs(nh) <= flat and nl > flat:
        return ChartPattern("ascending triangle", 1, pd_(recent), float(up_now), note=width_note, **lines)
    if abs(nl) <= flat and nh < -flat:
        return ChartPattern("descending triangle", -1, pd_(recent), float(lo_now), note=width_note, **lines)
    if nh < -flat and nl > flat:
        return ChartPattern("symmetrical triangle", 0, pd_(recent), None, note=width_note, **lines)
    if nh > flat and nl > flat:
        if converging:
            return ChartPattern("rising wedge", -1, pd_(recent), float(lo_now), note=width_note, **lines)
        return ChartPattern("ascending channel", 1, pd_(recent), None, note=width_note, **lines)
    if nh < -flat and nl < -flat:
        if (sh - sl) > 0:                                 # lower line falling faster → converging
            return ChartPattern("falling wedge", 1, pd_(recent), float(up_now), note=width_note, **lines)
        return ChartPattern("descending channel", -1, pd_(recent), None, note=width_note, **lines)
    return None




def detect_chart_pattern(pivs: list[Pivot], atr_now: float, now_i: int) -> ChartPattern | None:
    """Look at the most recent pivots and name the structure they form, if any."""
    if len(pivs) < 4 or not np.isfinite(atr_now) or atr_now <= 0:
        return None
    tol = 0.6 * atr_now
    recent = pivs[-6:]
    highs = [p for p in recent if p.kind == "H"]
    lows = [p for p in recent if p.kind == "L"]
    pd_ = lambda ps: [{"i": p.i, "price": p.price, "kind": p.kind} for p in ps]

    # ── two converging / parallel lines need 5+ pivots; they outrank the 3-pivot shapes ──
    if len(recent) >= 5:
        tl = _two_line_pattern(recent, highs, lows, atr_now, now_i, tol, pd_)
        if tl is not None:
            return tl

    # ── reversal shapes on the last 5 pivots ──
    last5 = pivs[-5:]
    if len(last5) == 5:
        k = "".join(p.kind for p in last5)
        if k == "HLHLH":
            h1, l1, h2, l2, h3 = last5
            if h2.price > h1.price + tol and h2.price > h3.price + tol and abs(h1.price - h3.price) <= 2 * tol:
                neck_slope = (l2.price - l1.price) / max(l2.i - l1.i, 1)
                neck_now = l2.price + neck_slope * (now_i - l2.i)
                return ChartPattern("head and shoulders", -1, pd_(last5), float(neck_now),
                                    note=f"neckline ≈ {pf(neck_now)}")
        if k == "LHLHL":
            l1, h1, l2, h2, l3 = last5
            if l2.price < l1.price - tol and l2.price < l3.price - tol and abs(l1.price - l3.price) <= 2 * tol:
                neck_slope = (h2.price - h1.price) / max(h2.i - h1.i, 1)
                neck_now = h2.price + neck_slope * (now_i - h2.i)
                return ChartPattern("inverse head and shoulders", 1, pd_(last5), float(neck_now),
                                    note=f"neckline ≈ {pf(neck_now)}")
    last3 = pivs[-3:]
    k3 = "".join(p.kind for p in last3)
    if k3 == "HLH" and abs(last3[0].price - last3[2].price) <= tol and \
            min(last3[0].price, last3[2].price) - last3[1].price > 2 * tol:
        return ChartPattern("double top", -1, pd_(last3), last3[1].price,
                            note=f"neckline {pf(last3[1].price)}")
    if k3 == "LHL" and abs(last3[0].price - last3[2].price) <= tol and \
            last3[1].price - max(last3[0].price, last3[2].price) > 2 * tol:
        return ChartPattern("double bottom", 1, pd_(last3), last3[1].price,
                            note=f"neckline {pf(last3[1].price)}")

    return None


# ── putting it together ─────────────────────────────────────────────────────

def analyse(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    """Everything the page needs: what formed on the last bars, what is forming
    now, and each pattern's record on this exact data."""
    d = {k: df[k].to_numpy() for k in ("open", "high", "low", "close")}
    a = _atr(df, 14).to_numpy()
    n = len(df)

    # candlesticks: record + what's on the last 3 bars
    cstats = {name: PatternStats(name) for name in CANDLE_TEXT}
    events = candlestick_events(df)
    for i, name in events:
        direction = CANDLE_DIR[name]
        if direction == 0:
            # neutral patterns: "worked" = resolved by a move either way within 10 bars (just informational)
            continue
        cstats[name].add(outcome(d, i, direction, a[i]))
    recent_candles = [{"bar": n - 1 - i, "time": str(df.index[i])[:16], "name": name,
                       "direction": CANDLE_DIR[name], "text": CANDLE_TEXT[name],
                       "stats": asdict(cstats[name]) if CANDLE_DIR[name] else None}
                      for i, name in events if i >= n - 3]

    # chart patterns: run the detector at every pivot confirmation, score the textbook direction
    pivs = pivots(df, left, right)
    pstats = {name: PatternStats(name) for name in CHART_TEXT}
    seen_at: list[tuple[int, str]] = []
    for k in range(4, len(pivs) + 1):
        upto = pivs[:k]
        now_i = upto[-1].confirmed
        if now_i >= n:
            break
        pat = detect_chart_pattern(upto, a[now_i], now_i)
        if pat is None:
            continue
        seen_at.append((now_i, pat.name))
        if pat.direction != 0:
            pstats[pat.name].add(outcome(d, now_i, pat.direction, a[now_i]))
        else:
            pstats[pat.name].n += 1          # no textbook direction → counted, not scored

    # what is forming right now (only pivots already confirmed by the last bar)
    visible = [p for p in pivs if p.confirmed <= n - 1]
    forming = detect_chart_pattern(visible, a[-1], n - 1) if visible else None
    forming_d = None
    if forming:
        forming_d = asdict(forming)
        forming_d["text"] = CHART_TEXT[forming.name]
        forming_d["stats"] = asdict(pstats[forming.name])
        for p in forming_d["pivots"]:
            p["time"] = str(df.index[p["i"]])[:16]

    return {
        "pivots": [{"i": p.i, "time": str(df.index[p.i])[:16], "price": p.price, "kind": p.kind}
                   for p in visible[-12:]],
        "forming": forming_d,
        "recent_candles": recent_candles,
        "candle_stats": sorted([asdict(s) for s in cstats.values() if s.n],
                               key=lambda s: -(s["work_rate"] or 0)),
        "chart_stats": sorted([asdict(s) for s in pstats.values() if s.n],
                              key=lambda s: -(s["work_rate"] or 0)),
        "chart_texts": CHART_TEXT, "candle_texts": CANDLE_TEXT,
        "rule": "worked = price travelled 1.5 ATR the textbook way before 1.5 ATR against, within 20 bars",
    }
