"""Liquidity and market-structure ("smart money") concepts, as fixed rules the backtest can replay.

Everything here is computed bar by bar with only what was visible at the time:
swing points are confirmed `RIGHT` bars after they print, the previous day's
high/low only exist once that day has closed, and a gap or order block can only
be traded after it has formed.

  liquidity sweep    a wick through the most recent confirmed swing high (low) that closes back
                     below (above) it — stops above the high got run, price rejected → SELL (BUY)
  pdh/pdl sweep      the same at the previous day's high / low (first sweep of each per day)
  asian range sweep  during London (07–10 UTC) a wick beyond the Asian range (00–07 UTC) that
                     closes back inside — the "Judas swing" → trade back into the range
  fvg retest         in an uptrend, price dips into an unfilled bullish fair value gap
                     (low[i] > high[i-2]) and closes back above its bottom on a green candle → BUY
  order block        a displacement candle (body ≥ 1.5 ATR) marks the last opposite candle before it
                     as an order block; the first retest that holds, with the trend → trade
  choch              change of character: after lower highs, the first close above the last swing
                     high → BUY (mirror → SELL)

They go through the same engine as every other rule (next-bar entry, 2 ATR stop,
3 ATR target, spread) so the scoreboard says whether any of it works on this market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEFT = RIGHT = 3          # swing point = highest high / lowest low of 7 bars, confirmed 3 bars later
EQ_ATR = 0.15             # two swing highs within this many ATRs = equal highs (a liquidity pool)
IST = pd.Timedelta(hours=5, minutes=30)

TEXT = {
    "liquidity sweep": "Liquidity sweep — price wicks through the latest swing high and closes back below it "
                       "(the stops above got taken, buyers failed) → SELL; mirror at a swing low → BUY.",
    "pdh/pdl sweep": "Previous-day sweep — a wick above yesterday's high that closes back below → SELL; "
                     "below yesterday's low and back above → BUY. First sweep of each per day.",
    "asian range sweep": "Asian range sweep (Judas swing) — between 12:30 and 15:30 IST, London wicks beyond the Asian "
                         "session's range and closes back inside → trade back into the range.",
    "fvg retest": "Fair value gap retest — in an uptrend, price comes back into an unfilled bullish gap "
                  "(a candle whose low is above the high two candles earlier) and closes back above the gap's "
                  "bottom on a green candle → BUY. Mirror in a downtrend.",
    "order block": "Order block — a big displacement candle (body ≥ 1.5 ATR) makes the last opposite candle "
                   "before it an order block; the first retest that holds, in the trend's direction, is the entry.",
    "choch": "Change of character — after a run of lower highs, the first close above the last swing high "
             "→ BUY (structure turned up); mirror → SELL.",
}
SETUPS = tuple(TEXT)
#: rules whose watch level is a price to be swept (wick through, close back) rather than closed beyond
SWEEP_RULES = ("liquidity sweep", "pdh/pdl sweep", "asian range sweep")


# ── building blocks ─────────────────────────────────────────────────────────

def swings(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Arrays the length of d: the latest swing high / low price *known* at each bar (NaN before the first)."""
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    n = len(d)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(n):
        p = i - RIGHT                       # the bar that becomes a swing point once bar i closes
        if p >= LEFT:
            win_h, win_l = h[p - LEFT:p + RIGHT + 1], l[p - LEFT:p + RIGHT + 1]
            if h[p] == win_h.max() and np.argmax(win_h) == LEFT:
                last_h = h[p]
            if l[p] == win_l.min() and np.argmin(win_l) == LEFT:
                last_l = l[p]
        sh[i], sl[i] = last_h, last_l
    return sh, sl


def prev_day(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Previous UTC day's high / low for every bar (intraday)."""
    day = pd.Series((d.index - IST).normalize(), index=d.index)
    dh = d["high"].groupby(day).max()
    dl = d["low"].groupby(day).min()
    return day.map(dh.shift(1)), day.map(dl.shift(1))


def add_indicators(d: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    d["swing_hi"], d["swing_lo"] = swings(d)
    if intraday:
        d["pdh"], d["pdl"] = prev_day(d)
    return d


# ── signals ─────────────────────────────────────────────────────────────────

def _first_per_day(mask: pd.Series, d: pd.DataFrame) -> pd.Series:
    day = pd.Series((d.index - IST).normalize(), index=d.index)
    return mask & (mask.astype(int).groupby(day).cumsum() == 1)


def signals(d: pd.DataFrame, setup: str) -> tuple[pd.Series, pd.Series]:
    no = pd.Series(False, index=d.index)
    h, l, c, o = d["high"], d["low"], d["close"], d["open"]
    if setup == "liquidity sweep":
        sh, sl = d["swing_hi"].shift(1).to_numpy(), d["swing_lo"].shift(1).to_numpy()   # known before this bar
        hh, ll, cc = h.to_numpy(), l.to_numpy(), c.to_numpy()
        n = len(d)
        buy, sell = np.zeros(n, bool), np.zeros(n, bool)
        used_h = used_l = np.nan                       # each swing level can be swept-and-traded once
        for i in range(n):
            if np.isfinite(sh[i]) and sh[i] != used_h and hh[i] > sh[i] and cc[i] < sh[i]:
                sell[i], used_h = True, sh[i]
            if np.isfinite(sl[i]) and sl[i] != used_l and ll[i] < sl[i] and cc[i] > sl[i]:
                buy[i], used_l = True, sl[i]
            if np.isfinite(sh[i]) and cc[i] > sh[i]:
                used_h = sh[i]                         # closed through it: no longer a sweep candidate
            if np.isfinite(sl[i]) and cc[i] < sl[i]:
                used_l = sl[i]
        return pd.Series(buy, index=d.index), pd.Series(sell, index=d.index)
    if setup == "pdh/pdl sweep":
        if "pdh" not in d:
            return no, no
        sell = _first_per_day(((h > d["pdh"]) & (c < d["pdh"])).fillna(False), d)
        buy = _first_per_day(((l < d["pdl"]) & (c > d["pdl"])).fillna(False), d)
        return buy, sell
    if setup == "asian range sweep":
        if "asia_hi" not in d:
            return no, no
        lon = d["london"]
        sell = _first_per_day((lon & (h > d["asia_hi"]) & (c < d["asia_hi"])).fillna(False), d)
        buy = _first_per_day((lon & (l < d["asia_lo"]) & (c > d["asia_lo"])).fillna(False), d)
        return buy, sell
    if setup == "fvg retest":
        return _fvg_signals(d)
    if setup == "order block":
        return _ob_signals(d)
    if setup == "choch":
        return _choch_signals(d)
    raise ValueError(setup)


def _fvg_signals(d: pd.DataFrame, keep: int = 5, max_age: int = 120):
    h, l, c, o = (d[k].to_numpy() for k in ("high", "low", "close", "open"))
    a = d["atr"].to_numpy()
    up, dn = d["uptrend"].to_numpy(), d["downtrend"].to_numpy()
    n = len(d)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    bulls: list[list] = []                # [bottom, top, born]
    bears: list[list] = []
    for i in range(2, n):
        # trade existing gaps first (they formed on earlier bars)
        for g in list(bulls):
            if i - g[2] > max_age or c[i] < g[0]:
                bulls.remove(g)
            elif l[i] <= g[1] and c[i] > g[0] and c[i] > o[i] and up[i]:
                buy[i] = True
                bulls.remove(g)
                break
        for g in list(bears):
            if i - g[2] > max_age or c[i] > g[1]:
                bears.remove(g)
            elif h[i] >= g[0] and c[i] < g[1] and c[i] < o[i] and dn[i]:
                sell[i] = True
                bears.remove(g)
                break
        if not np.isfinite(a[i]):
            continue
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= 0.1 * a[i]:
            bulls.append([h[i - 2], l[i], i])
            bulls = bulls[-keep:]
        if h[i] < l[i - 2] and l[i - 2] - h[i] >= 0.1 * a[i]:
            bears.append([h[i], l[i - 2], i])
            bears = bears[-keep:]
    return pd.Series(buy, index=d.index), pd.Series(sell, index=d.index)


def _ob_signals(d: pd.DataFrame, max_age: int = 60):
    h, l, c, o = (d[k].to_numpy() for k in ("high", "low", "close", "open"))
    a = d["atr"].to_numpy()
    up, dn = d["uptrend"].to_numpy(), d["downtrend"].to_numpy()
    n = len(d)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    bull_ob = bear_ob = None              # [low, high, born]
    for i in range(4, n):
        if bull_ob:
            if i - bull_ob[2] > max_age or c[i] < bull_ob[0]:
                bull_ob = None
            elif i > bull_ob[2] + 1 and l[i] <= bull_ob[1] and c[i] > bull_ob[0] and c[i] > o[i] and up[i]:
                buy[i], bull_ob = True, None
        if bear_ob:
            if i - bear_ob[2] > max_age or c[i] > bear_ob[1]:
                bear_ob = None
            elif i > bear_ob[2] + 1 and h[i] >= bear_ob[0] and c[i] < bear_ob[1] and c[i] < o[i] and dn[i]:
                sell[i], bear_ob = True, None
        if not np.isfinite(a[i]):
            continue
        body = c[i] - o[i]
        if body >= 1.5 * a[i]:            # bullish displacement → last red candle before it
            for k in range(i - 1, max(i - 4, 0), -1):
                if c[k] < o[k]:
                    bull_ob = [l[k], h[k], i]
                    break
        elif -body >= 1.5 * a[i]:
            for k in range(i - 1, max(i - 4, 0), -1):
                if c[k] > o[k]:
                    bear_ob = [l[k], h[k], i]
                    break
    return pd.Series(buy, index=d.index), pd.Series(sell, index=d.index)


def structure(d: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Market structure per bar (+1 up / -1 down / 0 unknown) and its break events (BOS / CHoCH)."""
    c = d["close"].to_numpy()
    sh, sl = d["swing_hi"].to_numpy(), d["swing_lo"].to_numpy()
    n = len(d)
    state = np.zeros(n, int)
    events: list[dict] = []
    cur, broken_h, broken_l = 0, np.nan, np.nan
    for i in range(1, n):
        ph, pl = sh[i - 1], sl[i - 1]
        if np.isfinite(ph) and c[i] > ph and ph != broken_h:
            events.append({"i": i, "kind": "CHoCH" if cur == -1 else "BOS", "side": "up", "price": float(ph)})
            cur, broken_h = 1, ph
        elif np.isfinite(pl) and c[i] < pl and pl != broken_l:
            events.append({"i": i, "kind": "CHoCH" if cur == 1 else "BOS", "side": "down", "price": float(pl)})
            cur, broken_l = -1, pl
        state[i] = cur
    return state, events


def _choch_signals(d: pd.DataFrame):
    _, ev = structure(d)
    buy, sell = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for e in ev:
        if e["kind"] == "CHoCH":
            (buy if e["side"] == "up" else sell)[e["i"]] = True
    return pd.Series(buy, index=d.index), pd.Series(sell, index=d.index)


# ── what would trigger next ─────────────────────────────────────────────────

def watch(setup: str, d: pd.DataFrame, p) -> dict:
    from .symbols import pf
    last = d.iloc[-1]
    a = float(last["atr"])
    c = float(last["close"])

    def lv(side, x, rule, kind="sweep"):
        x = float(x)
        return {"side": side, "trigger": x, "kind": kind, "rule": rule,
                "stop": x + p.stop_atr * a if side == "SELL" else x - p.stop_atr * a,
                "target": x - p.target_atr * a if side == "SELL" else x + p.target_atr * a}

    if setup == "liquidity sweep":
        sh, sl = last.get("swing_hi"), last.get("swing_lo")
        opts = []
        if pd.notna(sh) and sh > c:                     # a sell-side sweep needs the high above price
            opts.append(lv("SELL", sh, f"a wick above the swing high {pf(sh)} that closes back below it"))
        if pd.notna(sl) and sl < c:
            opts.append(lv("BUY", sl, f"a wick below the swing low {pf(sl)} that closes back above it"))
        return min(opts, key=lambda w: abs(w["trigger"] - c)) if opts else {"rule": "no swing points yet"}
    if setup == "pdh/pdl sweep":
        if pd.isna(last.get("pdh", np.nan)):
            return {"rule": "intraday timeframes only"}
        opts = [w for w in (lv("SELL", last["pdh"], f"a wick above yesterday's high {pf(last['pdh'])} that closes back below"),
                            lv("BUY", last["pdl"], f"a wick below yesterday's low {pf(last['pdl'])} that closes back above"))
                if (w["side"] == "SELL") == (w["trigger"] > c)]
        return min(opts, key=lambda w: abs(w["trigger"] - c)) if opts else {"rule": "price is outside yesterday's range — no sweep setup"}
    if setup == "asian range sweep":
        if pd.isna(last.get("asia_hi", np.nan)):
            return {"rule": "the Asian range (05:30–12:30 IST) is still forming, or not an intraday timeframe"}
        opts = [w for w in (lv("SELL", last["asia_hi"], f"London wicking above the Asian high {pf(last['asia_hi'])} and closing back inside (12:30–15:30 IST)"),
                            lv("BUY", last["asia_lo"], f"London wicking below the Asian low {pf(last['asia_lo'])} and closing back inside (12:30–15:30 IST)"))
                if (w["side"] == "SELL") == (w["trigger"] > c)]
        return min(opts, key=lambda w: abs(w["trigger"] - c)) if opts else {"rule": "price is outside the Asian range — no sweep setup"}
    lv_ = levels(d)
    if setup == "fvg retest":
        want = "bull" if bool(last["uptrend"]) else "bear"
        gaps = [g for g in lv_["fvgs"] if g["kind"] == want]
        if not gaps:
            return {"rule": f"an unfilled {'bullish' if want == 'bull' else 'bearish'} gap first — none open now"}
        g = gaps[-1]
        side = "BUY" if want == "bull" else "SELL"
        return lv(side, g["top"] if side == "BUY" else g["bottom"],
                  f"price dipping into the gap {pf(g['bottom'])}–{pf(g['top'])} and closing back out of it", kind="zone")
    if setup == "order block":
        obs = [z for z in lv_["obs"] if (z["kind"] == "bull") == bool(last["uptrend"])]
        if not obs:
            return {"rule": "a displacement candle first — no active order block in the trend's direction"}
        z = obs[-1]
        side = "BUY" if z["kind"] == "bull" else "SELL"
        return lv(side, z["top"] if side == "BUY" else z["bottom"],
                  f"a retest of the order block {pf(z['bottom'])}–{pf(z['top'])} that holds", kind="zone")
    if setup == "choch":
        st = lv_["structure"]
        if st == "down" and pd.notna(last.get("swing_hi")):
            return lv("BUY", last["swing_hi"], f"a close above the last swing high {pf(last['swing_hi'])} (structure turns up)", kind="above")
        if st == "up" and pd.notna(last.get("swing_lo")):
            return lv("SELL", last["swing_lo"], f"a close below the last swing low {pf(last['swing_lo'])} (structure turns down)", kind="below")
        return {"rule": "structure not established yet"}
    return {}


# ── levels for the chart, the card and the banner ───────────────────────────

def levels(d: pd.DataFrame, lookback: int = 300) -> dict:
    """Liquidity pools, open gaps / order blocks, structure and recent sweeps — as of the last closed bar."""
    n = len(d)
    last = d.iloc[-1]
    c = float(last["close"])
    a = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    tail = d.iloc[max(0, n - lookback):]
    off = max(0, n - lookback)
    h, l, cl, o = (tail[k].to_numpy() for k in ("high", "low", "close", "open"))

    # swing points in the window (confirmed only)
    piv_h, piv_l = [], []
    for p in range(LEFT, len(tail) - RIGHT):
        if h[p] == h[p - LEFT:p + RIGHT + 1].max():
            piv_h.append((p, h[p]))
        if l[p] == l[p - LEFT:p + RIGHT + 1].min():
            piv_l.append((p, l[p]))

    def unswept(points, above):
        out = []
        for p, px in points:
            later = h[p + 1:] if above else l[p + 1:]
            if (later > px).any() if above else (later < px).any():
                continue
            out.append((p, px))
        return out

    eq = []
    for pts, kind, above in ((piv_h, "EQH", True), (piv_l, "EQL", False)):
        live = unswept(pts, above)
        for j in range(len(live)):
            for k in range(j + 1, len(live)):
                if a and abs(live[j][1] - live[k][1]) <= EQ_ATR * a:
                    eq.append({"kind": kind, "price": float((live[j][1] + live[k][1]) / 2),
                               "from_i": off + live[j][0]})
    # dedupe pools that are within EQ_ATR of each other
    pools: list[dict] = []
    for e in sorted(eq, key=lambda e: e["price"]):
        if pools and pools[-1]["kind"] == e["kind"] and abs(pools[-1]["price"] - e["price"]) <= EQ_ATR * a:
            continue
        pools.append(e)

    # unfilled fair value gaps (last 4) and active order blocks (last 2) in the window
    fvgs = []
    for i in range(2, len(tail)):
        if l[i] > h[i - 2] and (not a or l[i] - h[i - 2] >= 0.1 * a):
            bot, top = h[i - 2], l[i]
            if not (l[i + 1:] <= bot).any():
                fvgs.append({"kind": "bull", "bottom": float(bot), "top": float(top), "from_i": off + i - 2,
                             "filled": bool((l[i + 1:] <= top).any())})
        if h[i] < l[i - 2] and (not a or l[i - 2] - h[i] >= 0.1 * a):
            bot, top = h[i], l[i - 2]
            if not (h[i + 1:] >= top).any():
                fvgs.append({"kind": "bear", "bottom": float(bot), "top": float(top), "from_i": off + i - 2,
                             "filled": bool((h[i + 1:] >= bot).any())})
    fvgs = [g for g in fvgs if not g["filled"]][-4:]
    obs = []
    for i in range(4, len(tail)):
        body = cl[i] - o[i]
        if a and abs(body) >= 1.5 * a:
            for k in range(i - 1, max(i - 4, 0), -1):
                if (body > 0 and cl[k] < o[k]) or (body < 0 and cl[k] > o[k]):
                    lo, hi = l[k], h[k]
                    broken = (cl[i + 1:] < lo).any() if body > 0 else (cl[i + 1:] > hi).any()
                    if not broken:
                        obs.append({"kind": "bull" if body > 0 else "bear", "bottom": float(lo), "top": float(hi),
                                    "from_i": off + k})
                    break
    obs = obs[-2:]

    state, events = structure(d)
    st = {1: "up", -1: "down"}.get(int(state[-1]), "unclear")
    last_break = events[-1] if events else None

    # recent sweeps (last 5) for markers
    sb, ss = signals(d, "liquidity sweep")
    sweeps = [{"i": int(i), "side": "BUY" if sb.iloc[i] else "SELL",
               "price": float(d["low"].iloc[i] if sb.iloc[i] else d["high"].iloc[i])}
              for i in np.flatnonzero((sb | ss).to_numpy())[-5:]]

    pool_levels = []
    for key, name, side in (("pdh", "PDH", 0), ("pdl", "PDL", 0), ("asia_hi", "Asia high", 0), ("asia_lo", "Asia low", 0),
                            ("swing_hi", "swing high", 1), ("swing_lo", "swing low", -1)):
        v = last.get(key, np.nan)
        if pd.notna(v) and (side == 0 or (side == 1 and v > c) or (side == -1 and v < c)):
            pool_levels.append({"name": name, "price": float(v)})
    for e in pools:
        pool_levels.append({"name": e["kind"], "price": e["price"]})
    above = sorted([x for x in pool_levels if x["price"] > c], key=lambda x: x["price"])[:3]
    below = sorted([x for x in pool_levels if x["price"] < c], key=lambda x: -x["price"])[:3]
    return {"structure": st, "last_break": last_break, "pools": pools[-6:], "fvgs": fvgs, "obs": obs,
            "sweeps": sweeps, "above": above, "below": below,
            "pdh": float(last["pdh"]) if pd.notna(last.get("pdh", np.nan)) else None,
            "pdl": float(last["pdl"]) if pd.notna(last.get("pdl", np.nan)) else None}
