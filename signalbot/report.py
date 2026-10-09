"""Assemble everything the page / alerts need for one instrument + timeframe."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import lines as L
from . import news
from . import patterns as P
from .data import SYMBOLS, TIMEFRAMES, load, source_for
from .strategy import SETUP_TEXT, SETUPS, Params, backtest, current, enrich, equity_curve, stats
from .symbols import pf

#: how many bars the chart shows per timeframe
CHART_BARS = {"1d": 260, "1h": 7 * 60, "15m": 25 * 10, "5m": 75 * 4}


def _t(ts: pd.Timestamp, tf: str):
    if tf == "1d":
        return ts.strftime("%Y-%m-%d")
    # lightweight-charts wants epoch seconds and displays them as UTC;
    # labelling IST wall-clock time as UTC makes the axis read 09:15, 10:15…
    return int(ts.tz_localize("UTC").timestamp())


def _bars(d: pd.DataFrame, n: int, tf: str) -> list[dict]:
    tail = d.iloc[-n:]
    return [{"time": _t(ts, tf), "open": round(float(r.open), 6), "high": round(float(r.high), 6),
             "low": round(float(r.low), 6), "close": round(float(r.close), 6)}
            for ts, r in tail.iterrows()]


def _line(d: pd.DataFrame, col: str, n: int, tf: str) -> list[dict]:
    tail = d[col].iloc[-n:]
    return [{"time": _t(ts, tf), "value": round(float(v), 6)} for ts, v in tail.items() if v == v]


def _credits(root: Path) -> dict | None:
    try:
        from . import twelvedata
        u = twelvedata.usage(root)
        return {"used": u["used"], "limit": u["limit"]}
    except Exception:
        return None


#: rules whose trigger is a level price must close beyond
_LEVEL_RULES = ("breakout", "volume breakout", "trendline break", "bollinger squeeze", "supertrend flip", "london breakout")


def verdict(setups: list[dict], d: pd.DataFrame, cal: dict, tf: str) -> dict:
    """One plain answer for the chart: trade or no trade, and the levels that would change it."""
    label = TIMEFRAMES[tf]["label"].lower()
    close = float(d["close"].iloc[-1])
    fresh = [s for s in setups if s["signal"]["status"] == "signal"]
    running = [s for s in setups if s["signal"]["status"] == "open"]

    # pending triggers, nearest first, one per side+level
    triggers, seen = [], set()
    for s in setups:
        w = s["signal"].get("watch") or {}
        if w.get("trigger") is None or s["signal"]["status"] != "none":
            continue
        kind = ("above" if w["side"] == "BUY" else "below") if s["setup"] in _LEVEL_RULES else "ema"
        key = (w["side"], round(float(w["trigger"]) / max(float(d["atr"].iloc[-1]), 1e-9) * 10))
        if key in seen:
            continue
        seen.add(key)
        triggers.append({"setup": s["setup"], "side": w["side"], "price": float(w["trigger"]), "kind": kind,
                         "rule": w["rule"], "target": w.get("target"), "stop": w.get("stop"),
                         "dist": abs(float(w["trigger"]) - close)})
    triggers.sort(key=lambda t: t["dist"])

    # chop: how often price crossed the 20 EMA over the last 12 closed candles
    tail = d.iloc[-13:]
    side_of = (tail["close"] > tail["ema_pull"]).astype(int)
    crosses = int((side_of.diff().abs() == 1).sum())
    chop = crosses >= 4

    def fmt(x):
        return f"{pf(x)}"

    if cal.get("in_blackout"):
        return {"state": "news", "headline": "NO TRADE — news window",
                "detail": cal["blackout_text"], "triggers": triggers[:4], "chop": chop}
    if fresh:
        s = fresh[0]
        sig = s["signal"]
        more = f" (+{len(fresh) - 1} more rule{'s' if len(fresh) > 2 else ''})" if len(fresh) > 1 else ""
        return {"state": "signal", "side": sig["side"],
                "headline": f"{sig['side']} — {s['setup']} fired on the {label} close{more}",
                "detail": f"enter at the next candle ≈ {fmt(sig['entry'])} · target {fmt(sig['target'])} · stop {fmt(sig['stop'])}"
                          + (f" · choppy: price crossed the 20 EMA {crosses}× in the last 12 candles, expect whipsaws" if chop else ""),
                "triggers": [], "chop": chop}
    buy = next((t for t in triggers if t["side"] == "BUY" and t["kind"] == "above"), None)
    sell = next((t for t in triggers if t["side"] == "SELL" and t["kind"] == "below"), None)
    parts = []
    if buy:
        parts.append(f"BUY if a {label} candle closes above {fmt(buy['price'])} ({buy['setup']})")
    if sell:
        parts.append(f"SELL if one closes below {fmt(sell['price'])} ({sell['setup']})")
    ema = next((t for t in triggers if t["kind"] == "ema"), None)
    if ema:
        parts.append(f"{ema['side']} on a pullback through the 20 EMA ≈ {fmt(ema['price'])}")
    if not sell and buy:
        parts.append("nothing on the sell side while the trend is up")
    if not buy and sell:
        parts.append("nothing on the buy side while the trend is down")
    running_txt = ""
    if running:
        r = running[0]["signal"]
        running_txt = (f" A {r['side']} from the {running[0]['setup']} rule is still running "
                       f"(entered {fmt(r['entry'])}, stop {fmt(r['stop'])}, target {fmt(r['target'])}) — no new entry.")
    zone = ""
    if buy and sell:
        zone = f" No-trade zone: {fmt(sell['price'])} – {fmt(buy['price'])}."
    return {"state": "running" if running else "wait",
            "headline": "NO TRADE ZONE — no rule has fired" if not running else "NO NEW TRADE — a rule trade is running",
            "detail": " · ".join(parts) + "." + zone + running_txt
                      + (f" Choppy: price crossed the 20 EMA {crosses}× in the last 12 candles." if chop else ""),
            "zone": [sell["price"], buy["price"]] if buy and sell else None,
            "triggers": triggers[:4], "chop": chop}


SESSIONS = [(0, 7, "Asia"), (7, 12, "London"), (12, 16, "London / New York overlap"), (16, 21, "New York"), (21, 24, "after hours")]


def session_now(now_utc: pd.Timestamp | None = None) -> dict:
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    h = now_utc.hour
    name = next(n for a, b, n in SESSIONS if a <= h < b)
    note = {"Asia": "usually the quietest gold session — tighter ranges, more false breaks",
            "London": "volume picks up; the London open often sets the day's first real move",
            "London / New York overlap": "the most liquid hours for gold — biggest moves, best fills",
            "New York": "active until about 21:00 UTC, fading into the close",
            "after hours": "thin market around the daily roll — spreads widen"}[name]
    return {"name": name, "note": note}


def trends(symbol: str, root: Path) -> dict:
    """Close vs 200 EMA on every timeframe, from the cached candles (no extra API calls)."""
    from .indicators import ema
    out = {}
    for tf in ("1d", "1h", "15m", "5m"):
        if not (root / "data" / f"{symbol}_{tf}.csv").exists():
            out[tf] = None                       # not downloaded yet: don't spend credits just for the arrow
            continue
        try:
            c = load(symbol, tf, root, refresh=False)["close"]
            e = ema(c, 200)
            out[tf] = None if pd.isna(e.iloc[-1]) else ("up" if c.iloc[-1] > e.iloc[-1] else "down")
        except Exception:
            out[tf] = None
    return out


def _enrich_verdict(v: dict, setups: list[dict], symbol: str, root: Path, tf: str, d: pd.DataFrame) -> dict:
    """Add the multi-timeframe trend, confluence and session to the verdict."""
    tr = trends(symbol, root)
    v["trends"] = tr
    known = [x for x in tr.values() if x]
    v["aligned"] = (len(known) == 4 and len(set(known)) == 1)
    # confluence: rules pointing each way right now (new signals + running trades)
    buys = [s["setup"] for s in setups if s["signal"]["status"] in ("signal", "open") and s["signal"]["side"] == "BUY"]
    sells = [s["setup"] for s in setups if s["signal"]["status"] in ("signal", "open") and s["signal"]["side"] == "SELL"]
    v["confluence"] = {"buy": buys, "sell": sells, "of": len(setups)}
    v["session"] = session_now()
    # a signal against the higher timeframes deserves a warning
    higher = {"5m": ["15m", "1h"], "15m": ["1h", "1d"], "1h": ["1d"], "1d": []}[tf]
    if v.get("state") == "signal":
        want = "up" if v["side"] == "BUY" else "down"
        against = [h for h in higher if tr.get(h) and tr[h] != want]
        if against:
            v["warning"] = f"against the {' & '.join(h.upper() for h in against)} trend — lower odds"
    return v


def scoreboard(setups: list[dict], d: pd.DataFrame, recent_days: int = 30) -> list[dict]:
    """Every rule ranked by its average result per trade (after costs), with its last-30-days record."""
    cutoff = str((d.index[-1] - pd.Timedelta(days=recent_days)))[:16]
    rows = []
    for s in setups:
        st, sig = s["stats"], s["signal"]
        rec = [t for t in s.get("all_trades", s["trades"]) if t["reason"] != "open" and t["entry_time"] >= cutoff[:len(t["entry_time"])]]
        rr = [t["r"] for t in rec]
        rows.append({"setup": s["setup"], "status": sig["status"], "side": sig["side"],
                     "trades": st["trades"], "win_rate": st["win_rate"], "avg_r": st["avg_r"],
                     "profit_factor": st["profit_factor"], "max_dd_r": st["max_dd_r"], "total_r": st["total_r"],
                     "recent_n": len(rr), "recent_avg_r": (sum(rr) / len(rr)) if rr else None,
                     "recent_win": (sum(1 for r in rr if r > 0) / len(rr)) if rr else None})
    rows.sort(key=lambda r: (r["trades"] >= 30, r["avg_r"]), reverse=True)
    return rows


def _volume_series(d: pd.DataFrame, n: int, tf: str) -> dict:
    tail = d.iloc[-n:]
    out = {"vol": [], "tvol": []}
    if "vol" in tail:
        for ts, r in tail.iterrows():
            if r["vol"] == r["vol"]:
                hot = r.get("rvol") == r.get("rvol") and r.get("rvol", 0) >= 1.5
                up = r["close"] >= r["open"]
                out["vol"].append({"time": _t(ts, tf), "value": float(r["vol"]),
                                   "color": ("#0f9d58" if up else "#d93025") if hot else
                                            ("rgba(15,157,88,.35)" if up else "rgba(217,48,37,.35)")})
    if "tvol" in tail:
        out["tvol"] = [{"time": _t(ts, tf), "value": float(v)} for ts, v in tail["tvol"].items() if v == v]
    return out


def _volume_latest(d: pd.DataFrame) -> dict:
    try:
        from . import volume
        return volume.latest(d)
    except Exception:
        return {}


def _setup_block(name: str, text: str, trades, sig, caveat: str = "") -> dict:
    return {
        "setup": name, "text": text, "caveat": caveat,
        "signal": asdict(sig), "stats": asdict(stats(trades, name)),
        "equity": equity_curve(trades),
        "trades": [asdict(t) for t in trades[-12:]][::-1],
        "all_trades": [{"entry_time": t.entry_time, "r": t.r, "reason": t.reason} for t in trades],
    }


def _long_only(trades, sig):
    trades = [t for t in trades if t.side == "BUY"]
    if sig.side == "SELL":
        sig.side, sig.status = None, "none"
        sig.note = "SELL signal ignored (long-only)"
        sig.entry = sig.stop = sig.target = sig.trade = None
    return trades, sig


def build(symbol: str, root: Path, tf: str = "1d", *, params: Params | None = None,
          refresh: bool = True, long_only: bool = False, flat_by_close: bool | None = None,
          stop_atr: float | None = None, target_atr: float | None = None,
          max_bars: int | None = None, cost: float | None = None) -> dict:
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf}")
    intraday = tf != "1d"
    meta = SYMBOLS[symbol]
    # gold trades round the clock: positions carry, no session square-off
    p = params or Params(intraday=False, timed=intraday)
    if stop_atr:
        p.stop_atr = float(stop_atr)
    if target_atr:
        p.target_atr = float(target_atr)
    if max_bars:
        p.max_bars = int(max_bars)
    raw = load(symbol, tf, root, refresh=refresh)
    try:                                         # volume is a help, never a blocker
        from . import volume
        raw = volume.attach(raw, root, tf, refresh=refresh, symbol=symbol)
    except Exception as e:
        import logging
        logging.getLogger("signalbot.report").warning("volume unavailable: %s", e)
    from .symbols import default_cost
    p.cost = float(cost) if cost else default_cost(meta, float(raw["close"].iloc[-1]))   # 0 / None = market default
    d = enrich(raw, p)
    n_days = int(d.index.normalize().nunique())
    caveat = ""
    if intraday:
        caveat = f"{n_days} days of {TIMEFRAMES[tf]['label']} data."

    setups = []
    for s in SETUPS:
        trades = backtest(d, s, p)
        sig = current(d, s, p, trades)
        if long_only:
            trades, sig = _long_only(trades, sig)
        setups.append(_setup_block(s, SETUP_TEXT[s], trades, sig, caveat))

    # trendlines / channels and the two rules built on them
    try:
        ln = L.analyse(d, p, long_only=long_only)
        for blk in ln["setups"]:
            blk["caveat"] = caveat
        setups += ln["setups"]
        lines_out = {"lines": ln["lines"], "channel": ln["channel"], "events": ln["events"]}
    except Exception as e:
        lines_out = {"lines": [], "channel": False, "events": [], "error": f"{type(e).__name__}: {e}"}

    # classical patterns: what formed, what is forming, and each one's record here
    try:
        pats = P.analyse(raw)
    except Exception as e:
        pats = {"forming": None, "recent_candles": [], "candle_stats": [], "chart_stats": [],
                "pivots": [], "error": f"{type(e).__name__}: {e}"}
    for key in ("pivots",):
        for pv in pats.get(key, []):
            pv["t"] = _t(d.index[pv["i"]], tf)
    if pats.get("forming"):
        for pv in pats["forming"]["pivots"]:
            pv["t"] = _t(d.index[pv["i"]], tf)
    for lnd in lines_out["lines"]:
        lnd["from"]["t"] = _t(d.index[lnd["from"]["i"]], tf)
        # the line is extended past the last bar; give the UI the bar step so it can place it
    step = None
    if intraday and len(d) > 1:
        step = int((d.index[-1] - d.index[-2]).total_seconds())
    for ev in lines_out["events"]:
        ev["t"] = _t(d.index[ev["i"]], tf)

    # news: upcoming releases that matter for this instrument, and the no-trade windows
    try:
        cal = news.status(root, meta["news"])
    except Exception as e:                       # the calendar is a convenience, never a blocker
        cal = {"events": [], "blackouts": [], "in_blackout": False, "blackout_text": "",
               "next_text": f"calendar unavailable ({type(e).__name__})", "window_rule": "", "source": ""}
    if cal["in_blackout"]:
        for blk in setups:
            sig = blk["signal"]
            if sig["status"] in ("signal", "open"):
                sig["news_warning"] = cal["blackout_text"]

    _sb = scoreboard(setups, d)
    for blk in setups:
        blk.pop("all_trades", None)                  # only needed for the scoreboard
    last = d.iloc[-1]
    n = CHART_BARS[tf]
    return {
        "symbol": symbol, "name": meta["name"], "td": meta["td"], "short": meta.get("short", meta["td"]),
        "kind": meta.get("kind", "metal"),
        "source": source_for(root, symbol),
        "credits": _credits(root),
        "session": meta["session"], "unit": meta.get("unit", ""),
        "news": cal,
        "patterns": pats, "trendlines": lines_out, "bar_step": step,
        "tf": tf, "tf_label": TIMEFRAMES[tf]["label"], "intraday": intraday,
        "flat_by_close": bool(p.intraday), "last_entry": p.last_entry if p.intraday else None,
        "as_of": str(d.index[-1])[:16 if intraday else 10],
        "last_close": round(float(last["close"]), 6),
        "trend": "up" if bool(last["uptrend"]) else "down",
        "ema_trend": round(float(last["ema_trend"]), 6),
        "atr": round(float(last["atr"]), 6),
        "history": {"from": str(d.index[0])[:10], "bars": int(len(d)), "days": n_days},
        "params": asdict(p), "long_only": long_only,
        "setups": setups,
        "verdict": _enrich_verdict(verdict(setups, d, cal, tf), setups, symbol, root, tf, d),
        "scoreboard": _sb,
        "chart": {"bars": _bars(d, n, tf), "ema_trend": _line(d, "ema_trend", n, tf),
                  "ema_pull": _line(d, "ema_pull", n, tf), **_volume_series(d, n, tf)},
        "volume": _volume_latest(d),
    }


def headline(rep: dict) -> list[str]:
    """One line per actionable signal — what the Telegram alert and the CLI print."""
    lines = []
    for s in rep["setups"]:
        sig, st = s["signal"], s["stats"]
        if sig["status"] not in ("signal", "open"):
            continue
        side_n = st["buys"] if sig["side"] == "BUY" else st["sells"]
        side_w = st["buy_win_rate"] if sig["side"] == "BUY" else st["sell_win_rate"]
        warn = f" ⚠ {sig['news_warning']}" if sig.get("news_warning") else ""
        lines.append(
            f"{rep['name']} · {rep['tf_label']} · {s['setup']} · {sig['side']} "
            f"{'(live)' if sig['status'] == 'open' else '(new)'}: "
            f"from {pf(sig['entry'])} → target {pf(sig['target'])}, stop {pf(sig['stop'])}. "
            f"Record: {st['trades']} trades, {st['win_rate']:.0%} won, avg {st['avg_r']:+.2f}R; "
            f"this side {side_n} trades, {side_w:.0%} won.{warn}"
        )
    return lines
