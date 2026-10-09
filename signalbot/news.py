"""Economic-calendar awareness: when NOT to trade.

Source: ForexFactory's free weekly calendar feed (this week + next week).
High-impact releases — NFP, CPI, FOMC, and the like — move gold and the
indices violently in the first minutes, with slippage and stop-runs that no
rule survives. So each symbol gets a list of upcoming events and a set of
*blackout windows* (15 minutes before to 30 minutes after a High-impact
release of a relevant currency). The page shows the windows on the chart,
counts down to the next one, and stamps any live signal that falls inside
one with a warning.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("signalbot.news")

FEEDS = ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
         "https://nfs.faireconomy.media/ff_calendar_nextweek.json"]
CACHE_SECONDS = 3600
BEFORE_MIN, AFTER_MIN = 15, 30


def _cache(root: Path) -> Path:
    return root / "state" / "ff_calendar.json"


def fetch_raw(root: Path) -> list[dict]:
    p = _cache(root)
    if p.exists() and time.time() - p.stat().st_mtime < CACHE_SECONDS:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    import urllib.request
    events: list[dict] = []
    for url in FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "signalbot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                events += json.loads(r.read())
        except Exception as e:
            log.info("calendar %s: %s", url, e)   # next-week feed is often not published yet
    if events:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(events))
        return events
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def events_for(root: Path, currencies: list[str], now: pd.Timestamp | None = None,
               horizon_hours: float = 72, min_impact: str = "Medium") -> list[dict]:
    """Relevant events from (now − 2h) to (now + horizon), in IST, soonest first."""
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    rank = {"Low": 0, "Medium": 1, "High": 2}
    out = []
    for e in fetch_raw(root):
        if e.get("country") not in currencies or e.get("impact") not in rank:
            continue
        if rank[e["impact"]] < rank[min_impact]:
            continue
        try:
            t = pd.Timestamp(e["date"]).tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            continue
        if t < now - pd.Timedelta(hours=2) or t > now + pd.Timedelta(hours=horizon_hours):
            continue
        out.append({"time": str(t)[:16], "ts": int(t.tz_localize("UTC").timestamp()),
                    "title": e["title"], "country": e["country"], "impact": e["impact"],
                    "forecast": e.get("forecast", ""), "previous": e.get("previous", ""),
                    "minutes_away": round((t - now).total_seconds() / 60)})
    out.sort(key=lambda x: x["ts"])
    return out


def blackouts(events: list[dict]) -> list[dict]:
    """No-trade windows around High-impact events (merged when they overlap)."""
    wins = []
    for e in events:
        if e["impact"] != "High":
            continue
        start = e["ts"] - BEFORE_MIN * 60
        end = e["ts"] + AFTER_MIN * 60
        if wins and start <= wins[-1]["end"]:
            wins[-1]["end"] = max(wins[-1]["end"], end)
            wins[-1]["titles"].append(e["title"])
        else:
            wins.append({"start": start, "end": end, "titles": [e["title"]]})
    return wins


def status(root: Path, currencies: list[str], now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    now_ts = int(now.tz_localize("UTC").timestamp())
    ev = events_for(root, currencies, now)
    wins = blackouts(ev)
    active = next((w for w in wins if w["start"] <= now_ts <= w["end"]), None)
    upcoming = next((w for w in wins if w["start"] > now_ts), None)

    def fmt(ts):
        return pd.Timestamp(ts, unit="s").strftime("%H:%M")

    return {
        "events": ev[:12],
        "blackouts": wins,
        "in_blackout": active is not None,
        "blackout_text": (f"NEWS WINDOW until {fmt(active['end'])} — {', '.join(active['titles'])}. "
                          f"Don't open anything; spreads and slippage spike here." if active else ""),
        "next_text": (f"next no-trade window {fmt(upcoming['start'])}–{fmt(upcoming['end'])} "
                      f"({', '.join(upcoming['titles'])})" if upcoming else "no High-impact release in the next 3 days"),
        "window_rule": f"{BEFORE_MIN} min before to {AFTER_MIN} min after each High-impact release",
        "source": "ForexFactory calendar",
    }
