"""Twelve Data as the gold feed: live spot XAU/USD with a free API key.

Sign up at twelvedata.com (free, open to India), copy the API key from the
dashboard and put it in .env:

    TWELVEDATA_API_KEY=...

The free plan is metered in credits (one per request): about 8 a minute and
800 a day. To stay inside that while running all day, this module

  * backfills history once per timeframe (a few dozen credits, 8 s apart);
  * afterwards builds the 5m / 15m / 1h candles from a shared 1-minute cache,
    fed by the live tick stream (stream.py) while the page is open and topped
    up with official 1-minute bars every 10 minutes (every REFRESH_SECONDS
    when the stream is not running);
  * tops up the daily candles once an hour;
  * skips pulls while the gold market is closed for the weekend;
  * keeps a credit ledger in state/ and serves cached candles instead of
    going over the daily allowance.

With the page open all day that is roughly 170 credits; without the stream about 600.
"""
from __future__ import annotations

import json
import re
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("signalbot.twelvedata")

BASE = "https://api.twelvedata.com"
SYMBOL = "XAU/USD"
INTERVAL = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1day"}
RESAMPLE = {"5m": "5min", "15m": "15min", "1h": "1h"}
BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}
MAX_OUT = 5000
#: first-run history: pages of 5,000 bars (5m ≈ 170 days, 15m ≈ 1.1 years, 1h ≈ 3 years, 1d ≈ 20 years)
BACKFILL_PAGES = {"1d": 2, "1h": 5, "15m": 8, "5m": 10}
#: markets added from the page start lighter, so they open in seconds (≈ 5m: 15,000 candles, 1h: 10,000);
#: their caches then grow every day signalbot runs
NEW_MARKET_PAGES = {"1d": 1, "1h": 2, "15m": 2, "5m": 3}
PER_MINUTE = 8
MINUTE_KEEP = 12000             # 1-minute bars kept on disk (~9 trading days)

_lock = threading.RLock()
_ledger_lock = threading.Lock()
IST = "Asia/Kolkata"


class LimitReached(RuntimeError):
    pass


# ── settings ────────────────────────────────────────────────────────────────

def _env(root: Path) -> dict:
    from .notify import load_env
    env = load_env(root)
    env.update({k: v for k, v in os.environ.items() if k.startswith("TWELVEDATA_")})
    return env


def credentials(root: Path) -> str | None:
    return _env(root).get("TWELVEDATA_API_KEY") or None


def available(root: Path) -> bool:
    return credentials(root) is not None


def daily_limit(root: Path) -> int:
    return int(_env(root).get("TWELVEDATA_DAILY_LIMIT") or 800)


def refresh_seconds(root: Path) -> int:
    return int(_env(root).get("TWELVEDATA_REFRESH_SECONDS") or 150)


# ── credit ledger ───────────────────────────────────────────────────────────

def _ledger_path(root: Path) -> Path:
    return root / "state" / "twelvedata_usage.json"


def usage(root: Path) -> dict:
    try:
        led = json.loads(_ledger_path(root).read_text())
    except Exception:
        led = {}
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if led.get("day") != today:
        led = {"day": today, "used": 0, "recent": []}
    led["limit"] = daily_limit(root)
    return led


def _spend(root: Path) -> None:
    """Count one credit; wait if the per-minute allowance is used up; refuse past the daily one."""
    with _ledger_lock:
        led = usage(root)
        if led["used"] >= led["limit"]:
            raise LimitReached(f"Twelve Data daily allowance used ({led['used']}/{led['limit']} credits, "
                               f"resets 00:00 UTC = 05:30 IST) — showing cached candles")
        now = time.time()
        recent = [t for t in led.get("recent", []) if now - t < 60]
        if len(recent) >= PER_MINUTE:
            wait = 60.5 - (now - min(recent))
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            recent = [t for t in recent if now - t < 60]
        recent.append(now)
        led["recent"], led["used"] = recent, led["used"] + 1
        p = _ledger_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: led[k] for k in ("day", "used", "recent")}))


# ── raw API ─────────────────────────────────────────────────────────────────

def _get(root: Path, path: str, spend: bool = True, **params) -> dict:
    key = credentials(root)
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY missing in .env — free key from twelvedata.com (README, step 1)")
    if spend:
        _spend(root)
    url = f"{BASE}{path}?" + urllib.parse.urlencode({**params, "apikey": key})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "signalbot/1.0"}),
                                    timeout=20) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            raise RuntimeError(f"Twelve Data HTTP {e.code}") from None
    if isinstance(data, dict) and data.get("status") == "error":
        code, msg = data.get("code"), data.get("message", "")
        if code in (401, 403):
            raise RuntimeError(f"Twelve Data rejected the API key ({code}): {msg[:160]} — check "
                               f"TWELVEDATA_API_KEY in .env, then run: python run.py td-test") from None
        if code == 429:
            raise LimitReached(f"Twelve Data rate limit: {msg[:160]}")
        if code == 400 and "no data" in msg.lower():
            return {"values": []}
        raise RuntimeError(f"Twelve Data error {code}: {msg[:160]}")
    return data


def _frame(values: list[dict]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="time"))
    df = pd.DataFrame(values)
    df["time"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
    for k in ("open", "high", "low", "close"):
        df[k] = df[k].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) if "volume" in df else 0.0
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    return df[~df.index.duplicated(keep="last")].sort_index()


def time_series(root: Path, interval: str, outputsize: int = MAX_OUT,
                end: pd.Timestamp | None = None, symbol: str = SYMBOL) -> pd.DataFrame:
    """The latest `outputsize` bars (up to `end`, IST-naive, when given)."""
    params = {"symbol": symbol, "interval": INTERVAL[interval], "outputsize": int(outputsize),
              "timezone": "UTC"}
    if end is not None:
        params["end_date"] = end.tz_localize(IST).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    return _frame(_get(root, "/time_series", **params).get("values", []))


# ── market hours ────────────────────────────────────────────────────────────

def market_closed(now: pd.Timestamp | None = None, kind: str = "metal") -> bool:
    """Is this kind of market shut right now?  crypto: never · stocks: outside 09:30–16:00 New York on
    weekdays · gold, forex, other metals: the weekend, roughly Fri 21–22:00 → Sun 22–23:00 UTC."""
    now = now or pd.Timestamp.now(tz="UTC")
    if kind == "crypto":
        return False
    if kind == "stock":
        ny = now.tz_convert("America/New_York")
        mins = ny.hour * 60 + ny.minute
        return ny.weekday() >= 5 or not (9 * 60 + 30 <= mins < 16 * 60)
    wd, h = now.weekday(), now.hour
    return wd == 5 or (wd == 4 and h >= 22) or (wd == 6 and h < 21)


def market_hours_only(df: pd.DataFrame, interval: str, kind: str = "metal") -> pd.DataFrame:
    """Drop candles from when gold is not trading.

    Twelve Data keeps publishing XAU/USD candles over the weekend and in the
    daily maintenance hour — synthetic, with the open frozen and ~$0.25 of
    jitter. Left in, they made up ~30% of the 5-minute history and the rules
    fired on that noise. Gold trades Sun 18:00 → Fri 17:00 New York time with a
    17:00–18:00 break each day; everything else is removed (DST handled by the
    New York time zone). Daily candles dated Saturday or Sunday (UTC) go too."""
    if df is None or df.empty or kind == "crypto":         # crypto trades every hour of every day
        return df
    utc = df.index - pd.Timedelta(hours=5, minutes=30)
    if interval == "1d" or kind == "stock":                 # stocks: Twelve Data sends session bars only
        ny_wd = utc.tz_localize("UTC").tz_convert("America/New_York").weekday
        return df[(utc.weekday < 5) if interval == "1d" else (ny_wd < 5)]
    ny = utc.tz_localize("UTC").tz_convert("America/New_York")
    wd, h = ny.weekday, ny.hour
    closed = ((wd == 4) & (h >= 17)) | (wd == 5) | ((wd == 6) & (h < 18)) | (h == 17)
    if interval == "1h":                     # an hourly candle that starts 16:30 NY runs into the break
        end = (ny + pd.Timedelta(minutes=59))
        closed |= (end.hour == 17) | ((end.weekday == 4) & (end.hour >= 17))
    return df[~closed]


# ── fetching ────────────────────────────────────────────────────────────────

def backfill(root: Path, interval: str, symbol: str = SYMBOL) -> pd.DataFrame:
    """Years of history on first use, paging backwards 5,000 bars at a time."""
    frames, end = [], None
    light = os.environ.get("SIGNALBOT_LIGHT_BACKFILL") == "1"        # set by streamlit_app.py: fast cold starts
    pages = BACKFILL_PAGES if (symbol == SYMBOL and not light) else NEW_MARKET_PAGES
    for _ in range(pages[interval]):
        try:
            df = time_series(root, interval, MAX_OUT, end=end, symbol=symbol)
        except LimitReached as e:
            if "minute" not in str(e).lower() or frames:
                if frames:
                    break                                # keep what we have; the cache grows later
                raise
            log.info("per-minute limit hit during backfill — waiting a minute")
            time.sleep(61)                               # the key is shared (e.g. Mac + cloud): wait and retry once
            df = time_series(root, interval, MAX_OUT, end=end, symbol=symbol)
        if df.empty:
            break
        frames.append(df)
        if len(df) < MAX_OUT:
            break
        end = df.index[0] - pd.Timedelta(seconds=1)
    if not frames:
        raise RuntimeError(f"Twelve Data returned no {interval} candles for {symbol}")
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _key(symbol: str) -> str:
    from .symbols import key_of
    return key_of(symbol)


def _minute_path(root: Path, symbol: str = SYMBOL) -> Path:
    return root / "data" / f"{_key(symbol)}_1m.csv"


def _stream_live(symbol: str = SYMBOL) -> bool:
    try:
        from . import stream
        st = stream.get()
        return bool(st and st.live(_key(symbol)))
    except Exception:
        return False


def _read_minutes(root: Path, symbol: str = SYMBOL) -> pd.DataFrame | None:
    p = _minute_path(root, symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col="time", parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def append_minutes(root: Path, rows: pd.DataFrame, symbol: str = SYMBOL) -> None:
    """Add 1-minute bars built from the live stream (newer minutes only)."""
    with _lock:
        cached = _read_minutes(root, symbol)
        if cached is not None:
            rows = rows.copy()
            for t in rows.index:                    # a minute already cached (partial, or the REST copy):
                if t in cached.index:               # keep its open, widen high/low, take the newer close
                    rows.loc[t, "open"] = cached.loc[t, "open"]
                    rows.loc[t, "high"] = max(rows.loc[t, "high"], cached.loc[t, "high"])
                    rows.loc[t, "low"] = min(rows.loc[t, "low"], cached.loc[t, "low"])
                    rows.loc[t, "volume"] = rows.loc[t, "volume"] + float(cached.loc[t, "volume"] or 0)
            df = pd.concat([cached[cached.index < rows.index.min()], rows])
        else:
            df = rows
        p = _minute_path(root, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.iloc[-MINUTE_KEEP:].to_csv(p)


def minute_bars(root: Path, force: bool = False, symbol: str = SYMBOL, kind: str = "metal") -> pd.DataFrame:
    """The newest 1-minute bars, shared through a file cache by the page, the
    CLI, the watcher and the live stream.

    While the live stream is running it appends each minute as it completes,
    so the official REST bars are only pulled every 10 minutes (to replace
    the tick-built ones); without the stream, every REFRESH_SECONDS."""
    cached = _read_minutes(root, symbol)
    stamp = root / "data" / f".{_key(symbol)}_1m.rest"
    age = time.time() - stamp.stat().st_mtime if stamp.exists() else 1e9
    every = 600 if _stream_live(symbol) else refresh_seconds(root)
    if cached is not None and not force and (age < every or market_closed(kind=kind)):
        return cached
    now = pd.Timestamp.now(tz=IST).tz_localize(None)
    n = 1500 if cached is None else int((now - cached.index[-1]).total_seconds() // 60) + 20
    if age < 1e9 and cached is not None:            # re-pull the stretch since the last official pull
        n = max(n, int(age // 60) + 5)
    try:
        new = time_series(root, "1m", max(10, min(MAX_OUT, n)), symbol=symbol)
    except Exception as e:
        if cached is not None:
            log.warning("1-minute pull failed (%s) — using cached bars", e)
            return cached
        raise
    with _lock:
        cached = _read_minutes(root, symbol)        # the stream may have appended meanwhile
        if cached is None or new.empty:
            df = new if cached is None else cached
        else:                                       # official bars replace tick-built ones; keep newer stream bars
            new = new.copy()                        # …and keep the tick counts where REST has no volume (spot fx/metals)
            if kind != "stock":
                new["volume"] = cached["volume"].reindex(new.index).fillna(0).to_numpy()
            df = pd.concat([cached[cached.index < new.index.min()], new, cached[cached.index > new.index.max()]])
        df = df.iloc[-MINUTE_KEEP:]
        p = _minute_path(root, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p)
        stamp.touch()
    return df


def resample(m1: pd.DataFrame, interval: str, origin: pd.Timestamp | None = None) -> pd.DataFrame:
    """1-minute bars → 5m / 15m / 1h candles on the same grid as the cached candles (`origin`,
    IST-naive: e.g. hourly stock candles start at :30 New York time, gold's on the UTC hour)."""
    if m1.empty:
        return m1
    utc = m1.copy()
    utc.index = utc.index.tz_localize(IST).tz_convert("UTC")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in utc:
        agg["volume"] = "sum"
    kw = {"origin": origin.tz_localize(IST).tz_convert("UTC")} if origin is not None else {}
    out = utc.resample(RESAMPLE[interval], label="left", closed="left", **kw).agg(agg).dropna(subset=["open"])
    out.index = out.index.tz_convert(IST).tz_localize(None)
    out.index.name = "time"
    return out


def fresh_minutes(root: Path, symbol: str = SYMBOL, kind: str = "metal") -> pd.DataFrame:
    """The 1-minute cache, made sure to hold the minute that just closed."""
    m1 = minute_bars(root, symbol=symbol, kind=kind)
    # no live stream on this server (e.g. Streamlit Cloud) and the cache is missing the minutes
    # that just closed: pull them now, so a candle is never judged on half its data (1 credit)
    now_min = pd.Timestamp.now(tz=IST).tz_localize(None).floor("1min")
    if (not m1.empty and m1.index[-1] < now_min - pd.Timedelta(minutes=1)
            and not market_closed(kind=kind) and not _stream_live(symbol)):
        m1 = minute_bars(root, force=True, symbol=symbol, kind=kind)
    return m1


def minute_history(root: Path, symbol: str = SYMBOL, kind: str = "metal", want: int = 8000,
                   refresh: bool = True) -> pd.DataFrame:
    """1-minute candles for the 1 min chart: the live cache, deepened once a day to ~`want` bars
    by paging further back (1 credit per 5,000 bars)."""
    m1 = fresh_minutes(root, symbol, kind) if refresh else (_read_minutes(root, symbol)
                                                            if _read_minutes(root, symbol) is not None
                                                            else minute_bars(root, symbol=symbol, kind=kind))
    stamp = root / "data" / f".{_key(symbol)}_1m.deep"
    tried = stamp.exists() and time.time() - stamp.stat().st_mtime < 86400
    if refresh and len(m1) < want * 0.8 and not tried:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
        frames, end = [], m1.index[0] - pd.Timedelta(seconds=1)
        for _ in range(max(1, -(-(want - len(m1)) // MAX_OUT))):
            try:
                old = time_series(root, "1m", MAX_OUT, end=end, symbol=symbol)
            except Exception as e:
                log.info("deeper 1-minute history unavailable: %s", e)
                break
            if old.empty:
                break
            frames.append(old)
            end = old.index[0] - pd.Timedelta(seconds=1)
        if frames:
            with _lock:
                cur = _read_minutes(root, symbol)
                cur = m1 if cur is None else cur
                old = pd.concat(frames)
                df = pd.concat([old[old.index < cur.index[0]], cur])
                df = df[~df.index.duplicated(keep="last")].sort_index().iloc[-max(MINUTE_KEEP, want):]
                df.to_csv(_minute_path(root, symbol))
                m1 = df
    return m1


def resample_tf(df: pd.DataFrame, tf: str, kind: str = "metal") -> pd.DataFrame:
    """30 min from 15 min, 4 hour from 1 hour, weekly from daily candles."""
    if df is None or df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    agg |= {c: "sum" for c in ("volume",) if c in df}
    if tf == "1w":                                   # weeks start Monday, labelled by that Monday
        out = df.resample("W-MON", label="left", closed="left").agg(agg).dropna(subset=["open"])
        out.index.name = "time"
        return out
    rule = {"30m": "30min", "4h": "4h"}[tf]
    utc = df.copy()
    utc.index = utc.index.tz_localize(IST).tz_convert("UTC")
    # on the UTC grid (00:00, 04:00 … like TradingView); 4-hour candles keep the minute their hourly
    # candles start on (US stocks: :30)
    off = pd.Timedelta(minutes=int(utc.index[0].minute)) if tf == "4h" else pd.Timedelta(0)
    out = utc.resample(rule, label="left", closed="left", origin="epoch", offset=off).agg(agg).dropna(subset=["open"])
    out.index = out.index.tz_convert(IST).tz_localize(None)
    out.index.name = "time"
    return out


def update(root: Path, interval: str, cached: pd.DataFrame | None, symbol: str = SYMBOL,
           kind: str = "metal") -> pd.DataFrame:
    """Full candle history for `interval`: backfill once, then cheap top-ups."""
    if cached is None or len(cached) < 300:
        return backfill(root, interval, symbol)
    last = cached.index[-1]
    if interval == "1d":
        n = (pd.Timestamp.now(tz=IST).tz_localize(None) - last).days + 3
        new = time_series(root, "1d", max(5, min(MAX_OUT, n)), symbol=symbol)
    else:
        m1 = fresh_minutes(root, symbol, kind)
        if not m1.empty and m1.index[0] <= last:
            new = resample(m1[m1.index >= last], interval, origin=last)
        else:                                   # gap longer than the 1-minute cache: ask for the bars directly
            n = int((pd.Timestamp.now(tz=IST).tz_localize(None) - last).total_seconds()
                    // 60 // BAR_MINUTES[interval]) + 5
            new = time_series(root, interval, max(10, min(MAX_OUT, n)), symbol=symbol)
    if new.empty:
        return cached
    return pd.concat([cached[cached.index < new.index.min()], new])


FREE_TYPES = ("Precious Metal", "Physical Currency", "Digital Currency", "Common Stock", "ETF")
US_EXCHANGES = ("NASDAQ", "NYSE", "NYSE ARCA", "NYSE AMERICAN", "BATS", "CBOE")


def search(root: Path, query: str, limit: int = 12) -> list[dict]:
    """Twelve Data symbol search, cleaned up: types the free plan streams, US listings for stocks,
    exact matches first. Costs 1 credit."""
    q = query.strip()
    if not q:
        return []
    data = _get(root, "/symbol_search", symbol=q, outputsize=60).get("data", [])
    out, seen = [], set()
    for r in data:
        t, ex = r.get("instrument_type", ""), (r.get("exchange") or "").upper()
        if t not in FREE_TYPES:
            continue
        if t in ("Common Stock", "ETF") and ex not in US_EXCHANGES:
            continue
        sym = r.get("symbol", "")
        if sym in seen:
            continue
        seen.add(sym)
        out.append({"td": sym, "name": r.get("instrument_name", sym), "type": t, "exchange": r.get("exchange", "")})
    qq = q.upper().replace(" ", "")
    base = qq.split("/")[0]
    out.sort(key=lambda r: (r["td"].replace("/", "") != qq.replace("/", ""),     # exact match first
                            r["td"] != f"{base}/USD",                           # then the USD pair
                            not r["td"].startswith(base + "/") and r["td"] != base,
                            r["type"] not in ("Precious Metal", "Physical Currency", "Digital Currency")))
    # crypto pairs often aren't in the search index at all: offer the USD pair directly
    crypto = ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB", "LTC", "AVAX", "DOT", "LINK", "MATIC", "TRX")
    metals = ("XAU", "XAG", "XPT", "XPD")
    if qq in crypto + metals and not any(r["td"] == f"{qq}/USD" for r in out):
        out.insert(0, {"td": f"{qq}/USD", "name": f"{qq} / US Dollar",
                       "type": "Digital Currency" if qq in crypto else "Precious Metal", "exchange": ""})
    return out[:limit]


def test_key(root: Path) -> dict:
    """Is the key accepted, how many credits are left today, and the latest price."""
    if not credentials(root):
        return {"ok": False, "message": "TWELVEDATA_API_KEY missing in .env"}
    try:
        df = time_series(root, "1m", 1)
    except Exception as e:
        return {"ok": False, "message": str(e)}
    led = usage(root)
    if df.empty:
        return {"ok": False, "message": "key accepted, but no XAU/USD candles came back"}
    return {"ok": True, "message": f"Key accepted. XAU/USD {df['close'].iloc[-1]:,.2f} at "
                                   f"{str(df.index[-1])[:16]} IST. Credits used today: "
                                   f"{led['used']}/{led['limit']}."}
