import json
import time

import pandas as pd
import pytest

from signalbot import data, twelvedata as td


@pytest.fixture
def root(tmp_path):
    (tmp_path / ".env").write_text("TWELVEDATA_API_KEY=abc\n")
    return tmp_path


def _values(start_utc: str, n: int, minutes: int, newest_first=True):
    t0 = pd.Timestamp(start_utc)
    vals = [{"datetime": (t0 + pd.Timedelta(minutes=minutes * k)).strftime("%Y-%m-%d %H:%M:%S"),
             "open": "4100.0", "high": str(4101.0 + k % 3), "low": "4099.0", "close": str(4100.0 + k)}
            for k in range(n)]
    return vals[::-1] if newest_first else vals           # the API sends newest first


def test_xauusd_is_the_only_symbol_and_comes_from_twelvedata(tmp_path):
    assert list(data.available_symbols(tmp_path)) == ["XAUUSD"]
    assert data.source_for(tmp_path, "XAUUSD") == "twelvedata"


def test_frame_parses_utc_into_ist_ascending():
    df = td._frame(_values("2026-10-09 06:00:00", 3, 1))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index[0]) == "2026-10-09 11:30:00" and df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 4102.0


def test_backfill_pages_backwards_until_a_short_page(root, monkeypatch):
    calls = []

    def fake_get(r, path, spend=True, **p):
        calls.append(p)
        n = 5000 if len(calls) == 1 else 7
        start = "2026-01-01 00:00:00" if len(calls) == 1 else "2025-12-01 00:00:00"
        return {"values": _values(start, n, 5)}

    monkeypatch.setattr(td, "_get", fake_get)
    df = td.backfill(root, "5m")
    assert len(calls) == 2 and "end_date" not in calls[0] and "end_date" in calls[1]
    assert len(df) == 5007 and df.index.is_monotonic_increasing
    assert calls[0]["symbol"] == "XAU/USD" and calls[0]["interval"] == "5min" and calls[0]["timezone"] == "UTC"


def test_resample_builds_5m_and_1h_on_the_utc_grid():
    m1 = td._frame(_values("2026-10-09 06:00:00", 120, 1))      # 11:30–13:29 IST
    m5 = td.resample(m1, "5m")
    assert len(m5) == 24 and str(m5.index[0]) == "2026-10-09 11:30:00"
    first = m1.iloc[:5]
    assert m5["open"].iloc[0] == first["open"].iloc[0] and m5["close"].iloc[0] == first["close"].iloc[-1]
    assert m5["high"].iloc[0] == first["high"].max()
    h1 = td.resample(m1, "1h")
    assert [str(t)[11:16] for t in h1.index] == ["11:30", "12:30"]   # 06:00 / 07:00 UTC


def test_intraday_topup_uses_minute_cache_not_extra_requests(root, monkeypatch):
    cached = td._frame(_values("2026-10-08 00:00:00", 400, 5))       # ends 2026-10-09 09:15 UTC
    m1 = td._frame(_values("2026-10-09 09:00:00", 60, 1))            # overlaps the last cached bar
    monkeypatch.setattr(td, "minute_bars", lambda r, **k: m1)
    monkeypatch.setattr(td, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no API call expected")))
    out = td.update(root, "5m", cached)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert out.index[-1] == m1.index[-1].floor("5min")
    assert len(out) > len(cached)


def test_minute_cache_is_shared_and_respects_refresh(root, monkeypatch):
    n = {"calls": 0}

    def fake_get(r, path, spend=True, **p):
        n["calls"] += 1
        return {"values": _values("2026-10-09 06:00:00", 30, 1)}

    monkeypatch.setattr(td, "_get", fake_get)
    monkeypatch.setattr(td, "market_closed", lambda now=None: False)
    a = td.minute_bars(root)
    b = td.minute_bars(root)                                          # within REFRESH_SECONDS → file cache
    assert n["calls"] == 1 and len(a) == len(b) == 30
    assert (root / "data" / "XAUUSD_1m.csv").exists()


def test_daily_allowance_is_enforced(root):
    (root / ".env").write_text("TWELVEDATA_API_KEY=abc\nTWELVEDATA_DAILY_LIMIT=2\n")
    td._spend(root)
    td._spend(root)
    with pytest.raises(td.LimitReached):
        td._spend(root)
    led = json.loads((root / "state" / "twelvedata_usage.json").read_text())
    assert led["used"] == 2 and led["day"] == time.strftime("%Y-%m-%d", time.gmtime())


def test_weekend_is_closed():
    assert td.market_closed(pd.Timestamp("2026-10-10 12:00", tz="UTC"))        # Saturday
    assert td.market_closed(pd.Timestamp("2026-10-11 10:00", tz="UTC"))        # Sunday morning
    assert not td.market_closed(pd.Timestamp("2026-10-12 10:00", tz="UTC"))    # Monday
    assert not td.market_closed(pd.Timestamp("2026-10-09 15:00", tz="UTC"))    # Friday afternoon


def test_missing_key_explains_itself(tmp_path):
    r = td.test_key(tmp_path)
    assert not r["ok"] and "TWELVEDATA_API_KEY" in r["message"]


def test_closed_market_candles_are_dropped():
    # IST-naive candles around the weekend of Fri 9 – Sun 11 Oct 2026 (New York on EDT)
    idx = pd.DatetimeIndex(pd.to_datetime([
        "2026-10-10 02:00",   # Fri 16:30 NY  → open
        "2026-10-10 02:35",   # Fri 17:05 NY  → closed (weekend starts)
        "2026-10-10 18:00",   # Sat           → closed
        "2026-10-12 03:00",   # Sun 17:30 NY  → closed
        "2026-10-12 03:35",   # Sun 18:05 NY  → open
        "2026-10-13 02:40",   # Mon 17:10 NY  → daily break
        "2026-10-13 03:40",   # Mon 18:10 NY  → open
    ]), name="time")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=idx)
    kept = td.market_hours_only(df, "5m").index.strftime("%m-%d %H:%M").tolist()
    assert kept == ["10-10 02:00", "10-12 03:35", "10-13 03:40"]
    daily = pd.DataFrame({"close": [1, 2, 3]}, index=pd.to_datetime(["2026-10-09 05:30", "2026-10-10 05:30", "2026-10-11 05:30"]))
    assert len(td.market_hours_only(daily, "1d")) == 1      # Saturday and Sunday candles dropped


def test_markets_registry_and_kinds(tmp_path):
    from signalbot import symbols as S
    S.load_watchlist(tmp_path)
    eur = S.add(tmp_path, "EUR/USD", "Euro / US Dollar", "Physical Currency")
    btc = S.add(tmp_path, "BTC/USD", "Bitcoin", "Digital Currency")
    aapl = S.add(tmp_path, "AAPL", "Apple", "Common Stock", "NASDAQ")
    assert [eur["kind"], btc["kind"], aapl["kind"]] == ["fx", "crypto", "stock"]
    assert eur["news"] == ["EUR", "USD", "All"]
    assert list(S.load_watchlist(tmp_path)) == ["XAUUSD", "EURUSD", "BTCUSD", "AAPL"]     # persisted
    S.remove(tmp_path, "BTCUSD")
    assert "BTCUSD" not in S.load_watchlist(tmp_path)
    assert S.pf(1.084231) == "1.08423" and S.pf(4194.214) == "4,194.21"
    assert abs(S.default_cost(eur, 1.08) - 0.000108) < 1e-9 and S.default_cost(S.SYMBOLS["XAUUSD"], 4200) == 0.30
    S.remove(tmp_path, "EURUSD"); S.remove(tmp_path, "AAPL")


def test_hours_by_kind():
    sat = pd.Timestamp("2026-10-10 12:00", tz="UTC")
    tue_night = pd.Timestamp("2026-10-13 03:00", tz="UTC")          # 23:00 New York
    assert td.market_closed(sat, "metal") and not td.market_closed(sat, "crypto") and td.market_closed(sat, "stock")
    assert not td.market_closed(tue_night, "fx") and td.market_closed(tue_night, "stock")
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-10-10 18:00", "2026-10-12 19:00"]), name="time")   # Sat, Mon (IST)
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)
    assert len(td.market_hours_only(df, "5m", "crypto")) == 2 and len(td.market_hours_only(df, "5m", "metal")) == 1
