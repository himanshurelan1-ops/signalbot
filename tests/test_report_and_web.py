import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

from signalbot import data, report, scalping, web


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Daily + 15m synthetic data on disk, and no network."""
    idx = pd.bdate_range("2015-01-01", periods=1500) + pd.Timedelta(hours=5, minutes=30)   # Twelve Data daily label
    c = 100 + np.cumsum(np.random.default_rng(1).normal(0.05, 1, len(idx)))
    daily = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1}, index=idx)
    daily.index.name = "time"
    m = pd.date_range("2026-03-02 09:15", periods=25, freq="15min")
    m15 = pd.DataFrame({"open": c[-1], "high": c[-1] + 1, "low": c[-1] - 1, "close": c[-1], "volume": 1}, index=m)
    m15.index.name = "time"
    (tmp_path / "data").mkdir()
    daily.to_csv(tmp_path / "data" / "XAUUSD_1d.csv")
    m15.to_csv(tmp_path / "data" / "XAUUSD_15m.csv")
    from signalbot import news, twelvedata
    monkeypatch.setattr(twelvedata, "update", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(news, "fetch_raw", lambda root: [])
    from signalbot import volume
    monkeypatch.setattr(volume, "_download", lambda tf: (_ for _ in ()).throw(RuntimeError("offline")))
    return tmp_path


def test_report_builds_from_cache_without_network(offline):
    rep = report.build("XAUUSD", offline, refresh=False)
    assert rep["symbol"] == "XAUUSD" and rep["history"]["bars"] == 1500
    assert {s["setup"] for s in rep["setups"]} == {"breakout", "pullback", "volume breakout", "trendline break", "channel bounce"} | set(scalping.SETUPS)
    assert "forming" in rep["patterns"] and "lines" in rep["trendlines"]
    for s in rep["setups"]:
        assert s["signal"]["status"] in ("signal", "open", "none")
        assert "win_rate" in s["stats"]
    assert rep["intraday"] is False and rep["tf"] == "1d"
    assert len(rep["chart"]["bars"]) == 260


def test_intraday_timeframe_carries_positions_and_labels_times(offline):
    rep = report.build("XAUUSD", offline, "15m", refresh=False)
    assert rep["intraday"] and not rep["flat_by_close"]                 # gold runs round the clock
    assert {s["setup"] for s in rep["setups"]} == {"breakout", "pullback", "volume breakout", "trendline break", "channel bounce"} | set(scalping.SETUPS)
    assert len(rep["as_of"]) == 16                                       # date AND time, not just the date
    assert isinstance(rep["chart"]["bars"][0]["time"], int)          # epoch seconds for intraday


def test_tuning_parameters_change_the_rules(offline):
    a = report.build("XAUUSD", offline, refresh=False)
    b = report.build("XAUUSD", offline, refresh=False, stop_atr=1.0, target_atr=4.0, max_bars=5)
    assert b["params"]["stop_atr"] == 1.0 and b["params"]["target_atr"] == 4.0 and b["params"]["max_bars"] == 5
    assert a["setups"][0]["stats"] != b["setups"][0]["stats"]


def test_long_only_drops_sell_trades(offline):
    rep = report.build("XAUUSD", offline, refresh=False, long_only=True)
    for s in rep["setups"]:
        assert s["stats"]["sells"] == 0
        assert s["signal"]["side"] != "SELL"


def test_web_serves_page_and_report(offline):
    page = (web.Path(web.__file__).parent / "web_ui.html").read_text()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(offline, page))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        html = urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "signalbot" in html
        meta = json.loads(urlopen(f"http://127.0.0.1:{port}/api/symbols").read())
        assert {s["symbol"] for s in meta["symbols"]} == {"XAUUSD"}
        assert [t["tf"] for t in meta["timeframes"]] == ["1d", "1h", "15m", "5m"]
        rep = json.loads(urlopen(f"http://127.0.0.1:{port}/api/report?symbol=XAUUSD&tf=15m&stop_atr=1.5").read())
        assert rep["symbol"] == "XAUUSD" and rep["tf"] == "15m" and rep["params"]["stop_atr"] == 1.5
    finally:
        httpd.shutdown()


def test_signal_keys_are_stable_and_deduped(offline, tmp_path):
    from signalbot.notify import already_sent, mark_sent, signal_keys
    rep = report.build("XAUUSD", offline, refresh=False)
    keys = signal_keys(rep)
    for k, msg in keys:
        assert rep["name"] in msg and "→" in msg
        assert not already_sent(tmp_path, k)
        mark_sent(tmp_path, k)
        assert already_sent(tmp_path, k)


def test_forming_candle_is_never_shown_to_the_rules():
    from signalbot.data import drop_incomplete
    idx = pd.to_datetime(["2026-10-07 09:15", "2026-10-07 09:30", "2026-10-07 09:45"])
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    assert len(drop_incomplete(df, "15m", pd.Timestamp("2026-10-07 09:50"))) == 2   # 09:45 bar still open
    assert len(drop_incomplete(df, "15m", pd.Timestamp("2026-10-07 10:00"))) == 3   # closed


def test_forming_candle_is_built_from_minute_bars(tmp_path, monkeypatch):
    from signalbot import live
    # closed 5m candles up to 10:50; minute bars run to 11:06 → forming bucket 11:05, bridge buckets 10:55 & 11:00
    closed = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                          index=pd.date_range("2026-10-07 09:15", "2026-10-07 10:50", freq="5min"))
    monkeypatch.setattr(live, "load", lambda *a, **k: closed)
    m = pd.date_range("2026-10-07 10:40", "2026-10-07 11:06", freq="1min")
    px = np.linspace(100, 110, len(m))
    m1 = pd.DataFrame({"open": px, "high": px + 0.5, "low": px - 0.5, "close": px + 0.2}, index=m)
    monkeypatch.setattr(live, "minute_bars", lambda symbol: m1)
    r = live.forming_candle(tmp_path, "XAUUSD", "5m")
    assert r["closed_until"] == "2026-10-07 10:50"
    assert pd.Timestamp(r["time"], unit="s").strftime("%H:%M") == "11:05"
    assert [pd.Timestamp(b["time"], unit="s").strftime("%H:%M") for b in r["bars"]] == ["10:55", "11:00", "11:05"]
    assert r["high"] >= r["close"] >= r["low"] and r["open"] == m1.loc["2026-10-07 11:05", "open"]
