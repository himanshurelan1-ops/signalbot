import numpy as np
import pandas as pd

from signalbot import data, report, twelvedata
from tests.test_report_and_web import offline  # noqa: F401  (fixture)


def _bars(start, n, freq):
    idx = pd.date_range(start, periods=n, freq=freq)
    c = 100 + np.cumsum(np.random.default_rng(2).normal(0, 1, n))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c + 0.5, "volume": 1.0}, index=idx)
    df.index.name = "time"
    return df


def test_four_hour_candles_sit_on_the_utc_grid():
    h1 = _bars("2026-03-02 05:30", 48, "1h")              # IST 05:30 = 00:00 UTC
    h4 = twelvedata.resample_tf(h1, "4h")
    assert len(h4) == 12
    assert [str(t)[11:16] for t in h4.index[:3]] == ["05:30", "09:30", "13:30"]
    first = h1.iloc[:4]
    assert h4.iloc[0]["open"] == first["open"].iloc[0] and h4.iloc[0]["close"] == first["close"].iloc[-1]
    assert h4.iloc[0]["high"] == first["high"].max() and h4.iloc[0]["volume"] == 4


def test_thirty_minutes_from_fifteen_and_weeks_start_monday():
    m30 = twelvedata.resample_tf(_bars("2026-03-02 09:15", 8, "15min"), "30m")
    assert [str(t)[11:16] for t in m30.index[:2]] == ["09:00", "09:30"]     # 03:30 UTC grid
    d = _bars("2026-03-04", 15, "B")                                         # Wednesday start
    w = twelvedata.resample_tf(d, "1w")
    assert all(t.weekday() == 0 for t in w.index) and str(w.index[0].date()) == "2026-03-02"


def test_derived_timeframes_build_reports(offline):  # noqa: F811
    for tf in ("1w",):
        rep = report.build("XAUUSD", offline, tf, refresh=False)
        assert rep["tf"] == tf and not rep["intraday"] and rep["chart"]["bars"]
        assert isinstance(rep["chart"]["bars"][0]["time"], str)
    assert list(data.TIMEFRAMES) == ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
