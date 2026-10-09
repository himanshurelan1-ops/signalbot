import numpy as np
import pandas as pd

from signalbot import scalping
from signalbot.strategy import Params, backtest, enrich, signals


def _df(close, start="2026-10-05 05:30", freq="5min"):
    c = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(c), freq=freq, name="time")
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 0.0}, index=idx)


def test_supertrend_follows_a_trend_and_flips():
    up = np.linspace(100, 160, 120)
    down = np.linspace(160, 100, 120)
    line, dirn = scalping.supertrend(_df(np.r_[up, down]))
    assert dirn.iloc[110] == 1 and dirn.iloc[-1] == -1
    assert line.iloc[110] < 160 and line.iloc[-1] > 100


def test_ema_cross_fires_once_with_the_trend():
    c = np.r_[np.linspace(100, 200, 260), np.linspace(200, 196, 20), np.linspace(196, 206, 20)]
    d = enrich(_df(c), Params(timed=True))
    s = signals(d, "ema 9/21 cross")
    assert (s == 1).sum() >= 1 and (s == -1).sum() == 0           # only BUYs while above the 200 EMA


def test_london_breakout_one_trade_a_day_after_the_asian_range():
    # 2026-10-05 00:00 UTC = 05:30 IST; flat Asian session, then a rally during London
    n = 24 * 12
    c = np.full(n, 100.0)
    c[7 * 12 + 2:] = np.linspace(101, 110, n - (7 * 12 + 2))
    d = enrich(_df(c), Params(timed=True))
    s = signals(d, "london breakout")
    fired = s[s != 0]
    assert len(fired) == 1 and fired.iloc[0] == 1
    t = (fired.index[0] - pd.Timedelta(hours=5, minutes=30))
    assert 7 <= t.hour < 10                                           # inside the London window (UTC)


def test_vwap_resets_each_utc_day():
    d = _df(np.r_[np.full(288, 100.0), np.full(288, 200.0)])           # two UTC days
    v = scalping.vwap(d)
    assert abs(v.iloc[287] - 100) < 1e-9 and abs(v.iloc[288] - 200) < 1e-9


def test_spread_cost_is_taken_off_every_trade():
    c = np.r_[np.linspace(100, 200, 300), [230, 230, 230, 230]]
    d = enrich(_df(c), Params())
    free = backtest(d, "breakout", Params(cost=0.0))
    paid = backtest(d, "breakout", Params(cost=1.0))
    assert len(free) == len(paid) > 0
    for a, b in zip(free, paid):
        risk = abs(a.entry - a.stop)
        assert abs((a.r - b.r) - 1.0 / risk) < 1e-9
