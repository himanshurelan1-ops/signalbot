import numpy as np
import pandas as pd

from signalbot import smc
from signalbot.strategy import Params, enrich, signals


def _df(rows, start="2026-10-05 12:30", freq="5min"):
    """rows: (open, high, low, close)"""
    idx = pd.date_range(start, periods=len(rows), freq=freq, name="time")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 0.0}, index=idx)


def _flat(n, px=100.0):
    return [(px, px + 0.5, px - 0.5, px)] * n


def test_swing_high_is_only_known_after_confirmation():
    rows = _flat(5) + [(100, 110, 99, 100)] + _flat(5)          # spike at bar 5
    sh, _ = smc.swings(_df(rows))
    assert np.isnan(sh[7]) and sh[8] == 110                    # confirmed 3 bars later


def test_liquidity_sweep_fires_once_on_wick_through_and_close_back():
    rows = _flat(230) + [(100, 110, 99, 100)] + _flat(6)       # swing high 110
    rows += [(100, 111, 99, 100)]                               # wick above, close back below → SELL
    rows += [(100, 111.5, 99, 100)]                             # same level again → no second trade
    d = enrich(_df(rows), Params(timed=True))
    s = signals(d, "liquidity sweep")
    assert s.iloc[-2] == -1 and s.iloc[-1] == 0 and (s == -1).sum() == 1


def test_pdh_sweep_uses_yesterdays_high():
    # day 1 (UTC) high 105; day 2 wicks to 106 and closes back below
    d1 = pd.date_range("2026-10-05 05:30", periods=288, freq="5min")          # 00:00–23:55 UTC
    rows = [(100, 101, 99, 100)] * 288
    rows[100] = (100, 105, 99, 100)
    rows += [(100, 101, 99, 100)] * 20 + [(100, 106, 99, 104)]
    d = enrich(_df(rows, start="2026-10-05 05:30"), Params(timed=True))
    assert d["pdh"].iloc[-1] == 105
    assert signals(d, "pdh/pdl sweep").iloc[-1] == -1


def test_fair_value_gap_retest_buys_in_an_uptrend():
    up = [(100 + k * 0.2, 100.5 + k * 0.2, 99.5 + k * 0.2, 100 + k * 0.2) for k in range(240)]
    base = up[-1][3]
    gap = [(base, base + 1, base - 0.5, base + 0.8), (base + 1, base + 4, base + 0.9, base + 3.8),
           (base + 3.8, base + 5, base + 2, base + 4.8)]               # low[2]=base+2 > high[0]=base+1
    back = [(base + 2.5, base + 3.2, base + 1.5, base + 3.0)]           # dips into the gap, closes green above its bottom
    d = enrich(_df(up + gap + back), Params(timed=True))
    assert signals(d, "fvg retest").iloc[-1] == 1


def _zigzag(points, legs=6):
    rows = []
    for a, b in zip(points, points[1:]):
        for k in range(1, legs + 1):
            px = a + (b - a) * k / legs
            prev = a + (b - a) * (k - 1) / legs
            rows.append((prev, max(px, prev) + 0.2, min(px, prev) - 0.2, px))
    return rows


def test_choch_after_lower_highs():
    # lower highs (196, 192) and lower lows (190, 186, 182), then a rally through the last swing high
    rows = _flat(200, 200) + _zigzag([200, 190, 196, 186, 192, 182, 200])
    d = enrich(_df(rows), Params(timed=True))
    s = signals(d, "choch")
    assert (s == 1).sum() == 1 and s[s == 1].index[0] > d.index[-8]       # fired on the final rally
    lv = smc.levels(d)
    assert lv["structure"] == "up" and lv["last_break"]["kind"] == "CHoCH"
