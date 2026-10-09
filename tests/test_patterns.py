import numpy as np
import pandas as pd

from signalbot import lines as L
from signalbot import patterns as P
from signalbot.strategy import Params, enrich


def _df(closes, wick=1.0, start="2026-01-01 09:15", freq="15min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick,
                         "close": c, "volume": 1}, index=idx)


def test_pivots_alternate_and_are_confirmed_late():
    path = [100, 102, 104, 106, 104, 102, 100, 98, 100, 102, 104, 106, 108, 106, 104]
    df = _df(path)
    pv = P.pivots(df, left=2, right=2)
    kinds = "".join(p.kind for p in pv)
    assert kinds in ("HLH", "LHLH", "HL", "HLHL") and "HH" not in kinds and "LL" not in kinds
    for p in pv:
        assert p.confirmed == p.i + 2


def test_double_top_is_detected_with_its_neckline():
    # two equal highs around 120 with a trough at 110 between, then price sitting below the second high
    pv = [P.Pivot(10, 100, "L", 13), P.Pivot(20, 120, "H", 23), P.Pivot(30, 110, "L", 33),
          P.Pivot(40, 120.2, "H", 43)]
    pat = P.detect_chart_pattern(pv, atr_now=2.0, now_i=45)
    assert pat is not None and pat.name == "double top" and pat.direction == -1
    assert pat.level == 110


def test_head_and_shoulders():
    pv = [P.Pivot(10, 100, "H", 13), P.Pivot(15, 95, "L", 18), P.Pivot(20, 110, "H", 23),
          P.Pivot(25, 95.5, "L", 28), P.Pivot(30, 100.3, "H", 33)]
    pat = P.detect_chart_pattern(pv, atr_now=2.0, now_i=35)
    assert pat is not None and pat.name == "head and shoulders"
    assert abs(pat.level - 96) < 1.5          # neckline through the two troughs, extended


def test_ascending_triangle_from_flat_highs_and_rising_lows():
    pv = [P.Pivot(0, 90, "L", 3), P.Pivot(10, 100, "H", 13), P.Pivot(20, 94, "L", 23),
          P.Pivot(30, 100.2, "H", 33), P.Pivot(40, 97, "L", 43), P.Pivot(50, 99.9, "H", 53)]
    pat = P.detect_chart_pattern(pv, atr_now=1.0, now_i=55)
    assert pat is not None and pat.name == "ascending triangle" and pat.direction == 1


def test_candlestick_patterns_fire_on_textbook_bars():
    # a steady decline (so price is below EMA20) then a bullish engulfing bar
    closes = list(np.linspace(120, 100, 30))
    df = _df(closes, wick=0.2)
    # previous bar: red 100.7 -> 100 ; engulfing bar: open 99.9, close 101.5
    df.iloc[-1, df.columns.get_loc("open")] = 99.9
    df.iloc[-1, df.columns.get_loc("close")] = 101.5
    df.iloc[-1, df.columns.get_loc("high")] = 101.7
    df.iloc[-1, df.columns.get_loc("low")] = 99.7
    names = {n for i, n in P.candlestick_events(df) if i == len(df) - 1}
    assert "bullish engulfing" in names


def test_outcome_scores_the_textbook_direction():
    d = {"open": np.array([100, 100, 100.0]), "high": np.array([100, 103, 104.0]), "low": np.array([100, 99.5, 99.0])}
    assert P.outcome(d, 0, +1, atr_i=1.0) == 1.0       # moved +1.5 ATR first → bullish call worked
    assert P.outcome(d, 0, -1, atr_i=1.0) == -1.0      # same move, bearish call failed
    assert P.outcome(d, 2, +1, atr_i=1.0) is None      # no future bars


def test_resistance_line_through_descending_highs_and_break_signal():
    # three descending swing highs respected, then a close through the line
    path = []
    for k, top in enumerate([110, 108, 106]):
        path += [100, 104, top, 104, 100, 98]
    path += [100, 103, 107, 109]                 # breaks the falling line
    df = _df(path, wick=0.3)
    p = Params()
    d = enrich(df, p)
    rs = L.rolling_signals(d, left=2, right=2)
    assert (rs["break"] == 1).any()
    bi = int(np.flatnonzero(rs["break"] == 1)[-1])
    assert bi >= len(path) - 3                   # the break is on one of the final bars


def test_analyse_returns_lines_and_records(tmp_path):
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1, 600))
    df = _df(closes)
    out = P.analyse(df)
    assert "forming" in out and isinstance(out["candle_stats"], list)
    d = enrich(df, Params())
    ln = L.analyse(d, Params())
    assert {s["setup"] for s in ln["setups"]} == {"trendline break", "channel bounce"}
    for l in ln["lines"]:
        assert l["kind"] in ("resistance", "support") and l["touches"] >= 2
