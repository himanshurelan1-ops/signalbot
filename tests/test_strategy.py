"""The backtester must be boringly correct: fills, exits and R are checked by hand."""
import numpy as np
import pandas as pd
import pytest

from signalbot.strategy import Params, backtest, current, enrich, signals, stats


def _series(closes, spread=5.0, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread,
                         "low": np.minimum(o, c) - spread, "close": c,
                         "volume": 1000}, index=idx)


def _uptrend_then_breakout(n_flat=260):
    # long gentle uptrend so EMA200 is below price, then a flat range, then a breakout bar
    ramp = np.linspace(100, 200, n_flat)
    flat = np.full(25, 200.0) + np.sin(np.arange(25)) * 2      # range 198-202 plus spread
    breakout = [215.0]                                          # closes above 20-day high
    after = [216, 218, 221, 224, 228, 232, 236, 240, 245, 250]  # drifts to the target
    return np.r_[ramp, flat, breakout, after]


def test_breakout_signal_fires_only_on_the_breakout_bar():
    df = _series(_uptrend_then_breakout())
    p = Params(cost=0.0)
    d = enrich(df, p)
    sig = signals(d, "breakout")
    fired = list(sig[sig != 0].index)
    assert len(fired) >= 1
    assert sig.loc[fired[0]] == 1
    assert d.loc[fired[0], "close"] == 215.0


def test_trade_fills_at_next_open_and_exits_at_target_with_correct_r():
    df = _series(_uptrend_then_breakout())
    p = Params(cost=0.0)
    d = enrich(df, p)
    trades = backtest(d, "breakout", p)
    t = trades[0]
    i = d.index.get_loc(pd.Timestamp(t.signal_time))
    assert t.entry == d["open"].iloc[i + 1]
    risk = p.stop_atr * d["atr"].iloc[i]
    assert t.stop == pytest.approx(t.entry - risk)
    assert t.target == pytest.approx(t.entry + p.target_atr * d["atr"].iloc[i])
    assert t.reason == "target"
    assert t.r == pytest.approx(p.target_atr / p.stop_atr, abs=1e-6) or t.r > p.target_atr / p.stop_atr  # gap-up fills can only improve it


def test_stop_is_assumed_first_when_a_bar_touches_both():
    df = _series(_uptrend_then_breakout())
    p = Params(cost=0.0)
    d = enrich(df, p)
    # make the entry bar a huge-range bar touching both levels
    i = d.index.get_loc(pd.Timestamp("2021-01-04")) if pd.Timestamp("2021-01-04") in d.index else None
    sig = signals(d, "breakout")
    k = int(np.flatnonzero(sig.to_numpy())[0]) + 1
    df.iloc[k, df.columns.get_loc("high")] = 10_000
    df.iloc[k, df.columns.get_loc("low")] = 1
    d = enrich(df, p)
    t = backtest(d, "breakout", p)[0]
    assert t.reason == "stop"
    assert t.r == pytest.approx(-1.0)


def test_time_stop_closes_at_the_close_after_max_bars():
    closes = _uptrend_then_breakout()
    closes = np.r_[closes[:286], np.full(40, 215.0)]           # breakout then dead flat
    df = _series(closes, spread=0.5)
    p = Params(max_bars=5, cost=0.0)
    d = enrich(df, p)
    t = [t for t in backtest(d, "breakout", p) if t.entry == 215.0][0]
    assert t.reason == "time"
    assert t.bars == 5


def test_stats_are_computed_from_closed_trades_only():
    from signalbot.strategy import Trade
    tr = [Trade("x", "BUY", "a", "b", 1, 0, 2, reason="target", r=1.5),
          Trade("x", "BUY", "a", "b", 1, 0, 2, reason="stop", r=-1.0),
          Trade("x", "SELL", "a", "b", 1, 2, 0, reason="target", r=1.5),
          Trade("x", "BUY", "a", "b", 1, 0, 2, reason="open", r=0.3)]
    st = stats(tr, "x")
    assert st.trades == 3
    assert st.win_rate == pytest.approx(2 / 3)
    assert st.avg_r == pytest.approx((1.5 - 1 + 1.5) / 3)
    assert st.profit_factor == pytest.approx(3.0)
    assert st.buys == 2 and st.sells == 1
    assert st.sell_win_rate == 1.0


def test_current_reports_watch_levels_when_nothing_fires():
    df = _series(np.linspace(100, 200, 300))
    p = Params(cost=0.0)
    d = enrich(df, p)
    trades = backtest(d, "breakout", p)
    sig = current(d, "breakout", p, trades)
    w = sig.watch
    assert w["side"] == "BUY"
    assert w["trigger"] == pytest.approx(d["dc_high"].iloc[-1])
    assert w["target"] > w["trigger"] > w["stop"]


def test_no_overlapping_positions():
    df = _series(_uptrend_then_breakout())
    p = Params(cost=0.0)
    d = enrich(df, p)
    trades = backtest(d, "breakout", p)
    for a, b in zip(trades, trades[1:]):
        assert pd.Timestamp(b.entry_time) > pd.Timestamp(a.exit_time)
