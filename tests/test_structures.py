import numpy as np
import pandas as pd

from signalbot import structures as S
from signalbot.lines import Line
from signalbot.strategy import Params, enrich


def _ln(slope, icpt):
    return Line("R", slope, icpt, 2, 0, 10)


def test_shapes():
    a = 1.0
    assert S.shape(_ln(-0.05, 110), _ln(-0.05, 100), a, 50) == "falling channel"
    assert S.shape(_ln(0.05, 110), _ln(0.05, 100), a, 50) == "rising channel"
    assert S.shape(_ln(-0.08, 110), _ln(-0.02, 100), a, 50) == "falling wedge"
    assert S.shape(_ln(0.02, 110), _ln(0.08, 100), a, 50) == "rising wedge"
    assert S.shape(_ln(-0.05, 110), _ln(0.05, 100), a, 50) == "triangle"
    assert S.shape(_ln(0.05, 110), _ln(-0.05, 100), a, 50) == "broadening"


def test_fib_levels_up_swing():
    lv = {round(x["r"], 3): x["price"] for x in S.fib_levels(100.0, 200.0, True)}
    assert lv[0.0] == 200 and lv[1.0] == 100
    assert lv[0.5] == 150 and abs(lv[0.618] - 138.2) < 1e-9
    targets = sorted(x["price"] for x in S.fib_levels(100.0, 200.0, True) if x.get("ext"))
    assert abs(targets[0] - 227.2) < 1e-9 and abs(targets[1] - 261.8) < 1e-9


def _zigzag(points, bars_per_leg=12):
    """Piecewise-linear closes through `points`, small candles around them."""
    c = []
    for a, b in zip(points[:-1], points[1:]):
        c += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    c.append(points[-1])
    c = np.array(c)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-01-05 05:30", periods=len(c), freq="15min")
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.3, "low": np.minimum(o, c) - 0.3, "close": c, "volume": 1.0}, index=idx)
    df.index.name = "time"
    return df


def test_golden_zone_fires_on_pullback_into_50_to_618():
    # base, rally 100 → 140, pull back to ~120 (≈ 50–55%), then bounce
    pts = [100, 104, 100, 104, 100] * 6 + [100, 140, 119.5, 135]
    d = enrich(_zigzag(pts), Params(timed=True))
    sig = S.rolling(d)["fib golden zone"]
    hits = np.flatnonzero(sig)
    assert len(hits) and sig[hits[-1]] == 1
    t = d.index[hits[-1]]
    assert d["low"].loc[t] <= 120.5 and d["close"].loc[t] >= 140 - 0.618 * 40


def test_analyse_runs_and_returns_setups():
    pts = [100, 110, 102, 108, 103, 106, 104] * 8
    d = enrich(_zigzag(pts), Params(timed=True))
    out = S.analyse(d, Params(timed=True))
    assert [s["setup"] for s in out["setups"]] == list(S.SETUPS)
    assert isinstance(out["channels"], list)
