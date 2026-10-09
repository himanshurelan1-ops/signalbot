import numpy as np
import pandas as pd

from signalbot import strategy, volume
from signalbot.strategy import Params, enrich, signals


def _spot(n=300, freq="5min", start="2026-10-01 09:30"):
    idx = pd.date_range(start, periods=n, freq=freq, name="time")
    c = 4000 + np.arange(n) * 0.5
    return pd.DataFrame({"open": c - 0.2, "high": c + 1, "low": c - 1, "close": c, "volume": 0.0}, index=idx)


def test_comex_volume_is_aligned_to_spot_candles(tmp_path, monkeypatch):
    spot = _spot(50)
    comex = pd.Series(100.0, index=spot.index[:-3], name="volume")        # newest 3 not printed yet (delay)
    comex.iloc[-1] = 400.0
    monkeypatch.setattr(volume, "_download", lambda tf: comex)
    d = volume.attach(spot, tmp_path, "5m")
    assert d["vol"].iloc[:-3].notna().all() and d["vol"].iloc[-3:].isna().all()
    assert round(d["rvol"].iloc[-4], 2) == 4.0                            # 400 vs an average of 100
    lt = volume.latest(d)
    assert lt["comex"] is None and lt["comex_until"] == str(spot.index[-4])[:16]


def test_daily_volume_matches_by_date(tmp_path, monkeypatch):
    spot = _spot(5, freq="1D", start="2026-10-05 05:30")                  # Twelve Data daily labels: 05:30 IST
    comex = pd.Series([1, 2, 3, 4, 5.0], index=pd.DatetimeIndex(pd.date_range("2026-10-05", periods=5), name="time"))
    monkeypatch.setattr(volume, "_download", lambda tf: comex)
    d = volume.attach(spot, tmp_path, "1d")
    assert d["vol"].tolist() == [1, 2, 3, 4, 5.0]


def test_tick_volume_fills_in_when_comex_is_late(tmp_path, monkeypatch):
    spot = _spot(40)
    spot["volume"] = 10.0
    spot.iloc[-1, spot.columns.get_loc("volume")] = 30.0                    # a busy forming… now closed candle
    monkeypatch.setattr(volume, "_download", lambda tf: pd.Series(dtype=float))
    d = volume.attach(spot, tmp_path, "5m")
    assert d["vol"].isna().all() and d["rvol_tick"].iloc[-1] == 3.0
    assert strategy._rvol(d).iloc[-1] == 3.0                               # the rule falls back to ticks


def test_volume_breakout_needs_the_volume(tmp_path, monkeypatch):
    spot = _spot(300)
    spot.iloc[-1, spot.columns.get_loc("close")] += 20                      # a clear breakout candle
    spot.iloc[-1, spot.columns.get_loc("high")] += 20
    quiet = pd.Series(100.0, index=spot.index)
    loud = quiet.copy(); loud.iloc[-1] = 300.0
    for vol, expect in ((quiet, 0), (loud, 1)):
        monkeypatch.setattr(volume, "_download", lambda tf, v=vol: v)
        (tmp_path / "data" / "COMEX_5m.csv").unlink(missing_ok=True)
        d = enrich(volume.attach(spot, tmp_path, "5m"), Params())
        assert signals(d, "breakout").iloc[-1] == 1
        assert signals(d, "volume breakout").iloc[-1] == expect
