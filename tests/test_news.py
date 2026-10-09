import json

import pandas as pd

from signalbot import news


def _feed(root, events):
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "ff_calendar.json").write_text(json.dumps(events))


def test_blackout_windows_wrap_high_impact_events_only(tmp_path):
    now = pd.Timestamp("2026-10-07 20:00")              # IST
    _feed(tmp_path, [
        {"title": "Non-Farm Employment Change", "country": "USD", "impact": "High",
         "date": "2026-10-07T13:30:00-04:00", "forecast": "150K", "previous": "142K"},   # 23:00 IST
        {"title": "Unemployment Rate", "country": "USD", "impact": "High",
         "date": "2026-10-07T13:30:00-04:00"},                                           # same minute → merged
        {"title": "FOMC Member Speaks", "country": "USD", "impact": "Medium",
         "date": "2026-10-07T15:00:00-04:00"},                                           # 00:30 IST, no window
        {"title": "GDP", "country": "EUR", "impact": "High", "date": "2026-10-07T05:00:00-04:00"},  # irrelevant ccy
    ])
    st = news.status(tmp_path, ["USD"], now)
    assert [e["title"] for e in st["events"]] == ["Non-Farm Employment Change", "Unemployment Rate", "FOMC Member Speaks"]
    assert len(st["blackouts"]) == 1
    w = st["blackouts"][0]
    assert w["titles"] == ["Non-Farm Employment Change", "Unemployment Rate"]
    assert pd.Timestamp(w["start"], unit="s").strftime("%H:%M") == "22:45"
    assert pd.Timestamp(w["end"], unit="s").strftime("%H:%M") == "23:30"
    assert not st["in_blackout"] and "22:45–23:30" in st["next_text"]

    inside = news.status(tmp_path, ["USD"], pd.Timestamp("2026-10-07 23:05"))
    assert inside["in_blackout"] and "Non-Farm" in inside["blackout_text"]


def test_gold_daily_bar_completes_next_morning():
    from signalbot.data import drop_incomplete
    # a daily candle is labelled by its start and completes 24 hours later
    d = pd.DataFrame({"close": [1]}, index=pd.to_datetime(["2026-10-06 02:30"]))
    assert len(drop_incomplete(d, "1d", pd.Timestamp("2026-10-07 02:00"), symbol="XAUUSD")) == 0
    assert len(drop_incomplete(d, "1d", pd.Timestamp("2026-10-07 02:31"), symbol="XAUUSD")) == 1
