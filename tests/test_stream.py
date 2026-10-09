import json

import pandas as pd

from signalbot import data, stream, twelvedata as td


def _server_frame(payload: bytes, op=0x1, fin=True):
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | op])
    head += bytes([n]) if n < 126 else bytes([126]) + n.to_bytes(2, "big")
    return head + payload


def test_frame_reader_handles_split_and_fragmented_frames():
    msg = json.dumps({"event": "price", "price": 4190.5, "pad": "x" * 300}).encode()   # 16-bit length
    raw = _server_frame(msg) + _server_frame(b"ping!", op=0x9)
    r = stream.FrameReader()
    out = []
    for i in range(0, len(raw), 7):                      # arrives in dribbles
        out += r.feed(raw[i:i + 7])
    assert out == [(0x1, msg), (0x9, b"ping!")]
    frag = _server_frame(b'{"a":', fin=False) + _server_frame(b'1}', op=0x0)
    assert stream.FrameReader().feed(frag) == [(0x1, b'{"a":1}')]


def test_client_frames_are_masked_and_decode_back():
    payload = b'{"action":"heartbeat"}'
    f = stream.encode_frame(payload)
    assert f[0] == 0x81 and f[1] & 0x80                    # FIN + text, masked
    assert stream.FrameReader().feed(f) == [(0x1, payload)]


def test_ticks_become_minute_bars_and_are_broadcast(tmp_path):
    st = stream.Stream(tmp_path)
    q = st.subscribe()
    base = pd.Timestamp("2026-10-09 06:40:00", tz="UTC").timestamp()      # 12:10 IST
    for dt, px in [(1, 4190.0), (20, 4192.0), (40, 4189.5), (59, 4191.0), (61, 4193.0)]:
        st.on_price(px, base + dt)
    m1 = pd.read_csv(tmp_path / "data" / "XAUUSD_1m.csv", index_col="time", parse_dates=True)
    assert list(m1.index.strftime("%H:%M")) == ["12:10"]                     # only the completed minute
    assert m1.iloc[0][["open", "high", "low", "close"]].tolist() == [4190.0, 4192.0, 4189.5, 4191.0]
    cur = st.current_bar()
    assert cur["time"].strftime("%H:%M") == "12:11" and cur["open"] == 4193.0
    msgs = [q.get_nowait() for _ in range(q.qsize())]
    ticks = [m for m in msgs if m["type"] == "tick"]
    assert len(ticks) == 5 and ticks[-1]["p"] == 4193.0


def test_partial_minute_merges_with_the_official_bar(tmp_path):
    (tmp_path / "data").mkdir()
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-10-09 12:09", "2026-10-09 12:10"]), name="time")
    pd.DataFrame({"open": [1.0, 4190.0], "high": [1.0, 4195.0], "low": [1.0, 4189.0], "close": [1.0, 4191.0],
                  "volume": 0.0}, index=idx).to_csv(tmp_path / "data" / "XAUUSD_1m.csv")
    row = pd.DataFrame([{"open": 4191.5, "high": 4192.0, "low": 4186.0, "close": 4187.0, "volume": 0.0}],
                       index=pd.DatetimeIndex([pd.Timestamp("2026-10-09 12:10")], name="time"))
    td.append_minutes(tmp_path, row)
    m1 = pd.read_csv(tmp_path / "data" / "XAUUSD_1m.csv", index_col="time", parse_dates=True)
    assert len(m1) == 2
    assert m1.iloc[-1][["open", "high", "low", "close"]].tolist() == [4190.0, 4195.0, 4186.0, 4187.0]


def test_cache_tops_up_as_soon_as_the_forming_candle_closes(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("5min")
    idx = pd.date_range(end=now - pd.Timedelta(minutes=5), periods=400, freq="5min", name="time")
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=idx) \
        .to_csv(tmp_path / "data" / "XAUUSD_5m.csv")              # written seconds ago, but its last candle has closed
    called = {}
    monkeypatch.setattr(td, "update", lambda root, interval, cached, **k: called.setdefault("yes", cached))
    data.load("XAUUSD", "5m", tmp_path)
    assert "yes" in called
