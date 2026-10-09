"""Live ticks for every market on the watchlist, from Twelve Data's WebSocket, pushed to the page.

One connection per signalbot server (the free plan includes it). Every price
change is

  * broadcast to the open page(s) over Server-Sent Events (/api/stream), so the
    forming candle and the price label move the moment Twelve Data sends a tick;
  * folded into 1-minute bars (with the tick count as tick volume), written to the shared minute cache as each
    minute completes — so a 5m / 15m / 1h candle is closed, and the rules re-run
    on it, seconds after it ends, without spending REST credits.

Standard library only (a small RFC 6455 client), so nothing extra to install.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import socket
import ssl
import struct
import threading
import time
import urllib.parse
from pathlib import Path

log = logging.getLogger("signalbot.stream")

HOST = "ws.twelvedata.com"
PATH = "/v1/quotes/price"
SYMBOL = "XAU/USD"
HEARTBEAT = 10           # Twelve Data asks for a heartbeat every 10 s
IST_OFFSET = 19800       # seconds


# ── minimal WebSocket client ────────────────────────────────────────────────

class WSClosed(Exception):
    pass


def _open_socket(host: str, port: int = 443, timeout: float = 20) -> socket.socket:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        pu = urllib.parse.urlparse(proxy if "://" in proxy else "http://" + proxy)
        raw = socket.create_connection((pu.hostname, pu.port or 80), timeout)
        auth = ""
        if pu.username:
            cred = base64.b64encode(f"{urllib.parse.unquote(pu.username)}:{urllib.parse.unquote(pu.password or '')}".encode()).decode()
            auth = f"Proxy-Authorization: Basic {cred}\r\n"
        raw.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}\r\n".encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = raw.recv(4096)
            if not chunk:
                raise ConnectionError("proxy closed the CONNECT tunnel")
            resp += chunk
        first = resp.split(b"\r\n", 1)[0]
        if b" 200" not in first:
            raise ConnectionError("proxy refused: " + first.decode(errors="ignore"))
    else:
        raw = socket.create_connection((host, port), timeout)
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile and os.path.exists(cafile) else ssl.create_default_context()
    return ctx.wrap_socket(raw, server_hostname=host)


class FrameReader:
    """Buffers bytes and yields complete (opcode, payload) frames; survives socket timeouts mid-frame."""

    def __init__(self):
        self.buf = b""
        self.parts: list[bytes] = []
        self.first_op = None

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        out = []
        while True:
            if len(self.buf) < 2:
                break
            b0, b1 = self.buf[0], self.buf[1]
            fin, op = b0 & 0x80, b0 & 0x0F
            masked, n = b1 & 0x80, b1 & 0x7F
            i = 2
            if n == 126:
                if len(self.buf) < 4:
                    break
                n = struct.unpack(">H", self.buf[2:4])[0]
                i = 4
            elif n == 127:
                if len(self.buf) < 10:
                    break
                n = struct.unpack(">Q", self.buf[2:10])[0]
                i = 10
            mask = b""
            if masked:
                if len(self.buf) < i + 4:
                    break
                mask = self.buf[i:i + 4]
                i += 4
            if len(self.buf) < i + n:
                break
            payload = self.buf[i:i + n]
            if mask:
                payload = bytes(c ^ mask[k % 4] for k, c in enumerate(payload))
            self.buf = self.buf[i + n:]
            if op in (0x8, 0x9, 0xA):                 # control frames are never fragmented
                out.append((op, payload))
                continue
            if op != 0:
                self.first_op, self.parts = op, []
            self.parts.append(payload)
            if fin:
                out.append((self.first_op or op, b"".join(self.parts)))
                self.parts, self.first_op = [], None
        return out


def encode_frame(payload: bytes, op: int = 0x1) -> bytes:
    """Client → server frames must be masked."""
    head = bytes([0x80 | op])
    n = len(payload)
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    mask = os.urandom(4)
    return head + mask + bytes(c ^ mask[k % 4] for k, c in enumerate(payload))


def ws_connect(host: str, path: str) -> tuple[ssl.SSLSocket, bytes]:
    s = _open_socket(host)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: signalbot/1.0\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            raise ConnectionError("server closed during the WebSocket handshake")
        resp += chunk
    head, rest = resp.split(b"\r\n\r\n", 1)
    status = head.split(b"\r\n", 1)[0].decode(errors="ignore")
    if " 101" not in status:
        raise ConnectionError(f"WebSocket handshake refused: {status}")
    return s, rest


# ── the stream ──────────────────────────────────────────────────────────────

class Stream:
    """One WebSocket for all watchlist markets. Ticks are broadcast to the page and folded into
    1-minute bars per market (tick counts as volume for spot fx/metals, which have no exchange volume)."""

    def __init__(self, root: Path):
        self.root = root
        self.state = "starting"
        self.message = ""
        self.last_price: dict[str, float] = {}
        self.last_tick_at: dict[str, float] = {}
        self.last_tick: float = 0.0
        self.bars: dict[str, dict] = {}
        self.subscribed: set[str] = set()
        self._resub = threading.Event()
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    # ── markets ──
    @staticmethod
    def _markets() -> dict[str, dict]:
        from .symbols import SYMBOLS
        return {m["td"]: m for m in SYMBOLS.values()}

    def resubscribe(self) -> None:
        """Call after the watchlist changes."""
        self._resub.set()

    # ── pub/sub for the page ──
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        q.put(self.status_msg())
        for key, px in self.last_price.items():
            q.put({"type": "tick", "s": key, "p": px, "t": int(self.last_tick_at.get(key, 0) * 1000)})
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _broadcast(self, msg: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def status_msg(self) -> dict:
        return {"type": "status", "state": self.state, "message": self.message}

    def _set_state(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message
        self._broadcast(self.status_msg())

    def live(self, key: str | None = None, within: float = 30) -> bool:
        if self.state != "live":
            return False
        t = self.last_tick_at.get(key, 0) if key else self.last_tick
        return time.time() - t < within

    # ── ticks → 1-minute bars ──
    def on_price(self, price: float, t: float | None = None, td: str = SYMBOL) -> None:
        from .symbols import key_of
        t = t or time.time()
        key = key_of(td)
        minute = int(t // 60) * 60
        done = None
        with self._lock:
            b = self.bars.get(key)
            if b is None or minute > b["m"]:
                done = b
                self.bars[key] = {"m": minute, "open": price, "high": price, "low": price, "close": price, "n": 1}
            elif minute == b["m"]:
                b["high"], b["low"], b["close"] = max(b["high"], price), min(b["low"], price), price
                b["n"] += 1
            self.last_price[key], self.last_tick_at[key], self.last_tick = price, t, t
        if done is not None:
            self._persist(td, done)
        self._broadcast({"type": "tick", "s": key, "p": price, "t": int(t * 1000)})

    def current_bar(self, key: str = "XAUUSD") -> dict | None:
        """The minute in progress for one market, as {time (IST-naive), open, high, low, close, volume}."""
        with self._lock:
            b = dict(self.bars[key]) if key in self.bars else None
        if not b:
            return None
        import pandas as pd
        b["time"] = pd.Timestamp(b.pop("m") + IST_OFFSET, unit="s")
        b["volume"] = float(b.pop("n"))
        return b

    def _persist(self, td: str, b: dict) -> None:
        try:
            import pandas as pd
            from . import twelvedata
            kind = self._markets().get(td, {}).get("kind", "metal")
            vol = float(b["n"]) if kind != "stock" else 0.0   # stocks: real volume comes from REST; crypto feed has none
            row = pd.DataFrame([{"open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"], "volume": vol}],
                               index=pd.DatetimeIndex([pd.Timestamp(b["m"] + IST_OFFSET, unit="s")], name="time"))
            twelvedata.append_minutes(self.root, row, symbol=td)
        except Exception as e:
            log.warning("could not store the 1-minute bar: %s", e)

    # ── connection loop ──
    def start(self) -> "Stream":
        if self.thread and self.thread.is_alive():
            return self
        self.thread = threading.Thread(target=self._run, name="td-stream", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _sync_subscriptions(self, s) -> None:
        want = set(self._markets())
        add, drop = want - self.subscribed, self.subscribed - want
        if drop:
            s.sendall(encode_frame(json.dumps({"action": "unsubscribe", "params": {"symbols": ",".join(sorted(drop))}}).encode()))
        if add:
            s.sendall(encode_frame(json.dumps({"action": "subscribe", "params": {"symbols": ",".join(sorted(add))}}).encode()))
        self.subscribed = want

    def _run(self) -> None:
        from . import twelvedata
        backoff = 2
        while not self._stop.is_set():
            key = twelvedata.credentials(self.root)
            if not key:
                self._set_state("off", "TWELVEDATA_API_KEY missing in .env")
                self._stop.wait(30)
                continue
            kinds = {m.get("kind", "metal") for m in self._markets().values()}
            if all(twelvedata.market_closed(kind=k) for k in kinds):
                self._set_state("closed", "markets closed")
                self._stop.wait(60)
                continue
            try:
                self._set_state("connecting")
                s, rest = ws_connect(HOST, f"{PATH}?apikey={urllib.parse.quote(key)}")
                self.subscribed = set()
                self._sync_subscriptions(s)
                s.settimeout(1.0)
                reader, hb = FrameReader(), time.time()
                frames = reader.feed(rest)
                backoff = 2
                connected_at = time.time()
                while not self._stop.is_set():
                    for op, payload in frames:
                        if op == 0x1:
                            self._on_message(payload)
                        elif op == 0x9:
                            s.sendall(encode_frame(payload, 0xA))
                        elif op == 0x8:
                            raise WSClosed("server closed the stream")
                    if self._resub.is_set():
                        self._resub.clear()
                        self._sync_subscriptions(s)
                    if time.time() - hb >= HEARTBEAT:
                        s.sendall(encode_frame(b'{"action":"heartbeat"}'))
                        hb = time.time()
                    quiet = time.time() - max(self.last_tick, connected_at)
                    if quiet > 120 and not all(twelvedata.market_closed(kind=k) for k in kinds):
                        raise WSClosed("no ticks for 2 minutes")
                    try:
                        data = s.recv(65536)
                    except socket.timeout:
                        frames = []
                        continue
                    if not data:
                        raise WSClosed("connection dropped")
                    frames = reader.feed(data)
                try:
                    s.close()
                except Exception:
                    pass
            except Exception as e:
                log.info("stream: %s — reconnecting in %ss", e, backoff)
                self._set_state("reconnecting", str(e)[:160])
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30)

    def _backfill_minutes(self) -> None:
        try:
            from . import twelvedata
            for td, m in self._markets().items():
                twelvedata.minute_bars(self.root, force=True, symbol=td, kind=m.get("kind", "metal"))
        except Exception as e:
            log.info("minute backfill after connect: %s", e)

    def _on_message(self, payload: bytes) -> None:
        try:
            msg = json.loads(payload)
        except Exception:
            return
        ev = msg.get("event")
        if ev == "price" and msg.get("price") is not None:
            if self.state != "live":
                self._set_state("live")
                # fill the minutes from before this connection with official bars (1 credit per market)
                threading.Thread(target=self._backfill_minutes, daemon=True).start()
            self.on_price(float(msg["price"]), td=msg.get("symbol", SYMBOL))
        elif ev == "subscribe-status":
            fails = msg.get("fails") or []
            if msg.get("status") != "ok" and not msg.get("success"):
                self._set_state("error", f"subscription refused: {json.dumps(msg)[:160]}")
            else:
                if fails:
                    self.message = "not streamed on this plan: " + ", ".join(f.get("symbol", "?") for f in fails)
                self._set_state("live" if self.last_tick and time.time() - self.last_tick < 60 else "subscribed", self.message)
        elif msg.get("status") == "error" or ev == "error":
            self._set_state("error", str(msg.get("message", msg))[:160])


_stream: Stream | None = None


def get(root: Path | None = None, start: bool = False) -> Stream | None:
    global _stream
    if _stream is None and root is not None:
        _stream = Stream(root)
    if start and _stream is not None:
        _stream.start()
    return _stream
