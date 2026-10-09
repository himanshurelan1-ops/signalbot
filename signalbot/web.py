"""Local web page: what the rules say for spot gold, and the record behind it."""
from __future__ import annotations

import json
import os
import queue
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import SYMBOLS, TIMEFRAMES, available_symbols
from .report import build

log = logging.getLogger("signalbot.web")

_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()
_key_locks: dict = {}
CACHE_SECONDS = 300


#: intraday pages go stale fast; daily can sit for a while
CACHE_BY_TF = {"1w": 900, "1d": 300, "4h": 300, "1h": 300, "30m": 120, "15m": 120, "5m": 60, "1m": 30}


def report_cached(symbol: str, root: Path, tf: str, long_only: bool, flat: bool | None,
                  force: bool = False, **tuning) -> dict:
    key = (symbol, tf, long_only, flat, tuple(sorted(tuning.items())))
    with _lock:
        lk = _key_locks.setdefault((symbol, tf), threading.Lock())
    with lk:                                     # per market+timeframe: BTC downloading never blocks gold
        hit = _cache.get(key)
        if hit and not force and time.time() - hit[0] < CACHE_BY_TF.get(tf, 300):
            return hit[1]
        rep = build(symbol, root, tf, long_only=long_only, flat_by_close=flat, **tuning)
        _cache[key] = (time.time(), rep)
        return rep


def symbols_meta() -> list[dict]:
    return [{"symbol": k, "name": v["name"], "short": v["short"], "kind": v["kind"], "td": v["td"]} for k, v in SYMBOLS.items()]


def popular_meta() -> list[dict]:
    from .symbols import POPULAR
    have = {v["td"] for v in SYMBOLS.values()}
    return [{"td": t, "name": n, "type": ty, "added": t in have} for t, n, ty in POPULAR]


def make_handler(root: Path, page: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):          # quiet
            log.debug(fmt, *a)

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass                                 # the page reloaded or closed mid-request

        def _stream(self):
            """Server-Sent Events: every tick from Twelve Data, the moment it arrives."""
            from . import stream
            st = stream.get(root, start=True)
            q = st.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 2000\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        line = f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        line = ": keep-alive\n\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                st.unsubscribe(q)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, page.encode(), "text/html")
            if u.path == "/api/stream":
                return self._stream()
            if u.path == "/api/live":
                from .live import forming_candle
                sym = (q.get("symbol") or ["XAUUSD"])[0].upper()
                tf = (q.get("tf") or ["1d"])[0]
                if sym not in SYMBOLS or tf not in TIMEFRAMES:
                    return self._send(404, {"error": "unknown symbol/timeframe"})
                try:
                    return self._send(200, forming_candle(root, sym, tf))
                except Exception as e:
                    return self._send(200, {"error": f"{type(e).__name__}: {e}"})
            if u.path == "/api/symbols":
                return self._send(200, {
                    "version": VERSION,
                    "symbols": symbols_meta(),
                    "popular": popular_meta(),
                    "timeframes": [{"tf": k, "label": v["label"], "short": v["short"]} for k, v in TIMEFRAMES.items()],
                })
            if u.path == "/api/search":
                from . import twelvedata
                try:
                    return self._send(200, {"results": twelvedata.search(root, (q.get("q") or [""])[0])})
                except Exception as e:
                    return self._send(200, {"results": [], "error": str(e)})
            if u.path == "/api/watchlist":
                from . import stream, symbols
                if q.get("add"):
                    item = json.loads(q["add"][0])
                    m = symbols.add(root, item["td"], item.get("name"), item.get("type"), item.get("exchange"))
                    added = m["key"]
                else:
                    added = None
                if q.get("remove"):
                    symbols.remove(root, q["remove"][0])
                st = stream.get()
                if st:
                    st.resubscribe()
                _cache.clear()
                return self._send(200, {"symbols": symbols_meta(), "added": added})
            if u.path == "/api/stream/status":
                from . import stream
                st = stream.get(root)
                return self._send(200, st.status_msg() | {"last_price": st.last_price, "last_tick": st.last_tick}
                                  if st else {"state": "off"})
            if u.path == "/api/report":
                sym = (q.get("symbol") or ["XAUUSD"])[0].upper()
                if sym not in SYMBOLS:
                    return self._send(404, {"error": f"unknown symbol {sym}"})
                tf = (q.get("tf") or ["1d"])[0]
                if tf not in TIMEFRAMES:
                    return self._send(404, {"error": f"unknown timeframe {tf}"})
                long_only = (q.get("long_only") or ["0"])[0] in ("1", "true")
                flat_q = (q.get("flat") or [""])[0]
                flat = None if flat_q == "" else flat_q in ("1", "true")
                force = (q.get("refresh") or ["0"])[0] in ("1", "true")
                tuning = {}
                for k, cast in (("stop_atr", float), ("target_atr", float), ("max_bars", int), ("cost", float)):
                    v = (q.get(k) or [""])[0]
                    if v:
                        try:
                            tuning[k] = cast(v)
                        except ValueError:
                            return self._send(400, {"error": f"{k} must be a number"})
                try:
                    return self._send(200, report_cached(sym, root, tf, long_only, flat, force, **tuning))
                except Exception as e:
                    log.exception("report failed")
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return self._send(404, {"error": "not found"})
    return H


def code_version() -> str:
    """Fingerprint of the code on disk, so a newer copy can recognise an older running one."""
    import hashlib
    h = hashlib.sha1()
    here = Path(__file__).parent
    for f in sorted(list(here.glob("*.py")) + [here / "web_ui.html"]):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


VERSION = code_version()


def _stop_old_server(port: int) -> bool:
    """Stop an older signalbot holding the port (macOS / Linux: lsof). True if the port came free."""
    import signal
    import socket
    import subprocess
    try:
        pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True,
                              text=True, timeout=5).stdout.split()
    except Exception:
        pids = []
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    for _ in range(25):
        time.sleep(0.2)
        with socket.socket() as sk:
            if sk.connect_ex(("127.0.0.1", port)) != 0:
                return True
    return False


def serve(root: Path, port: int = 8790, open_browser: bool = True) -> None:
    page = (Path(__file__).parent / "web_ui.html").read_text()
    from .symbols import load_watchlist
    load_watchlist(root)
    # Twelve Data's free plan allows one live stream per key: a second signalbot
    # server would knock the first one's stream off (and vice versa). If one is
    # already running, open it instead of starting another.
    # A running copy of THIS version is reused; an older version is stopped and replaced.
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/symbols", timeout=2) as r:
            running = json.loads(r.read())
        if any(s.get("symbol") == "XAUUSD" for s in running.get("symbols", [])):
            url = f"http://127.0.0.1:{port}"
            if running.get("version") == VERSION:
                print(f"\n  signalbot is already running at {url} — opening it (only one copy can hold the live stream)\n")
                if open_browser:
                    webbrowser.open(url)
                return
            print(f"\n  an older signalbot is running on port {port} — stopping it and starting this version…")
            if not _stop_old_server(port):
                raise SystemExit(f"  could not stop it — close its Terminal window (or run: pkill -f 'run.py serve') and try again")
    except SystemExit:
        raise
    except Exception:
        pass
    httpd = None
    for candidate in range(port, port + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), make_handler(root, page))
            httpd.daemon_threads = True          # open live streams must not block ctrl-c
            port = candidate
            break
        except OSError as e:
            if e.errno not in (48, 98):
                raise
    if httpd is None:
        raise SystemExit(f"ports {port}-{port + 9} busy — pkill -f 'run.py serve'")
    from . import stream
    stream.get(root, start=True)                 # live ticks from Twelve Data, from the start
    url = f"http://127.0.0.1:{port}"
    print(f"\n  signalbot is running at  {url}\n  (ctrl-c to stop)\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
