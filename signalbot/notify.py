"""Telegram alerts for new signals, with a memory of what was already sent."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("signalbot.notify")


def load_env(root: Path) -> dict[str, str]:
    env = {}
    p = root / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("TELEGRAM_")})
    return env


def send(root: Path, text: str) -> bool:
    env = load_env(root)
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing — see README, 'Telegram'")
    data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                   "disable_web_page_preview": "true"}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    with urllib.request.urlopen(req, timeout=20) as r:
        ok = json.loads(r.read()).get("ok", False)
    return bool(ok)


def whoami(root: Path) -> list[dict]:
    """Chats that have messaged the bot — to find your chat id."""
    env = load_env(root)
    token = env.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing in .env")
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20) as r:
        upd = json.loads(r.read()).get("result", [])
    seen = {}
    for u in upd:
        m = u.get("message") or u.get("channel_post") or {}
        c = m.get("chat")
        if c:
            seen[c["id"]] = {"id": c["id"], "name": c.get("title") or c.get("username")
                             or f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()}
    return list(seen.values())


# ── de-duplication ──────────────────────────────────────────────────────────

def _sent_path(root: Path) -> Path:
    return root / "state" / "sent.json"


def already_sent(root: Path, key: str) -> bool:
    try:
        return key in json.loads(_sent_path(root).read_text())
    except Exception:
        return False


def mark_sent(root: Path, key: str) -> None:
    p = _sent_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = {}
    data[key] = time.time()
    # keep it from growing forever
    cutoff = time.time() - 60 * 86400
    data = {k: v for k, v in data.items() if v > cutoff}
    p.write_text(json.dumps(data, indent=1))


def signal_keys(rep: dict) -> list[tuple[str, str]]:
    """(dedupe-key, message) for every signal worth telling the user about.

    Keyed by index + timeframe + setup + side + the bar it fired on, so each
    signal is sent once and a live trade is not re-sent every run.
    """
    from .report import headline
    active = [s for s in rep["setups"] if s["signal"]["status"] in ("signal", "open")]
    out = []
    for s, line in zip(active, headline(rep)):
        sig = s["signal"]
        when = sig["trade"]["signal_time"] if sig.get("trade") else sig["as_of"]
        out.append((f"{rep['symbol']}|{rep['tf']}|{sig['setup']}|{sig['side']}|{when}", line))
    return out
