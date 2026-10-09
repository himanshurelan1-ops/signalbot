"""signalbot on Streamlit — the same page as `run.py serve`, reachable from anywhere.

Python (this file) replays the rules and hands each result to the page through
a Streamlit component; the page keeps its own live connection to Twelve Data
for tick-by-tick prices and asks for fresh results the moment a candle closes.

Secrets (Streamlit Cloud → app → Settings → Secrets):
    TWELVEDATA_API_KEY = "…"
    APP_PASSWORD = "…"            # anyone with the link needs this
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="signalbot", page_icon="🟡", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
  #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"]{display:none !important}
  .block-container{padding:0.4rem 0.6rem 1rem !important; max-width:1240px}
  iframe{border:0}
</style>""", unsafe_allow_html=True)


# ── secrets → environment (the library reads TWELVEDATA_* from the environment) ──
def _secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)


for k in ("TWELVEDATA_API_KEY", "TWELVEDATA_DAILY_LIMIT", "TWELVEDATA_REFRESH_SECONDS", "WATCHLIST",
          "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
    v = _secret(k)
    if v:
        os.environ[k] = str(v)

# ── password gate ──
pw = _secret("APP_PASSWORD")
if pw:
    if not st.session_state.get("authed"):
        st.markdown("### 🔒 signalbot")
        entered = st.text_input("Password", type="password")
        if entered and entered == str(pw):
            st.session_state["authed"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password")
        st.stop()
else:
    st.warning("No APP_PASSWORD secret set — anyone with this link can open the app and see your Twelve Data key. "
               "Add APP_PASSWORD in the app's Secrets.")

if not os.environ.get("TWELVEDATA_API_KEY"):
    st.error("Add TWELVEDATA_API_KEY to the app's Secrets (Streamlit Cloud → your app → Settings → Secrets).")
    st.stop()

from signalbot import symbols as SY  # noqa: E402
from signalbot.data import SYMBOLS, TIMEFRAMES  # noqa: E402
from signalbot.report import build  # noqa: E402

if not st.session_state.get("watchlist_loaded"):
    SY.load_watchlist(ROOT)
    st.session_state["watchlist_loaded"] = True

# ── the page, served as a Streamlit component ──
COMP_DIR = ROOT / ".component"
COMP_DIR.mkdir(exist_ok=True)
shutil.copyfile(ROOT / "signalbot" / "web_ui.html", COMP_DIR / "index.html")
signalbot_page = components.declare_component("signalbot_page", path=str(COMP_DIR))

# ── rule settings ──
with st.expander("⚙️ Rule settings", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    long_only = c1.toggle("Long only", value=False)
    stop_atr = c2.number_input("Stop × ATR", 0.5, 10.0, 2.0, 0.5)
    target_atr = c3.number_input("Target × ATR", 0.5, 20.0, 3.0, 0.5)
    max_bars = c4.number_input("Time stop (bars)", 1, 500, 20, 1)
    cost = c5.number_input("Spread (price units, 0 = auto)", 0.0, 1000.0, 0.0, 0.01, format="%.5f",
                           help="Round-trip spread + commission taken off every backtested trade. 0 = the market's "
                                "default: $0.30 on gold, about 1 pip on forex, 0.02% on crypto, 0.02% on stocks.")

# what the page asked for last: a timeframe / market, a refresh after a candle closed,
# a symbol search, or adding / removing a market
ev = st.session_state.get("sb") or {}
search = st.session_state.get("search_result")
if ev.get("nonce") and ev.get("nonce") != st.session_state.get("handled"):
    st.session_state["handled"] = ev["nonce"]
    if ev.get("search"):
        from signalbot import twelvedata
        try:
            res = twelvedata.search(ROOT, ev["search"])
        except Exception as e:
            res = []
            st.toast(f"search failed: {e}")
        search = st.session_state["search_result"] = {"nonce": ev["nonce"], "results": res}
    if ev.get("add"):
        a = ev["add"]
        m = SY.add(ROOT, a["td"], a.get("name"), a.get("type"), a.get("exchange"))
        st.session_state["symbol"] = m["key"]
    if ev.get("remove"):
        SY.remove(ROOT, ev["remove"])
        if st.session_state.get("symbol") == ev["remove"]:
            st.session_state["symbol"] = next(iter(SYMBOLS))
    if ev.get("symbol") in SYMBOLS:
        st.session_state["symbol"] = ev["symbol"]
    if ev.get("tf") in TIMEFRAMES:
        st.session_state["tf"] = ev["tf"]
tf = st.session_state.get("tf", "5m")
symbol = st.session_state.get("symbol") if st.session_state.get("symbol") in SYMBOLS else next(iter(SYMBOLS))
st.session_state["symbol"] = symbol


def _bucket(tf: str) -> str:
    """Start of the candle that is forming now — the cache key changes the moment a candle closes."""
    now = pd.Timestamp.now(tz="UTC")
    if tf == "1d":
        return str(now.date())
    step = {"1h": "1h", "15m": "15min", "5m": "5min"}[tf]
    return str(now.floor(step))


@st.cache_resource
def _store():
    return {"lock": threading.Lock(), "reports": {}}


def _clean(o):
    """JSON-safe: NaN / inf → None, numpy → python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def report_for(symbol: str, tf: str) -> dict:
    key = (symbol, tf, long_only, stop_atr, target_atr, max_bars, cost, _bucket(tf))
    store = _store()
    with store["lock"]:
        if key in store["reports"]:
            return store["reports"][key]
        rep = build(symbol, ROOT, tf, long_only=long_only, stop_atr=stop_atr, target_atr=target_atr,
                    max_bars=int(max_bars), cost=cost)
        try:
            from signalbot.live import forming_candle
            rep["forming"] = forming_candle(ROOT, symbol, tf)
        except Exception:
            rep["forming"] = None
        rep = _clean(json.loads(json.dumps(rep, default=str)))
        store["reports"] = {k: v for k, v in store["reports"].items() if k[-1] == key[-1] or k[:2] != key[:2]}
        store["reports"][key] = rep
        return rep


try:
    with st.spinner(f"Replaying the rules on the latest {SYMBOLS[symbol]['short']} candles…"):
        rep = report_for(symbol, tf)
except Exception as e:
    st.error(f"Could not build the report: {type(e).__name__}: {e}")
    st.stop()

have = {v["td"] for v in SYMBOLS.values()}
meta = {"symbols": [{"symbol": k, "name": v["name"], "short": v["short"], "kind": v["kind"], "td": v["td"]} for k, v in SYMBOLS.items()],
        "popular": [{"td": t, "name": n, "type": ty, "added": t in have} for t, n, ty in SY.POPULAR],
        "timeframes": [{"tf": k, "label": v["label"]} for k, v in TIMEFRAMES.items()]}
signalbot_page(report=rep, meta=meta, search=search, td_key=os.environ["TWELVEDATA_API_KEY"], key="sb", default=None, height=900)
