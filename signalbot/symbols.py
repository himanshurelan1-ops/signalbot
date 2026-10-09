"""The markets signalbot follows: gold by default, plus anything added from the page.

Each market is a Twelve Data symbol ("XAU/USD", "EUR/USD", "BTC/USD", "AAPL") with
a short key used for tabs and cache files ("XAUUSD", "EURUSD", "BTCUSD", "AAPL")
and a *kind* that decides trading hours, spread, volume source and news currencies.

The watchlist lives in state/watchlist.json; on Streamlit Cloud (whose disk is
reset on restart) the WATCHLIST secret ("XAU/USD, EUR/USD, BTC/USD") seeds it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: Twelve Data instrument_type → kind
KINDS = {"Precious Metal": "metal", "Physical Currency": "fx", "Digital Currency": "crypto",
         "Common Stock": "stock", "ETF": "stock", "Depositary Receipt": "stock", "REIT": "stock",
         "Commodity": "commodity", "Index": "index"}

#: default round-trip spread as a fraction of price (gold's $0.30 at ~$4,200 ≈ 0.7 bp)
SPREAD_BP = {"metal": 0.7, "fx": 1.0, "crypto": 2.0, "stock": 2.0, "commodity": 3.0, "index": 1.0}

#: Yahoo futures whose volume stands in for spot metals (spot has no exchange volume)
FUTURES_VOLUME = {"XAU/USD": "GC=F", "XAG/USD": "SI=F", "XPT/USD": "PL=F", "XPD/USD": "PA=F"}

#: one-click list on the page (no API credits); all on Twelve Data's free plan
POPULAR = [
    ("XAU/USD", "Gold spot", "Precious Metal"), ("XAG/USD", "Silver spot", "Precious Metal"),
    ("XPT/USD", "Platinum spot", "Precious Metal"),
    ("EUR/USD", "Euro / US Dollar", "Physical Currency"), ("GBP/USD", "British Pound / US Dollar", "Physical Currency"),
    ("USD/JPY", "US Dollar / Japanese Yen", "Physical Currency"), ("AUD/USD", "Australian Dollar / US Dollar", "Physical Currency"),
    ("USD/CAD", "US Dollar / Canadian Dollar", "Physical Currency"), ("USD/CHF", "US Dollar / Swiss Franc", "Physical Currency"),
    ("GBP/JPY", "British Pound / Japanese Yen", "Physical Currency"), ("USD/INR", "US Dollar / Indian Rupee", "Physical Currency"),
    ("BTC/USD", "Bitcoin", "Digital Currency"), ("ETH/USD", "Ethereum", "Digital Currency"),
    ("SOL/USD", "Solana", "Digital Currency"),
    ("SPY", "SPDR S&P 500 ETF", "ETF"), ("QQQ", "Invesco QQQ (Nasdaq 100)", "ETF"),
    ("AAPL", "Apple", "Common Stock"), ("NVDA", "NVIDIA", "Common Stock"), ("TSLA", "Tesla", "Common Stock"),
]

DEFAULT = "XAU/USD"


def key_of(td: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", td.upper())


def make(td: str, name: str | None = None, itype: str | None = None, exchange: str | None = None) -> dict:
    """Everything the rest of the code needs to know about one market."""
    td = td.strip().upper()
    pop = next((p for p in POPULAR if p[0] == td), None)
    if itype is None:
        itype = pop[2] if pop else ("Physical Currency" if re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", td) else "Common Stock")
    kind = KINDS.get(itype, "stock")
    if td.startswith(("XAU/", "XAG/", "XPT/", "XPD/")):
        kind = "metal"
    base, _, quote = td.partition("/")
    if kind == "fx":
        news = sorted({base, quote} & {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY"}) + ["All"]
    else:
        news = ["USD", "All"]
    nm = name or (pop[1] if pop else td)
    return {"td": td, "key": key_of(td), "name": f"{nm} ({td})" if td not in nm else nm, "short": td,
            "kind": kind, "type": itype, "exchange": exchange or "",
            "session": "24x7" if kind == "crypto" else ("exchange" if kind == "stock" else "24h"),
            "news": news, "unit": quote if kind == "fx" else "$"}


#: live registry (mutated in place so every module's import sees changes)
SYMBOLS: dict[str, dict] = {}
SYMBOLS[key_of(DEFAULT)] = make(DEFAULT, "Gold spot")


def _path(root: Path) -> Path:
    return root / "state" / "watchlist.json"


def load_watchlist(root: Path) -> dict:
    """Fill SYMBOLS from state/watchlist.json (and the WATCHLIST env/secret on first use)."""
    items: list[dict] = []
    p = _path(root)
    if p.exists():
        try:
            items = json.loads(p.read_text())
        except Exception:
            items = []
    if not items and os.environ.get("WATCHLIST"):
        items = [{"td": t.strip()} for t in os.environ["WATCHLIST"].split(",") if t.strip()]
    new = {key_of(DEFAULT): SYMBOLS.get(key_of(DEFAULT)) or make(DEFAULT, "Gold spot")}
    for it in items:
        try:
            m = make(it["td"], it.get("name"), it.get("type"), it.get("exchange"))
            new[m["key"]] = m
        except Exception:
            continue
    SYMBOLS.clear()
    SYMBOLS.update(new)
    return SYMBOLS


def _save(root: Path) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"td": m["td"], "name": m["name"].split(" (")[0], "type": m["type"],
                              "exchange": m["exchange"]} for m in SYMBOLS.values()], indent=1))


def add(root: Path, td: str, name: str | None = None, itype: str | None = None, exchange: str | None = None) -> dict:
    m = make(td, name, itype, exchange)
    SYMBOLS[m["key"]] = m
    _save(root)
    return m


def remove(root: Path, key: str) -> None:
    if key in SYMBOLS and len(SYMBOLS) > 1:
        del SYMBOLS[key]
        _save(root)


def default_cost(meta: dict, price: float) -> float:
    """Round-trip spread in price units: gold keeps its $0.30, everything else scales with price."""
    if meta["td"] == "XAU/USD":
        return 0.30
    bp = 5.0 if meta["kind"] == "metal" else SPREAD_BP.get(meta["kind"], 2.0)   # silver/platinum trade wider than gold
    return round(price * bp / 1e4, 8)


def decimals(price: float) -> int:
    p = abs(price)
    return 5 if p < 2 else 4 if p < 20 else 3 if p < 200 else 2


def pf(x) -> str:
    """Format a price with sensible decimals for its size (1.08423, 154.321, 4,194.21)."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{x:,.{decimals(x)}f}"
