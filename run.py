#!/usr/bin/env python3
"""signalbot — what the rules say for spot gold (XAU/USD, Twelve Data), and their full record.

  python run.py serve          open the page (the normal way to use it)
  python run.py scan --tf 1d 1h 15m 5m     signals and levels, any timeframes
  python run.py backtest --tf 1d 1h 15m 5m the statistics table
  python run.py watch --tf 15m 5m          check every 5 minutes, Telegram on new signals
  python run.py td-test                    check the Twelve Data API key in .env
  python run.py telegram-test  send a test message
  python run.py telegram-whoami  list chat ids that have messaged your bot
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from signalbot.data import SYMBOLS, TIMEFRAMES  # noqa: E402
from signalbot.symbols import load_watchlist  # noqa: E402

load_watchlist(HERE)                       # gold + the markets added on the page
from signalbot.symbols import pf  # noqa: E402


def cmd_scan(args) -> int:
    from signalbot.report import build, headline
    for sym in args.symbols:
        for tf in args.tf:
            rep = build(sym, HERE, tf, long_only=args.long_only)
            print(f"\n{rep['name']} · {rep['tf_label']}  close {pf(rep['last_close'])} ({rep['as_of']})  "
                  f"trend {rep['trend']}  ATR {pf(rep['atr'])}")
            nw = rep.get("news") or {}
            if nw.get("in_blackout"):
                print(f"  ⚠ {nw['blackout_text']}")
            elif nw.get("next_text"):
                print(f"  news: {nw['next_text']}")
            for s in rep["setups"]:
                sig, st, w = s["signal"], s["stats"], s["signal"]["watch"]
                tag = f"{sig['side']} ({sig['status']})" if sig["side"] else "no trade"
                print(f"  {s['setup']:<16} {tag}" + ("" if sig["side"] else f" — {sig['note']}"))
                if sig["entry"] is not None:
                    print(f"            from {pf(sig['entry'])} → to {pf(sig['target'])}, stop {pf(sig['stop'])}  "
                          f"[{st['trades']} trades, {st['win_rate']:.0%} won, avg {st['avg_r']:+.2f}R]")
                if w.get("trigger") is not None:
                    print(f"            watching: {w['side']} on {w['rule']} → from {pf(w['trigger'])} "
                          f"to {pf(w['target'])}, stop {pf(w['stop'])}")
            pats, tl = rep.get("patterns") or {}, rep.get("trendlines") or {}
            f = pats.get("forming")
            if f:
                st = f["stats"]
                rate = (f"{st['worked']}/{st['worked'] + st['failed']} worked ({st['work_rate']:.0%})"
                        if st.get("work_rate") is not None else "no textbook direction")
                print(f"  pattern   {f['name']} ({'bullish' if f['direction'] > 0 else 'bearish' if f['direction'] < 0 else 'neutral'}) — "
                      f"{f['note']}; record here: {rate}")
            for c in pats.get("recent_candles", []):
                if c["bar"] == 0 and c.get("stats"):
                    st = c["stats"]
                    print(f"  candle    {c['name']} — record here: {st['worked']}/{st['worked'] + st['failed']} worked "
                          f"({(st['work_rate'] or 0):.0%})")
            for ln in tl.get("lines", []):
                print(f"  line      {ln['kind']} through {ln['touches']} pivots, now ≈ {pf(ln['now'])}")
            if not headline(rep):
                print("  → nothing fires right now; the levels above are what would.")
    return 0


def cmd_backtest(args) -> int:
    from signalbot.report import build
    print(f"{'index':<11}{'tf':<7}{'setup':<17}{'trades':>7}{'won':>6}{'target':>8}{'avg R':>8}{'PF':>7}"
          f"{'total R':>9}{'max DD':>8}{'buys':>12}{'sells':>12}")
    for sym in args.symbols:
        for tf in args.tf:
            rep = build(sym, HERE, tf, long_only=args.long_only, refresh=not args.offline)
            for s in rep["setups"]:
                st = s["stats"]
                pf = "∞" if st["profit_factor"] is None else f"{st['profit_factor']:.2f}"
                buys = f"{st['buys']} ({st['buy_win_rate']:.0%})"
                sells = f"{st['sells']} ({st['sell_win_rate']:.0%})"
                print(f"{rep['name'][:10]:<11}{tf:<7}{s['setup']:<17}{st['trades']:>7}{st['win_rate']:>6.0%}{st['target_rate']:>8.0%}"
                      f"{st['avg_r']:>+8.2f}{pf:>7}{st['total_r']:>+9.1f}{st['max_dd_r']:>8.1f}{buys:>12}{sells:>12}")
    print("\nR = multiples of the initial risk (entry→stop). PF = gross wins / gross losses.")
    return 0


def cmd_serve(args) -> int:
    from signalbot.web import serve
    serve(HERE, port=args.port, open_browser=not args.no_open)
    return 0


def cmd_watch(args) -> int:
    from signalbot.notify import already_sent, mark_sent, send, signal_keys
    from signalbot.report import build
    log = logging.getLogger("signalbot.watch")
    while True:
        for sym in args.symbols:
            for tf in args.tf:
                try:
                    rep = build(sym, HERE, tf, long_only=args.long_only)
                except Exception as e:
                    log.warning("%s %s: %s", sym, tf, e)
                    continue
                for key, msg in signal_keys(rep):
                    if already_sent(HERE, key):
                        continue
                    try:
                        send(HERE, msg)
                        mark_sent(HERE, key)
                        log.info("sent: %s", msg)
                    except Exception as e:
                        log.warning("telegram: %s", e)
        if args.once:
            return 0
        time.sleep(args.every * 60)


def cmd_td_test(args) -> int:
    from signalbot import twelvedata
    r = twelvedata.test_key(HERE)
    print(("✓ " if r["ok"] else "✗ ") + r["message"])
    return 0 if r["ok"] else 1


def cmd_telegram_test(args) -> int:
    from signalbot.notify import send
    ok = send(HERE, "signalbot: test message — alerts are working.")
    print("sent" if ok else "Telegram said no")
    return 0 if ok else 1


def cmd_telegram_whoami(args) -> int:
    from signalbot.notify import whoami
    chats = whoami(HERE)
    if not chats:
        print("No chats yet — open your bot in Telegram, press Start, send it any message, then re-run.")
        return 1
    for c in chats:
        print(f"  TELEGRAM_CHAT_ID={c['id']}   ({c['name']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", nargs="*", default=list(SYMBOLS), choices=list(SYMBOLS))
    common.add_argument("--long-only", action="store_true", help="ignore SELL signals")
    common.add_argument("--tf", nargs="*", default=["1d"], choices=list(TIMEFRAMES),
                        help="timeframes: 1d 1h 15m 5m (default: 1d)")

    sub.add_parser("scan", parents=[common], help="today's signals and levels")
    bt = sub.add_parser("backtest", parents=[common], help="full statistics")
    bt.add_argument("--offline", action="store_true", help="use cached data only")
    sv = sub.add_parser("serve", help="open the web page")
    sv.add_argument("--port", type=int, default=8790)
    sv.add_argument("--no-open", action="store_true")
    w = sub.add_parser("watch", parents=[common], help="Telegram on new signals")
    w.add_argument("--every", type=int, default=5, help="minutes between checks")
    w.add_argument("--once", action="store_true")
    sub.add_parser("td-test", help="check the Twelve Data API key in .env")
    sub.add_parser("telegram-test")
    sub.add_parser("telegram-whoami")
    return p


COMMANDS = {"scan": cmd_scan, "backtest": cmd_backtest, "serve": cmd_serve, "watch": cmd_watch,
            "td-test": cmd_td_test,
            "telegram-test": cmd_telegram_test, "telegram-whoami": cmd_telegram_whoami}


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
