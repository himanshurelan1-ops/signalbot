# signalbot

What the rules say right now for **spot gold (XAU/USD)**, the same price
TradingView shows as `OANDA:XAUUSD` give or take the spread, on **1-minute, 5-minute, 15-minute, 30-minute, hourly, 4-hour, daily and weekly**
candles. Each rule gives BUY / SELL / NO TRADE with *from → to → stop* levels,
next to how those exact rules have done across every bar on record. It runs as a
local web page, a CLI, and optional Telegram alerts.

(The Nifty / Bank Nifty / COMEX / crude version, and the OANDA one, are kept in
`_backup_with_indices_2026-10-09/`.)

## 1. Twelve Data API key (required, free, open to India)

1. Sign up at **twelvedata.com** (free Basic plan, no card needed).
2. Copy your **API key** from the dashboard.
3. In `.env`: `TWELVEDATA_API_KEY=…`
4. Check it with `.venv/bin/python run.py td-test`. It prints the latest XAU/USD price.

The first start backfills history: about 170 days of 5-minute candles, 1 year of 15-minute, 3 years of hourly and 20 years of daily. That takes about 25 credits and a few minutes, because the free plan allows 8 requests a minute.

**Live ticks.** While the page is open, signalbot keeps one connection to Twelve Data's live price stream (included in the free plan). Every tick (gold updates about every 1–3 seconds) is pushed to the page within milliseconds, so the price and the forming candle move with each one. When a candle closes, the rules re-run on it within a few seconds, without you clicking anything. The **⛶ Full screen** button (or the F key) blows the chart up to fill the screen; Esc exits. The free plan allows one live stream per API key, so run one copy of signalbot at a time (starting it again just opens the running one).

**Volume.** Spot gold has no exchange volume (and Twelve Data sends none), so the chart's volume pane shows two things. The bars are COMEX gold futures contracts per candle (from Yahoo, free, about 10 minutes delayed), drawn solid when a candle trades at least 1.5× its 20-candle average. The blue line is live tick volume: price updates per candle, counted from the stream as they arrive (what TradingView shows as volume on OANDA:XAUUSD). The **volume breakout** rule is the trend breakout taken only on a ≥1.5× volume candle, backtested on COMEX volume; when COMEX hasn't printed the newest candle yet, the live signal uses tick volume against its own average. COMEX intraday history is about 60 days from Yahoo and grows in the local cache the longer signalbot runs.

**Credits.** The free plan is about 800 credits a day (one per REST request; the live stream doesn't use them). The stream also builds the 1-minute bars, so with the page open signalbot only pulls official 1-minute bars every 10 minutes, plus the daily candle once an hour: roughly 170 credits a day. Without the page (the Telegram watcher on its own), it pulls every `TWELVEDATA_REFRESH_SECONDS` (default 150), about 600 a day. It pauses over the gold weekend. The page shows today's count; if the allowance runs out, it keeps working on cached candles until 05:30 IST.

## 2. Run it

```bash
cd ~/signalbot
./install.sh                      # once
.venv/bin/python run.py serve     # opens http://127.0.0.1:8790
```

Or double-click **Open signalbot.command**.

## The rules

| card | rule |
|---|---|
| **breakout** | In an uptrend (close above the 200-bar EMA), buy a close above the previous 20 bars' high. Mirror for SELL in a downtrend. |
| **pullback** | In an uptrend, buy when price dips below the 20-bar EMA and closes back above it with RSI > 50. Mirror for SELL. |
| **volume breakout** | The breakout rule, but only when the breakout candle's volume is at least 1.5× its 20-candle average. |
| **ema 9/21 cross** | The 9 EMA crossing the 21 EMA in the direction of the 200-EMA trend. |
| **macd cross** | MACD (12, 26, 9) crossing its signal line with the trend. |
| **rsi bounce** | In an uptrend, RSI(14) dips under 30 and closes back above (mirror over 70 in a downtrend). |
| **bollinger squeeze** | Bands at their tightest 20% of the last 120 candles, then a close outside them with the trend. |
| **supertrend flip** | Supertrend (10, 3) flipping in the direction of the trend. |
| **vwap reclaim** | A close back above the day's VWAP (resets 05:30 IST) after a close below it, with the trend. |
| **london breakout** | First close beyond the Asian range (05:30–12:30 IST) during 12:30–15:30 IST; one trade a day. |
| **liquidity sweep** | A wick through the latest swing high (low) that closes back inside: stops got run, trade the reversal. Each level once. |
| **pdh/pdl sweep** | The same at the previous day's high / low; first sweep of each per day. |
| **asian range sweep** | London (12:30–15:30 IST) wicks beyond the Asian range and closes back inside (the "Judas swing"). |
| **fvg retest** | With the trend, price dips into an unfilled fair value gap and closes back out of it on a candle in the trend's colour. |
| **order block** | A displacement candle (body ≥ 1.5 ATR) marks the last opposite candle as an order block; the first retest that holds, with the trend. |
| **choch** | Change of character: after lower highs, the first close above the last swing high (and mirror). |

The **Liquidity / SMC** switch on the chart draws the previous day's high/low, equal highs/lows, unfilled fair value gaps, active order blocks, recent sweeps and the last BOS/CHoCH; the *Liquidity & structure* card lists the nearest liquidity above and below price.

**Spread.** Every backtested trade pays a round-trip spread (default $0.30, editable in the toolbar), because scalping results without costs flatter every rule.

**Timeframes.** 1m, 5m, 15m, 1H and 1D come from Twelve Data. 30m, 4H and 1W are built from 15m, 1H and 1D candles (4H on the UTC grid like TradingView, weeks start Monday), so they cost no extra credits. The 1-minute chart uses the shared 1-minute cache, deepened once a day to about a week of candles (2 credits). On Streamlit Cloud, watching 1m pulls the newest minute about once a minute (1 credit each), so leave it on 5m or higher when you are not scalping.

**Market hours only.** Twelve Data keeps publishing XAU/USD candles over the weekend and in the daily 17:00–18:00 New York break: synthetic, flat prices with about $0.25 of jitter. They were about 30% of the 5-minute history, so signalbot drops every candle from when gold isn't trading (Fri 17:00 → Sun 18:00 New York, plus the daily break).

**Layout.** The chart is at the top, with all eight timeframe buttons on it. The verdict banner, rule settings, scoreboard and cards are below.

**Scoreboard.** Below the chart, every rule is ranked by its average result per trade after the spread, with its last-30-days record next to it, so you can see which edges are still working. The banner under the chart also shows the 200 EMA trend on 1D, 1H, 15m and 5m, how many rules agree right now, the trading session, and a warning when a signal goes against the higher-timeframe trend.
| **trendline break** | A close through a trendline that price had respected, traded in the direction of the break. |
| **channel bounce** | Inside a parallel channel, a bar that touches one line and closes back inside, traded toward the other line. |

Stop = 2 × ATR(14) from entry, target = 3 × ATR (1.5 : 1), time stop after 20 bars. All three are editable at the top of the page. Gold trades round the clock, so positions carry overnight and there is no session square-off.

Each card shows the signal (*new* = fired on the latest close, so enter at the next open; *live* = an earlier signal still running), entry / target / stop, the record (trades, % won, average R, profit factor, total R, worst drawdown, and the BUY and SELL sides separately), the equity curve, the level that would trigger next, and the last 12 trades.

R = multiples of the initial risk (entry → stop). A trade that reaches its target is +1.5R; a stop is −1R.

## Any market

Click **+ Add market** above the chart. The popular ones are one click (metals, major forex pairs, BTC, ETH, SOL, SPY, QQQ and a few big US stocks), or search anything Twelve Data's free plan streams: forex, metals, crypto, US stocks and ETFs (search costs 1 credit). Each market gets its own tab with the same rules, scoreboard, banner and live ticks. Remove one with the × on its tab.

What changes per market:

- **Spread:** $0.30 on gold, about 1 pip on forex, 0.02% on crypto, 0.02% on stocks; override it in the settings.
- **Hours:** crypto 24/7, US stocks 09:30–16:00 New York, forex and metals Sunday to Friday.
- **Volume:** exchange volume for stocks and crypto, futures volume for gold, silver, platinum and palladium, tick volume for forex.
- **News:** the pair's own currencies.

Every new market downloads its history once (about 25 credits, a couple of minutes on the free plan's 8-a-minute limit). The free plan's live stream covers up to about 8 markets at once.

## Use it from anywhere (Streamlit Cloud)

`streamlit_app.py` runs the same page on Streamlit Community Cloud (free), password-protected, phone-friendly.

1. **Put the code on GitHub** (from this folder, on your Mac). The `.gitignore` keeps your `.env` key, price data and backups out:
   ```bash
   brew install gh            # once, if you don't have it
   gh auth login              # once: GitHub.com → HTTPS → log in with a web browser
   cd ~/signalbot
   git init -b main
   git add .
   git commit -m "signalbot"
   gh repo create signalbot --private --source . --push
   ```
2. **Deploy:** go to **share.streamlit.io** → sign in with GitHub → **Create app** → *Deploy a public app from GitHub* → repository `signalbot`, branch `main`, main file `streamlit_app.py` → **Advanced settings**: Python 3.12, and paste into **Secrets**:
   ```toml
   TWELVEDATA_API_KEY = "your-key"
   APP_PASSWORD = "choose-a-password"
   WATCHLIST = "XAU/USD, EUR/USD"     # optional: the markets it starts with
   ```
   → **Deploy**. You get a link like `https://your-app.streamlit.app` that works on any phone or computer.
3. **Updates:** after changing code on your Mac, `git add . && git commit -m "update" && git push`; the app redeploys by itself.

Good to know:

- **One live stream per key.** The free plan gives one live stream per API key. If the Mac app and the cloud app are open at the same time, they take turns knocking each other's live ticks off. Use one at a time, or get a second free key for the cloud.
- **Sleep and restarts.** Streamlit's free tier puts apps to sleep after a while without visitors; opening the link wakes it in under a minute. Its disk is wiped on restart, so history is downloaded again then (about 25 credits per market), and markets added on the page are forgotten unless they're in the `WATCHLIST` secret.
- **Password.** The password stops strangers using your Twelve Data key, which the page needs for the live ticks.
- **Telegram alerts** run from the Mac (`./install.sh start`), not from Streamlit.

## Patterns, trendlines, news

- **Chart patterns** (double top/bottom, head & shoulders, triangles, wedges, channels, rectangles) and **candlestick patterns** are detected mechanically. Each one is shown with how often it actually went the textbook way on gold at that timeframe (1.5 ATR in its direction before 1.5 ATR against).
- **News windows:** the ForexFactory calendar (USD and All events) marks a no-trade window from 15 minutes before to 30 minutes after every High-impact release (NFP, CPI, FOMC and so on). Live or new signals inside a window get a red warning.
- **Live chart:** the forming candle moves with every tick from the live stream. The rules only ever see closed candles.

## CLI

```bash
.venv/bin/python run.py scan --tf 5m 15m            # signals + levels now
.venv/bin/python run.py backtest --tf 1d 1h 15m 5m  # the statistics table
.venv/bin/python run.py scan --tf 15m --long-only
```

## Telegram alerts

1. In Telegram, message **@BotFather** → `/newbot` → copy the token into `.env` as `TELEGRAM_BOT_TOKEN=…`.
2. Open your bot, press **Start**, and send it any message.
3. Run `.venv/bin/python run.py telegram-whoami` and copy the `TELEGRAM_CHAT_ID=…` line into `.env`.
4. Run `.venv/bin/python run.py telegram-test` to get a test message.
5. `./install.sh start` starts a background check of all timeframes every 5 minutes; each new signal is sent once. `./install.sh stop` removes it.

## How the backtest is kept honest, and what it leaves out

- Entries fill at the **next bar's open**. If one bar touches both the stop and the target, the stop counts first. Gaps fill at the open.
- No overlapping positions per rule. Parameters are round numbers and were not tuned to the history.
- **Spread is not modelled:** prices are mid. On 5-minute gold the stop is about $7 away, so a $0.50 round-trip spread costs about 0.07R per trade, which is roughly the whole edge of the best 5-minute rule. The 15-minute and daily timeframes are much less affected.

A BUY means the rule fired. The win rate beside it is how often the same situation worked out before, not a forecast. Not financial advice.

## Layout

```
run.py                    CLI
streamlit_app.py          the same page on Streamlit Cloud (password, any device)
signalbot/data.py         candle loader + on-disk cache (data/)
signalbot/twelvedata.py   Twelve Data API, credit ledger, 1-minute → 5m/15m/1h
signalbot/live.py         the forming candle, from 1-minute bars + the minute in progress
signalbot/stream.py       Twelve Data live tick stream (WebSocket) → page (SSE) + 1-minute bars with tick volume
signalbot/volume.py       COMEX futures volume (Yahoo) + tick volume, relative volume
signalbot/symbols.py      the market list, kinds, spreads, price decimals
signalbot/smc.py          liquidity sweeps, FVGs, order blocks, market structure (BOS/CHoCH)
signalbot/scalping.py     EMA cross, MACD, RSI, Bollinger squeeze, Supertrend, VWAP, London breakout
signalbot/news.py         ForexFactory calendar → events + no-trade windows
signalbot/patterns.py     swing pivots, candlestick + chart patterns, each with its record
signalbot/lines.py        trendlines / channels + the break and bounce rules
signalbot/indicators.py   EMA, ATR, RSI, Donchian
signalbot/strategy.py     setups, bar-by-bar backtest, stats, current signal
signalbot/report.py       everything the page / alerts need
signalbot/web.py + web_ui.html   local page
signalbot/notify.py       Telegram + de-duplication (state/sent.json)
tests/                    56 tests, synthetic data, no network
```
