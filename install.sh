#!/bin/bash
# signalbot installer / launcher helper for macOS.
#   ./install.sh          create the venv and install dependencies
#   ./install.sh start    install the background watcher (Telegram alerts) with launchd
#   ./install.sh stop     remove the background watcher
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
PLIST=~/Library/LaunchAgents/com.signalbot.watch.plist

case "${1:-install}" in
  install)
    PY=$(command -v python3.12 || command -v python3.13 || command -v python3.14 || command -v python3)
    [ -d .venv ] && ! .venv/bin/python -c "import sys" 2>/dev/null && rm -rf .venv
    [ -d .venv ] || "$PY" -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
    echo
    echo "Installed. Run:   .venv/bin/python run.py serve"
    ;;
  start)
    [ -f .venv/bin/python ] || { echo "run ./install.sh first"; exit 1; }
    [ -f .env ] || { echo "create .env with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first (see README)"; exit 1; }
    mkdir -p ~/Library/LaunchAgents state
    sed -e "s#__HERE__#$HERE#g" launchd/com.signalbot.watch.plist.template > "$PLIST"
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "Watcher installed: checks every 15 minutes during market hours and messages you on new signals."
    echo "Log: $HERE/state/watch.log"
    ;;
  stop)
    launchctl unload "$PLIST" 2>/dev/null && rm -f "$PLIST" && echo "watcher removed" || echo "watcher was not installed"
    ;;
  *) echo "usage: ./install.sh [install|start|stop]"; exit 1;;
esac
