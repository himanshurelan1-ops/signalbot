#!/bin/bash
cd "$(dirname "$0")"
[ -f .venv/bin/python ] || ./install.sh
exec .venv/bin/python run.py serve
