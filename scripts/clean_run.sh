#!/usr/bin/env bash
# One-command clean run (Linux/macOS): install deps, pytest, start API.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
python -m pytest -q
export ASK_ECHO=1
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
