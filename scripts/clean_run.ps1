# One-command clean run (Windows PowerShell): install deps, pytest, start API.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m pip install -r requirements.txt
python -m pytest -q
$env:ASK_ECHO = "1"
uvicorn api.main:app --host 0.0.0.0 --port 8000
