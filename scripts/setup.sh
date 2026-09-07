#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3.12}" -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
printf '%s\n' 'Setup complete. Copy configs/environment.example.yaml to configs/environment.yaml, then follow README.md.'
