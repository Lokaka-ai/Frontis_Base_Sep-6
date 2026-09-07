#!/usr/bin/env bash
# Sequential execution avoids cross-task contention. Pass a unique batch name.
set -euo pipefail
cd "$(dirname "$0")/.."
batch="${1:?Usage: scripts/run_all.sh BATCH_NAME [SEED]}"
seed="${2:-2026090601}"
for task in uci-sms-spam uci-adult-income uci-bike-sharing; do
  .venv/bin/frontis run --task "$task" --run-id "${batch}-${task}" --seed "$seed"
done
