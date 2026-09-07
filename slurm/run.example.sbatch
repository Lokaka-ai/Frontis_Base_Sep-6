#!/usr/bin/env bash
# Template only: supply account, partition, CPU, memory and wall-time through
# sbatch options appropriate to your cluster. This file requests no GPUs.
# Ask the scheduler for a pre-timeout SIGUSR1 to the batch shell if supported.
set -euo pipefail
: "${FRONTIS_REPO:?Set FRONTIS_REPO to your checkout}"
: "${FRONTIS_TASK:?Set FRONTIS_TASK to one task ID from the README}"
: "${FRONTIS_RUN_ID:?Set FRONTIS_RUN_ID to a unique run name}"
cd "$FRONTIS_REPO"
child_pid=''
trap 'if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid"; fi' USR1 TERM INT
.venv/bin/frontis run --task "$FRONTIS_TASK" --run-id "$FRONTIS_RUN_ID" &
child_pid=$!
# A trapped signal interrupts wait; continue waiting for a clean boundary stop.
while true; do
  set +e
  wait "$child_pid"
  result=$?
  set -e
  if ! kill -0 "$child_pid" 2>/dev/null; then exit "$result"; fi
done
