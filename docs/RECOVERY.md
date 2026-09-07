# Pause and recovery

## Planned pause

Use `frontis pause RUN_DIR` or send SIGUSR1 to the launcher. It finishes the current
slot, saves the journal/population/RNG bundle, and stops before another selection.
Wait for `paused`. Resume with the same task, seed, environment, source and `--smoke`
flag where appropriate. Budget usage survives restarts. The code refuses to
continue a completed formal run or to change frozen scientific inputs.

## Unexpected interruption

A killed process may leave `status.json` saying `running`. Inspect the job and
process before treating this as a live run. The process lock prevents two local
launchers from owning the same run at once.

`inflight_slot.json` marks a slot that began. If it is already in the authoritative
committed checkpoint, the upstream loader can clear the stale guard. Otherwise
ordinary resume refuses to continue. This is intentional: restarting from the
last generation would resample API outputs and erase real work.

Return an archive with `--allow-incomplete` after the process has stopped. If the
status is stale `running`, the analysis team must verify the process has ended and
record that investigation before marking the lifecycle state stopped. Do not
silently rewrite the search checkpoint. Provider/sandbox retries and orphaned jobs
are part of the evidence.

This release supports automatic recovery at committed boundaries. It does not
claim general automatic recovery from a crash in the middle of a stochastic
request, evaluation, Debug chain or population update. Such recovery requires an
explicit incident-specific audit; preserving the evidence makes that possible.
