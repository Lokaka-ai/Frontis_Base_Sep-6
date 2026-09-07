# Validation status

Local engineering validation performed on 2026-09-06:

- All three public/private data splits reproduced their frozen historical hashes.
- 48 targeted tests passed: common graders, sandbox mounts and cleanup, policy/RNG
  invariance, compact action probabilities, transport retry classification,
  pre-slot budget stopping, checkpoint state/restore and integrity rejection.
- A fake OpenAI-compatible API and real Docker CPU containers exercised the full
  lifecycle on all three tasks, 15 evolutionary slots per task.
- The SMS fixture included a failing candidate and successful Debug, a pause in
  initialization, and resume preserving every committed prefix record.
- Crossover, memory summaries, admissions, code/prediction retention, analysis and
  archive export were exercised. Removing a request ledger caused audit rejection.
- These checks made no paid API calls and produced no scientific result.

Re-run the targeted suite:

```bash
.venv/bin/python -m pytest -q tests \
  upstream/OpenRSI/OpenMLE-Evo/third_party/aira-evo/tests/test_evo_population_checkpoint.py \
  upstream/OpenRSI/OpenMLE-Evo/third_party/aira-evo/tests/test_parent_policy.py
```

Optional full local integration check (requires prepared data and the candidate
image, uses no API credentials):

```bash
.venv/bin/python tests/integration_cpu.py
```

It stores excluded evidence under `artifacts/integration-*/`, including a result
JSON and complete exported test archives. These generated files are not in Git.

## Limits

No 12-hour scientific run, real OpenCode smoke, Mila scheduler, or alternative
container backend has been validated here. The experiment runner must run the real API preflight
and excluded search smoke on his deployment. The fake API proves the execution
and recording path, not Muse Spark response quality, account quota, or service stability.

Only committed-boundary automatic resume is supported. An interruption inside a
stochastic slot requires investigation. The 12-hour threshold is a soft boundary
that finishes the current attempt; it is not a strict 12-hour scheduler deadline.

The upstream optional tree visualizer can report a missing `id` attribute while
still exporting search JSON. This does not affect the evidence audit; use the
journal, decision records and our analysis tables. We have not made visualization
polish a prerequisite for this experiment.

The Muse Spark update additionally tests the Responses endpoint, output-token
parameter mapping, raw response/usage retention, and HTTP error propagation.
The full three-task fake-API integration suite passed on the Responses transport.
Real Muse Spark access still requires an API preflight in the deployment environment.
