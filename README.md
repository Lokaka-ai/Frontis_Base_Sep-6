# Frontis CPU experiments with OpenCode

Run the original Frontis selection controller in local, standard OpenMLE-Evo on
three CPU tasks. OpenCode Go **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`) generates programs; isolated local
containers execute them. This is **not a Frontis-MA1 model reproduction** or an
MLE-Bench leaderboard submission.

The analysis team needs the **complete exported run archive**, not just the best program or final score.
You may adapt paths, scheduling and container deployment to your infrastructure.
Coordinate changes to the search policy, model, dataset, metric, candidate limits
or budget before starting a formal run.

## What will run

| Task ID | Prediction problem | Controller validation metric |
| --- | --- | --- |
| `uci-sms-spam` | SMS spam classification | Binary log loss, lower is better |
| `uci-adult-income` | Income above 50K | Binary log loss, lower is better |
| `uci-bike-sharing` | Hourly bike rental count | RMSLE, lower is better |

Each task gets its own **12-hour cumulative candidate-attempt budget**. An attempt
includes selection, API waiting/retries, execution, evaluation and its Debug chain.
The current attempt finishes before the next budget check, so elapsed time can
exceed 12 hours. Setup, downtime between restarts and generation/checkpoint writing
outside an attempt are excluded. Both actual elapsed time and budget usage are
recorded. Read [EXPERIMENT.md](EXPERIMENT.md) before interpreting results.

Start with one run per task. Extra independent replicates are optional, using the
other listed controller seeds. A seed does **not** make the remote model deterministic.

## Model and API route

Muse Spark uses OpenCode's `/responses` endpoint, not `/chat/completions`.
The adapter maps `max_tokens` to `max_output_tokens` and retains the full response
and usage metadata. Both preflight and search use this transport.
[OpenCode model and endpoint documentation](https://opencode.ai/docs/go/#endpoints).

The Contributor tier permits provider training on prompts and completions and has
regional availability restrictions. You should check availability and any
Contributor opt-in in his OpenCode account before the real preflight.
[OpenCode privacy terms](https://opencode.ai/docs/go/#privacy).

This is protocol `cpu-frontis-opencode-muse-12h-v2`. Use new run IDs and rerun
preflight after updating; do not resume a MiMo run with Muse Spark.

## 1. Install

Requirements: Python 3.12, a working Docker-compatible deployment for the supplied
reference sandbox, and access to the OpenCode API. **No GPU or local LLM is needed.**
The CPU sandbox limits are experiment settings, not assumptions about your machine.

```bash
git clone https://github.com/Lokaka-ai/Frontis_Base_Sep-6.git
cd Frontis_Base_Sep-6
bash scripts/setup.sh
cp configs/environment.example.yaml configs/environment.yaml
```

Edit `configs/environment.yaml` with your data/output paths and API endpoint.
Keep credentials in your environment, not in the YAML file or Git:

```bash
read -rs -p 'OpenCode API key: ' OPENCODE_API_KEY
export OPENCODE_API_KEY
```

The command above is for Bash. Your cluster's secret-loading mechanism is also fine.
Do not put the key in a submitted job script that you intend to share.

The supplied backend uses Docker. If your cluster uses Apptainer or another
container system, follow [ADAPTING.md](docs/ADAPTING.md). Merely running candidate
Python directly on the host does not preserve the evaluation boundary.

Build the CPU candidate image:

```bash
docker build -t frontis-cpu:0.1 -f containers/Dockerfile .
```

The image contains only Python, numpy, pandas, scipy and scikit-learn. The run
records its actual image ID. The host dependency lock and container library pins
are separate because they serve different roles.

## 2. Prepare data

```bash
.venv/bin/frontis prepare
```

This downloads three pinned public UCI archives, checks their SHA-256 hashes,
creates exactly the historical splits and checks every resulting CSV against the
frozen data manifest. To use previously downloaded archives, place the named ZIPs
in `<data_root>/sources/` and run `frontis prepare --no-download`.

`<data_root>/public/` contains candidate-visible files. `<data_root>/private/`
contains evaluator labels and must never be mounted into a candidate container.
Do not change a split or bypass a hash mismatch to get a run started.

## 3. Preflight

```bash
.venv/bin/frontis preflight --live-api
```

This executes a small fixed candidate on each task, tests scoring and sandbox
isolation, verifies inputs, and makes **one small billable model request**.
It saves an excluded engineering report. Omit `--live-api` for sandbox-only checks;
a formal run requires the complete preflight. Re-run it after source/config changes.

Then test the complete search path on a small excluded run:

```bash
.venv/bin/frontis run --task uci-sms-spam --run-id smoke-sms --smoke
.venv/bin/frontis audit runs/smoke-sms
```

This runs three generations of five attempts, including any Debug and memory calls.
It is billable and is not scientific evidence. Check the status and audit before
starting the long runs. Commands below assume the default `runs_root: runs`;
substitute your configured output path if different.

## 4. Run the experiment

Run one task:

```bash
.venv/bin/frontis run --task uci-sms-spam --run-id baseline1-sms
```

Or run all three sequentially:

```bash
bash scripts/run_all.sh baseline1
```

Use a unique run ID. Existing directories are never overwritten. Run the command
inside your scheduler allocation or durable job environment. We deliberately do
not hard-code an account, partition or scheduler resource request.

A starting Slurm template is in `slurm/run.example.sbatch`. It is a template,
not a tested configuration for your cluster. Schedule one task per job if the
allocation is too short for three tasks. Allow time for setup and the final
inflight attempt. Request an early pause before the scheduler kills the process.

Controller seed defaults to `2026090601`. Optional independent repeats use
`--seed 2026090602` or `--seed 2026090603`; give them different run IDs.

## 5. Monitor, pause and resume

```bash
.venv/bin/frontis status runs/baseline1-sms
.venv/bin/frontis pause runs/baseline1-sms
.venv/bin/frontis resume --task uci-sms-spam --run-id baseline1-sms
```

Pause takes effect after the current attempt, which can include several Debug/API
calls. Wait until the state becomes `paused` before stopping the job. Resume uses
the same data, code, environment settings and seed, verifies the checkpoint, and
continues with the remaining budget. Use `--smoke` when resuming a smoke run.

**A forced kill during an attempt is different.** Its request and sandbox records
are preserved, but the launcher refuses to regenerate that attempt. Send the
incomplete diagnostic archive to the analysis team. Do not delete the inflight
marker or roll back a generation. See [RECOVERY.md](docs/RECOVERY.md).

## 6. Return results to the analysis team

For each completed run:

```bash
.venv/bin/frontis audit runs/baseline1-sms
.venv/bin/frontis analyze runs/baseline1-sms
.venv/bin/frontis export runs/baseline1-sms --output exports/baseline1-sms.tar.gz
```

Send the **whole `.tar.gz` file**. Export checks completeness, includes exact source
snapshots and all research records, and creates a file-hash manifest. Do not remove
failed programs, old checkpoints, API ledgers, sandbox jobs or large log files.

If a run stopped unexpectedly, preserve the directory and export it explicitly as
incomplete:

```bash
.venv/bin/frontis export runs/baseline1-sms \
  --output exports/baseline1-sms-diagnostic.tar.gz --allow-incomplete
```

This archive is labeled diagnostic-only and cannot pass as a completed experiment.
Known environment credential values are checked before export; no `.env` is allowed.
Treat the archive as research data for our team, not as a public GitHub asset:
model-generated code, prompts and outputs can contain dataset text.

## Where things are

| Path | Purpose |
| --- | --- |
| `configs/protocol.yaml`, `configs/baseline.yaml` | Scientific settings |
| `configs/environment.example.yaml` | Fields you configures |
| `tasks/` | Exact task instructions and adapter metadata |
| `upstream/` | Pinned, instrumented OpenMLE-Evo source |
| `frontis_mila/` | Lifecycle, recording, audit and export |
| `frontis_local/sandbox/` | CPU execution and host-only grading |
| `provenance/` | Upstream identity, inherited patches, data hashes |
| `docs/DATA_DICTIONARY.md` | How to analyze the returned records |
| `docs/VALIDATION.md` | Tests performed and deployment limits |

## Tests and troubleshooting

```bash
.venv/bin/python -m pytest -q tests
```

- Missing Python library: install the lock file in the repository virtualenv.
- No Docker access: adapt the sandbox following the documented contract.
- API authentication/rate limit: check the key and account limits; preserve request ledgers.
- Changed input or missing artifact: stop and send the error to the analysis team.
- `running` after a machine crash: this is the last persisted state, not proof of a
  live process. Investigate before changing it or attempting recovery.

## Provenance and license

See [UPSTREAM.md](UPSTREAM.md). The original upstream license and AIRA-Dojo notices
are retained. This distribution contains inherited instrumentation and documented
local fixes; it is not a pristine upstream checkout. The active configuration
locks selection to the original Frontis utility and method-family novelty.
