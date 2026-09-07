# Reading a returned run

## Main files

| Relative path inside a run | Meaning |
| --- | --- |
| `manifest.json` | Task, controller seed, data/source hashes, model protocol, environment and image ID |
| `frozen_repo/` | Exact source/configuration snapshot used to generate this run |
| `runner.yaml`, `task/config.yaml` | Effective runner and task inputs, without API keys |
| `status.json` | Last lifecycle state and job segments; not a live process guarantee |
| `events.jsonl` | Append-only slot, selection and population mutation events |
| `request_ledger/*.json` | Each logical model request, all physical attempts, messages, parameters, output and usage |
| `search/aira_evo/checkpoint/slot_current.json` | Authoritative pointer and hashes for the last committed checkpoint |
| `search/aira_evo/checkpoint/slot_commits/<id>/` | Immutable journal, population and solver state bundles |
| `search/aira_evo/checkpoint/parent_decision_states/` | Exact journal/population before each slot's selection |
| `search/step_*/` | Mirrored code, prompts, response, feedback and statistics |
| `sandbox_jobs/api_records/<job_id>.json` | Submitted code, timings and terminal evaluator result |
| `sandbox_jobs/<execution_job_id>/` | Actual code, execution command, stdout, stderr and output/submission.csv |
| `analysis/attempts.csv` | One row per evolutionary slot, including initialization and failures |
| `analysis/summary.json`, `analysis/best.py` | Derived summary and best valid program |
| `EXPORT_MANIFEST.json` | Completeness result and file hashes for the handoff |

## IDs and denominators

The root journal node is not a generated candidate. One evolutionary slot can
produce an initial program and several Debug nodes. A failed code extraction has
a model response but no executable program. Memory summaries are API requests but
not new candidates. Do not equate slots, journal nodes, sandbox jobs and API calls.

`candidate_slots` links `(generation, individual)` to ordered `node_ids`, valid
nodes and budget time. Generation zero is initialization. Journal `parents` and
`children` contain **journal step numbers**, while population membership and
selection events use node IDs. Join them through the journal's `step` and `id`.
The `step_*` folder number is not guaranteed to equal journal `step`; use its
experience card/statistics IDs or journal records, not an assumed offset.

`metric_info.job_id` links a node to an API job record. That record's
`result.execution_job_id` links to actual candidate.py and submission.csv.
`operators_metrics[*].usage.request_ledger_id` links generated nodes to model
requests. Ledger `slot` also links memory/debug calls to the originating slot.

## Selection and population

Selection events record operator probabilities, selected operator, island,
ordered eligible parent IDs, component utilities, parent probabilities, selected
parents and conditional legal-action probabilities. Crossover samples without
replacement: ordered pair `(i,j)` has probability `p_i*p_j/(1-p_i)`. The recorded
legal-action probabilities are conditional on the chosen operator, not multiplied
by the operator probability. With one island, its probability is one. The parent
weight distribution supports reconstructing both operators' legal distributions.

Pre-selection snapshots preserve Python/numpy RNG, operator generation state,
summary/experience state, global fitness range and exact island order. Provider
internal RNG/state is unavailable, so these are policy checkpoints, not a promise
to reproduce future API responses.

`population_update` contains ordered offered nodes plus population before/after.
`island_mutation` records registration and removal within the update. A registration
followed by removal differs from rejection. A valid child awaiting the end of its
generation is pending admission, not rejected. At a mid-generation pause or budget
stop, `generation_progress` preserves these pending children. Migration is disabled
in this protocol; instrumentation retains mutation types if the backend is extended.

## Metrics, usage and missingness

`metric` is the common controller-validation score. Use it across candidates;
model-printed validation may use different internal splits. Neither the word
`hidden` nor the filename `test.csv` makes controller feedback an untouched test.
`status_code=200` denotes a completed service interaction, not candidate validity.
Use `is_buggy`, failure type and the sandbox terminal result.

Usage records distinguish provider counts from estimated counts. Tokens for a
failed transport attempt with no delivered usage are unknown, not zero. The cost
field inherited from the backend is not a provider invoice. Count all ledger calls,
including memory and debugging, and report unknown usage separately.

Keep all snapshots, retries, failures and rejected candidates. Derived summaries
are convenience files; the immutable checkpoint, event log and job/request records
are the evidence. Hash verification detects alteration, not scientific validity.

## Compact distributions and storage

To avoid saving up to 249,500 pair rows at every decision, schema version 2 records
`legal_action_distribution` as ordered node IDs, normalized probabilities and
sample size. This is a lossless factorization of the complete action distribution;
`selected_action_probability` is also stored directly. The formula above expands
any pair on demand. It does not change sampling or omit legal parents.

Full journal/state snapshots intentionally duplicate some content. Check available
storage and growth during the smoke run; do not prune them during a formal run.
The export uses compression. Workspace/output symlinks are recorded as metadata
in the export manifest and never followed into host files.
