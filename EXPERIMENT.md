# CPU Frontis controller baseline protocol

## Objective and estimand

Describe the search trajectory produced by standard synchronous OpenMLE-Evo with
OpenCode Go `mimo-v2.5` under a fixed CPU envelope and a 43,200-second cumulative
candidate-slot budget on each of three historical UCI tasks. A statistical unit is
one complete task/run. Generations, related candidates and repeated parents are
dependent observations. This study generates baseline evidence and hypotheses for
future interventions; it does not estimate an intervention effect.

## Frozen settings

The machine-readable authority is `configs/protocol.yaml` plus `configs/baseline.yaml`.
One island; capacity 500; five attempts per generation; generation zero is five
fresh Drafts. Improve/Crossover use the original score/delta/method-family novelty
utility (weights 1.0/0.4/0.25), score/delta min-max normalization, temperature 1.0.
Crossover probability is 0.5 beginning at generation 2. Admission uses the live
island mean. Migration is disabled, as in the one-island baseline. No AST/WL,
protected-elite, BAVS, forced action or learned selector is enabled.

The imported baseline uses an explicit sequential weighted sampler without
replacement. It exposes the same ordered-pair distribution as weighted sampling
without replacement, but is not an identical RNG-consumption path to every
upstream version. This is documented, frozen and tested; do not call trajectories
bitwise reproductions of the original paper.

The generation/step caps are set deliberately high so the time budget, rather than
the previous 20-generation setting, is the primary stopping rule. Debug depth 10
and Debug time cap 21,600 seconds follow the imported standard solver defaults.
Each actual candidate process is limited to 120 seconds (SMS) or 180 seconds
(Adult/Bike), 2 CPU cores and 6 GB memory. Debug depth counts additional programs,
not extra evolutionary slots. The package envelope remains the historical four
CPU libraries. Operator/model limits are serialized in `runner.yaml` and the
resolved upstream config; API requests preserve their actual effective arguments.

## Budget and stopping

Budget is the sum of committed slot `effective_seconds`, measured with a monotonic
clock from before selection to the end of generation/evaluation/Debug for that
slot. The local sandbox has no GPU queue exemptions. API time and transport
backoff count. Model-serving setup, data preparation, checkpoint writes outside
the measured slot, inter-generation overhead and downtime do not count.

At 43,200 seconds the next slot is refused; a slot already started may complete,
including its Debug chain. Report the overshoot and elapsed job segments. A pause
also waits for this boundary. A crash in a slot freezes the run for investigation;
it must not silently erase already generated candidates or API work.

This is an explicit **12-hour local CPU search protocol**, not the paper's GPU
compute-equivalent budget. The pinned upstream experiment preset separately uses
43,200 seconds of sandbox time and 64,800 seconds model-plus-sandbox time. Those
are different estimands; this study must not be presented as matching that budget.

## Data and evaluation boundary

Preparation is copied from the historical S1/S2 task adapter. Source ZIP hashes,
row counts, split rules and output CSV hashes are pinned in provenance.

- SMS: stratified exact-text-group split, fixed split seed 20260830.
- Adult: official training/test partition, normalized label strings.
- Bike: temporal cutoff 2012-10-01, excluding `casual` and `registered`.

The files named `test.csv` are **controller-validation inputs**. The evaluator's
labels are unavailable to candidate programs and the API. Their aggregate score
is used by selection and therefore is adaptive validation, not untouched test
performance. Numeric selection feedback is sanitized from generation prompts
using the inherited adapter, but this does not make the split an independent test.
No fresh test split is invented in this repository.

Common evaluator scores are log loss for SMS/Adult and RMSLE for Bike. Model-
printed validation scores are recorded separately and never substitute for the
common metric. Final selection follows the best valid journal candidate; the live
population is a different set. A run with no valid candidate has a missing final
score and is reported as unsuccessful, not silently dropped.

## Sampling, failures and controls

Controller seeds: 2026090601, 2026090602, 2026090603. Model seed is null because the
provider's earlier accepted seed was not repeatable. These are stochastic runs,
not seed-paired model outcomes. First deliverable is one run on each task; optional
replicates use the remaining seeds and are reported separately.

Only transport errors (timeout/connection/rate limit/HTTP 5xx) retry, at most three
physical attempts per logical request. Invisible provider generation may have
occurred on failed transport attempts; its unreported usage is unknown.
Format failures, candidate runtime failures, timeout, invalid submission, evaluator
failure and infrastructure failures remain distinguishable. Infrastructure errors
stop the run, never become evidence that a candidate is poor.

Future comparisons must preserve task, data, CPU envelope, model/API settings,
initialization design, evaluator and budget. Preserve these baseline trajectories;
interventions require new isolated runs or explicitly verified clones.

## Required interpretation

Lead with descriptive observations. Repeated stagnation, parent quality versus
child gain, crossover performance and novelty effects are hypotheses until a
matching intervention tests them. Three tasks here are UCI adaptations, not three
official MLE-Bench competitions. Results from MiMo do not establish Frontis-MA1
model performance, GPU-task performance or general benchmark superiority.
