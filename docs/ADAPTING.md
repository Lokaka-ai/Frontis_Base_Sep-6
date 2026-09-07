# Connecting the experiment to your infrastructure

## Expected local edits

Set data/output paths, an API endpoint and credential environment variable in
`configs/environment.yaml`. Select your scheduler account, queue and host resource
allocation outside the scientific config. The launcher selects a free loopback
port and runs one sandbox service per experiment. It does not need a fixed port,
GPU, model server or your home-directory layout.

The host needs more resources than the candidate's 2-core/6-GB envelope, because
the controller, evaluator, logs and container runtime also use memory and CPU.
Determine the job allocation from the preflight on your infrastructure. No exact
host allocation is assumed here. Run tasks sequentially unless you allocate
independent equivalent resource envelopes.

## If Docker is unavailable

The shipped backend is a concrete reference implementation, not a promise that
Docker is available on your cluster. Xixian may replace `docker_command` and
`run_candidate` in `frontis_local/sandbox/runner.py` or supply an equivalent local
container backend. Keep the `ExecutionResult` contract and persist the same job
artifacts. Record the new backend/configuration/image identity in the manifest,
update preflight and rerun validation before scientific generation. The current
CLI deliberately rejects an unknown backend instead of pretending to support it.

An equivalent backend must:

1. Run each candidate in a fresh isolated process/container with enforced limits.
2. Mount only that task's public files read-only, candidate code read-only, and a
   fresh writable output directory. Never expose host home, private labels,
   source ZIPs, evaluator code, credentials, Docker socket or another run's files.
3. Disable candidate network and GPU access, and use the pinned Python libraries.
4. Preserve executed code, stdout/stderr, exit/timeout status and submission.
5. Stop and clean up only its own timed-out process tree; distinguish a failed
   program from a container/host failure.
6. Grade on the trusted host and retain job IDs so node-to-submission links remain
   verifiable. Do not execute generated Python in the controller environment.

Do not claim equivalent containment simply from an Apptainer `--cleanenv` flag;
check filesystem binds and network isolation in the actual cluster setup.

## Changes requiring coordination

Changing model ID, API sampling arguments, prompts, score handling, input data,
CPU limits, execution timeout, Debug limits, selection/admission or budget creates
a different experiment configuration. Discuss it with the analysis team first.
Commit environment adapters before a formal run, and send the modified code with
the resulting archive. Frozen-source checks prevent changing a run midstream.

## Deployment validation

Local tests do not validate your scheduler or quota. Run the supplied preflight,
then the excluded search smoke test. Test a graceful pause and resume before the
12-hour job. A long run makes API calls continuously and may encounter account
rate limits; use your account's available quota and preserve retry evidence.
