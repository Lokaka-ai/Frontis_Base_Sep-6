# Source identity and changes

The upstream base is FrontisAI/OpenRSI commit
`ece6cbdf115ed72c3b62643a836504d77365e3a0`.
See https://github.com/FrontisAI/OpenRSI.

We imported the existing instrumented local OpenMLE-Evo runtime. The import is
recorded file-by-file in `provenance/upstream.json`. Tracked differences from the
base commit are in `provenance/inherited-local-changes.patch`; files added outside
that diff are still included in the import's file-hash inventory. The pinned
commit identifies the ancestor, not a claim that the working source is pristine.

## Inherited changes

- OpenCode API adapter, configurable transport handling and request accounting.
- CPU task/evaluator boundary, consistent sandbox validation fitness and prompt
  score sanitization.
- Robust code extraction, failed-response retention and raw operator records.
- Explicit parent-action distributions and exact population/RNG checkpoints.
- Per-slot commits, pending-generation state and rejection of unresolved replay.
- Dormant AST/WL, BAVS and protected-elite code from previous research; these paths
  are not enabled by this repo's baseline configuration.

The explicit weighted sampler preserves the intended distribution but need not
consume RNG identically to the original numpy sampling call. Numerical/adapter
fixes and fallback behaviour are part of this versioned baseline. We do not
claim bitwise equivalence to every upstream release or to paper trajectories.

## This repository's changes

- Portable CPU/OpenCode run lifecycle and frozen source/data/environment manifest.
- A documented 12-hour cumulative slot budget, clean boundary pause and resume.
- Additional observational events for operator probabilities and population updates.
- Durable sandbox API records linked to actual code, logs and predictions.
- HTTP 5xx retry classification; infrastructure failures stop generation.
- Token-usage provenance, including estimated versus provider-reported counts.
- Strict evidence audit, derived analysis and complete diagnostic/result export.

Each run archives exact source files, so modifications by a teammate are retained.
The inherited patch is historical provenance; do not apply it to the already
instrumented vendored tree a second time.

## License

Original OpenRSI code retains its CC BY-NC 4.0 license in
`upstream/OpenRSI/LICENSE`. Vendored AIRA-Dojo retains its own license and
THIRD_PARTY_LICENSES.md. New project glue is provided for this research handoff;
no additional public relicensing of collaborators' work is asserted here.
