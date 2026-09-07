"""Read-only evidence validation, derived tables and complete handoff archives."""

from __future__ import annotations
import csv
import json
import math
import os
import tarfile
from pathlib import Path
from collections import Counter
from frontis_mila.common import digest, read, write


def rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def checkpoint(run):
    root = run / "search/aira_evo/checkpoint"
    pointer = read(root / "slot_current.json")
    if not __import__("re").fullmatch("[a-f0-9]{32}", pointer["bundle"]):
        raise ValueError("Invalid checkpoint bundle ID")
    if set(pointer["sha256"]) != {
        "state.json",
        "population_state.json",
        "journal.jsonl",
    }:
        raise ValueError("Incomplete checkpoint bundle")
    bundle = root / "slot_commits" / pointer["bundle"]
    for name, sha in pointer["sha256"].items():
        if digest(bundle / name) != sha:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    return root, bundle


def status_report(run):
    state = read(run / "status.json")
    if (run / "search/aira_evo/checkpoint/slot_current.json").exists():
        root, bundle = checkpoint(run)
        pop = read(bundle / "population_state.json")
        state.update(
            committed_slots=len(pop["candidate_slots"]),
            generation=pop["solver_state"]["current_generation"],
            used_seconds=pop["solver_state"]["running_time"],
            inflight=(root / "inflight_slot.json").exists(),
        )
    return state


def audit(run, require_complete=True, allow_pause=False):
    run = Path(run)
    manifest = read(run / "manifest.json")
    status = read(run / "status.json")
    if require_complete and status["state"] != "complete":
        raise ValueError("Run is not complete")
    for name, sha in manifest["source_sha256"].items():
        if digest(run / "frozen_repo" / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    root, bundle = checkpoint(run)
    pop = read(bundle / "population_state.json")
    state = read(bundle / "state.json")
    journal = rows(bundle / "journal.jsonl")
    ids = [n["id"] for n in journal]
    known = set(ids)
    if (
        state != pop["solver_state"]
        or ids != pop["journal_node_ids"]
        or len(known) != len(ids)
        or state["current_step"] != len(journal)
    ):
        raise ValueError("Journal and population disagree")
    for island in pop["population"]["islands"]:
        if not set(island["node_ids"]) <= known:
            raise ValueError("Unknown live node")
    slots = pop["candidate_slots"]
    keys = {(s["generation"], s["individual"]) for s in slots}
    if len(keys) != len(slots):
        raise ValueError("Duplicate candidate slots")
    guard = root / "inflight_slot.json"
    if guard.exists():
        pending = read(guard)
        if (pending["generation"], pending["individual"]) not in keys:
            raise ValueError(
                "Uncommitted inflight slot: manual investigation required; never delete/replay it"
            )
    if not math.isclose(
        sum(s["effective_seconds"] for s in slots), state["running_time"], abs_tol=0.01
    ):
        raise ValueError("Budget and slot accounting disagree")
    events = rows(run / "events.jsonl")
    updates = [e for e in events if e["kind"] == "population_update"]
    if len(updates) != state["current_generation"]:
        raise ValueError("Missing or uncommitted population update event")
    for first, second in zip(updates, updates[1:]):
        if first["after"]["islands"] != second["before"]["islands"]:
            raise ValueError("Population event history is discontinuous")
    if updates and updates[-1]["after"]["islands"] != pop["population"]["islands"]:
        raise ValueError("Population event history disagrees with checkpoint")
    selections = [e for e in events if e["kind"] == "selection"]
    if len(selections) != len(slots):
        raise ValueError("Missing or uncommitted selection event")
    by_slot = {e["slot"]: e for e in events if e["kind"] == "selection"}
    ledger = {p.stem: read(p) for p in (run / "request_ledger").glob("*.json")}
    completed_requests = {
        e["request_ledger_id"] for e in events if e["kind"] == "api_complete"
    }
    if completed_requests != set(ledger):
        raise ValueError("Missing or unaccounted API request ledger")
    for node in journal:
        for metric in node.get("operators_metrics") or []:
            request_id = (metric.get("usage") or {}).get("request_ledger_id")
            if request_id and request_id not in ledger:
                raise ValueError("Node references missing API request")
    for key, record in ledger.items():
        if record["status"] != "complete":
            raise ValueError(f"Unresolved model request {key}")
    for s in slots:
        slot = f'g{s["generation"]:04d}-i{s["individual"]:04d}'
        if slot not in by_slot:
            raise ValueError(f"Missing selection event: {slot}")
        if not set(s["node_ids"]) <= known:
            raise ValueError("Slot references missing node")
        if not any(r["slot"] == slot for r in ledger.values()):
            raise ValueError(f"Missing model ledger for {slot}")
        matches = list(
            (root / "parent_decision_states").glob(
                f'generation_{s["generation"]:04d}_individual_{s["individual"]:04d}_*'
            )
        )
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous decision state for {slot}")
        for filename in ("population_state.json", "journal.jsonl"):
            if not (matches[0] / filename).is_file():
                raise ValueError("Incomplete decision state")
        pre = read(matches[0] / "population_state.json")
        if not all(
            k in pre
            for k in (
                "rng_state",
                "llm_generation_state",
                "experience_cards",
                "rich_summaries",
                "generation_progress",
            )
        ):
            raise ValueError("Incomplete policy snapshot")
        trace = by_slot[slot]["trace"]
        if not math.isclose(
            sum(trace["operator_probabilities"].values()), 1, abs_tol=1e-9
        ):
            raise ValueError("Invalid operator distribution")
        if trace.get("candidates"):
            if not math.isclose(
                sum(c["probability"] for c in trace["candidates"]), 1, abs_tol=1e-9
            ):
                raise ValueError("Invalid parent distribution")
            if "legal_action_distribution" in trace:
                distribution = trace["legal_action_distribution"]
                ids = distribution["node_ids"]
                p = distribution["probabilities"]
                selected = trace["selected_node_ids"]
                if (
                    len(ids) != len(set(ids))
                    or len(ids) != len(p)
                    or not all(0 < x <= 1 for x in p)
                    or not math.isclose(sum(p), 1, abs_tol=1e-9)
                ):
                    raise ValueError("Invalid factorized distribution")
                count = distribution["sample_size"]
                if (
                    len(selected) != count
                    or len(set(selected)) != count
                    or not set(selected) <= set(ids)
                ):
                    raise ValueError("Invalid selected support")
                probability = p[ids.index(selected[0])]
                if count == 2:
                    probability *= p[ids.index(selected[1])] / (1 - probability)
                if not math.isclose(
                    probability, trace["selected_action_probability"], abs_tol=1e-12
                ):
                    raise ValueError("Selected probability mismatch")
            else:
                actions = trace["legal_actions"]
                if not math.isclose(
                    sum(a["probability"] for a in actions), 1, abs_tol=1e-8
                ):
                    raise ValueError("Invalid action distribution")
                selected = [
                    a for a in actions if a["action_id"] == trace["selected_action_id"]
                ]
                if (
                    len(selected) != 1
                    or selected[0]["parent_node_ids"] != trace["selected_node_ids"]
                ):
                    raise ValueError("Selected action mismatch")
    all_job_ids = set()
    for node in journal:
        info = node.get("metric_info") or {}
        job = info.get("job_id")
        if not job:
            continue  # Root and extraction failures have no sandbox execution.
        all_job_ids.add(job)
        record = read(run / "sandbox_jobs/api_records" / f"{job}.json")
        result = record["result"]
        exec_id = result.get("execution_job_id")
        if not exec_id:
            raise ValueError(f"Missing execution linkage for {job}")
        execution = run / "sandbox_jobs" / exec_id
        for name in ("candidate.py", "stdout.txt", "stderr.txt"):
            if not (execution / name).is_file():
                raise ValueError(f"Missing {name} for {job}")
        if record["request"]["code"] != (execution / "candidate.py").read_text():
            raise ValueError("Executed code mismatch")
        if not node["is_buggy"]:
            if result.get("score") != node["metric"]:
                raise ValueError("Node metric is not the common evaluator score")
            if not (execution / "output/submission.csv").is_file():
                raise ValueError("Missing successful submission")
    api_records = {p.stem for p in (run / "sandbox_jobs/api_records").glob("*.json")}
    if api_records != all_job_ids:
        raise ValueError(
            "Orphan or missing sandbox jobs: investigate before handing off"
        )
    if (
        require_complete
        and not manifest["smoke"]
        and state["running_time"] < manifest["protocol"]["budget_seconds"]
    ):
        raise ValueError("Run ended before the formal time budget")
    return {
        "status": "passed",
        "nodes": len(journal),
        "slots": len(slots),
        "api_requests": len(ledger),
        "sandbox_jobs": len(api_records),
        "used_seconds": state["running_time"],
        "complete": status["state"] == "complete",
        "metric_scope": "adaptive_controller_validation",
    }


def analyze(run):
    verification = audit(run)
    _, bundle = checkpoint(run)
    journal = rows(bundle / "journal.jsonl")
    pop = read(bundle / "population_state.json")
    nodes = {n["id"]: n for n in journal}
    slots = pop["candidate_slots"]
    derived = run / "analysis"
    derived.mkdir(exist_ok=True)
    valid = [
        n
        for n in journal
        if n.get("code")
        and not n["is_buggy"]
        and isinstance(n.get("metric"), (int, float))
        and math.isfinite(n["metric"])
    ]
    best = min(valid, key=lambda n: n["metric"]) if valid else None
    attempted = {n for s in slots for n in s["node_ids"]}
    cumulative = 0.0
    incumbent = None
    table = []
    for slot in slots:
        cumulative += slot["effective_seconds"]
        outcomes = [nodes[n] for n in slot["node_ids"] if n in nodes]
        scores = [
            n["metric"]
            for n in outcomes
            if not n["is_buggy"]
            and isinstance(n.get("metric"), (int, float))
            and math.isfinite(n["metric"])
        ]
        if scores:
            incumbent = min(scores + [incumbent] if incumbent is not None else scores)
        table.append(
            {
                "generation": slot["generation"],
                "individual": slot["individual"],
                "operation": slot["operator"],
                "effective_seconds": cumulative,
                "incumbent": incumbent,
                "valid": bool(scores),
                "node_ids": " ".join(slot["node_ids"]),
            }
        )
    if table:
        with (derived / "attempts.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    ledger = [read(p) for p in (run / "request_ledger").glob("*.json")]
    totals = {
        k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in ledger)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    summary = {
        **verification,
        "best_node_id": best["id"] if best else None,
        "best_score": best["metric"] if best else None,
        "attempted_nodes": len(attempted),
        "valid_nodes": len(valid),
        "failed_nodes": sum(nodes[n]["is_buggy"] for n in attempted),
        "valid_slots": sum(r["valid"] for r in table),
        "operations": dict(Counter(s["operator"] for s in slots)),
        "final_population": [i["node_ids"] for i in pop["population"]["islands"]],
        "tokens": totals,
        "physical_api_attempts": sum(len(r["attempts"]) for r in ledger),
        "token_caveat": "Delivered response usage; unavailable usage may be estimated. Unseen provider work is not included.",
        "budget_overshoot_seconds": max(
            0,
            verification["used_seconds"]
            - read(run / "manifest.json")["protocol"]["budget_seconds"],
        ),
    }
    write(derived / "summary.json", summary)
    if best:
        (derived / "best.py").write_text(best["code"])
    return summary


def export(run, output, allow_incomplete=False):
    state = read(run / "status.json")["state"]
    if state == "running":
        raise ValueError("Pause the run before exporting a stable archive")
    if output.exists():
        raise ValueError("Refusing to overwrite an export")
    if run == output or run in output.parents:
        raise ValueError("Export outside the run directory")
    try:
        verification = audit(run)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        if not allow_incomplete:
            raise
        verification = {"status": "INCOMPLETE_DIAGNOSTIC_ONLY", "reason": str(exc)}
    # Detect credentials before archiving; no .env is ever part of the export.
    secrets_values = [
        v.encode()
        for k, v in os.environ.items()
        if len(v) > 12
        and any(x in k.upper() for x in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))
    ]
    files = []
    hashes = {}
    links = {}
    manifest = read(run / "manifest.json")
    for p in sorted(run.rglob("*")):
        if p.is_symlink():
            # Preserve link metadata without following it or packaging host data.
            links[str(p.relative_to(run))] = os.readlink(p)
            continue
        if (
            not p.is_file()
            or "__pycache__" in p.parts
            or p.suffix == ".pyc"
            or p.name in ("process.lock", "EXPORT_MANIFEST.json")
        ):
            continue
        if p.name == ".env":
            raise ValueError("Credential file in run directory")
        with p.open("rb") as f:
            tail = b""
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                block = tail + chunk
                if any(v in block for v in secrets_values):
                    raise ValueError(
                        f"Credential value detected in {p.relative_to(run)}"
                    )
                tail = block[-4096:]
        files.append(p)
        hashes[str(p.relative_to(run))] = digest(p)
    write(
        run / "EXPORT_MANIFEST.json",
        {
            "verification": verification,
            "sha256": hashes,
            "symlink_metadata_not_followed": links,
        },
    )
    files.append(run / "EXPORT_MANIFEST.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for p in files:
            archive.add(
                p, arcname=str(Path(run.name) / p.relative_to(run)), recursive=False
            )
    print(
        f'Exported {output} ({verification["status"]}); send this entire archive to the analysis team.'
    )
