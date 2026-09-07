"""Portable experiment lifecycle. No scheduler, account or GPU assumptions."""

from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
import yaml
from frontis_mila.common import (
    ROOT,
    TASKS,
    digest,
    read,
    write,
    load_environment,
    source_hashes,
    source_files,
    verify_data,
)


def sandbox_config(env, run, task, identity=None):
    resource = env["sandbox"]
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    graders = dict(
        zip(TASKS, ("sms_binary_log_loss", "adult_binary_log_loss", "bike_rmsle"))
    )
    from frontis_local.sandbox.config import SandboxConfig

    payload = {
        k: resource[k]
        for k in ("image", "cpu_count", "memory", "pids_limit", "tmpfs_size")
    }
    if identity is not None:
        payload["image"] = identity  # Never resolve a mutable image tag during a run.
    data = Path(env["data_root"])
    payload.update(
        timeout_seconds=protocol["tasks"][task]["execution_timeout"],
        jobs_root=str(run / "sandbox_jobs"),
        tasks={
            f"/datasets/validation/{task}": {
                "public_data": str(data / "public" / task),
                "hidden_answers": str(data / "private" / task / "answers.csv"),
                "grader": graders[task],
                "phase": "validation",
            }
        },
    )
    write(run / "sandbox_config.json", payload)
    return SandboxConfig.from_file(run / "sandbox_config.json")


def image_id(env):
    return subprocess.check_output(
        ["docker", "image", "inspect", env["sandbox"]["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()


def preflight(env, live=False):
    verify_data(env["data_root"])
    identity = image_id(env)
    report = {
        "schema_version": 1,
        "data_verified": True,
        "image_id": identity,
        "source_sha256": source_hashes(),
        "sandbox": env["sandbox"],
        "api": env["api"],
        "tasks": {},
    }
    from frontis_local.sandbox.runner import run_candidate
    from frontis_local.sandbox.grading import grade_submission

    directory = Path(env["runs_root"]) / (
        "_preflight-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    directory.mkdir(parents=True, exist_ok=False)
    code = """import os, pathlib, socket
import pandas as pd
assert os.environ.get("OPENCODE_API_KEY") is None
assert not pathlib.Path("/var/run/docker.sock").exists()
assert not pathlib.Path("/datasets/input/answers.csv").exists()
assert not pathlib.Path("/datasets/input/private").exists()
try:
    socket.create_connection(("1.1.1.1",443), timeout=1)
except OSError:
    pass
else:
    raise RuntimeError("Candidate network is not isolated")
pd.read_csv(os.path.join(os.environ["DATA_DIR"],"sample_submission.csv")).to_csv("submission.csv",index=False)
print("Final Validation Score: 1.0")
"""
    for task in TASKS:
        config = sandbox_config(env, directory / task, task, identity)
        result = run_candidate(
            config, logical_data_dir=f"/datasets/validation/{task}", code=code
        )
        if result.returncode or result.submission_path is None:
            raise RuntimeError(
                f"Sandbox preflight failed for {task}: inspect {directory}"
            )
        score = grade_submission(
            config.tasks[f"/datasets/validation/{task}"].grader,
            result.submission_path,
            config.tasks[f"/datasets/validation/{task}"].hidden_answers,
        )
        report["tasks"][task] = {"execution_job_id": result.job_id, "score": score}
    report["live_api_verified"] = False
    if live:
        import httpx

        key = os.environ[env["api"]["key_env"]]
        response = httpx.post(
            env["api"]["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())[
                    "model_id"
                ],
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 128,
            },
            timeout=120,
        )
        response.raise_for_status()
        write(directory / "api-probe.json", response.json())
        report["live_api_verified"] = True
    write(directory / "report.json", report)
    write(Path(env["runs_root"]) / "preflight.json", report)
    print(f"Preflight passed. Evidence: {directory}")
    if not live:
        print("API was not called. Use preflight --live-api before the formal run.")


def start(env, task, run_id, seed, resume=False, smoke=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError(
            "Use a simple run ID containing letters, numbers, dash or underscore"
        )
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    if seed not in protocol["seeds"]:
        raise ValueError("Seed must be one of the preregistered controller seeds")
    key = os.environ.get(env["api"]["key_env"])
    if not key:
        raise ValueError(f'Set {env["api"]["key_env"]} in your environment')
    data = verify_data(env["data_root"])
    identity = image_id(env)
    fingerprint = source_hashes()
    from importlib.metadata import distributions

    runtime_versions = {d.metadata["Name"]: d.version for d in distributions()}
    gate = read(Path(env["runs_root"]) / "preflight.json")
    if (
        gate["source_sha256"] != fingerprint
        or gate["image_id"] != identity
        or gate["sandbox"] != env["sandbox"]
        or gate["api"] != env["api"]
        or not gate["live_api_verified"]
    ):
        raise ValueError(
            "Run preflight --live-api with this code, API and sandbox configuration first"
        )
    run = Path(env["runs_root"]) / run_id
    if resume:
        manifest = read(run / "manifest.json")
        if (
            manifest["source_sha256"] != fingerprint
            or manifest["image_id"] != identity
            or manifest["data"] != data
            or manifest["environment"] != env
            or manifest["runtime_versions"] != runtime_versions
        ):
            raise ValueError(
                "Frozen inputs changed. Do not resume under changed conditions."
            )
        if (manifest["task"], manifest["seed"], manifest["smoke"]) != (
            task,
            seed,
            smoke,
        ):
            raise ValueError("Resume task, seed and run kind must match the manifest")
        from frontis_mila.analysis import audit

        audit(run, require_complete=False, allow_pause=True)
        lifecycle = read(run / "status.json")["state"]
        if lifecycle in {"running", "complete"}:
            raise ValueError(
                f"Run state is {lifecycle}; investigate stale status or use pause before resume"
            )
        (run / "PAUSE").unlink(missing_ok=True)
    else:
        run.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "task": task,
            "seed": seed,
            "smoke": smoke,
            "evidence_class": (
                "excluded_engineering" if smoke else "descriptive_baseline"
            ),
            "source_sha256": fingerprint,
            "runtime_versions": runtime_versions,
            "image_id": identity,
            "data": data,
            "environment": env,
            "protocol": protocol,
            "created_at": time.time(),
            "seed_control": "controller_seeded_model_unseeded",
        }
        write(run / "manifest.json", manifest)
        write(run / "preflight.json", gate)
        for path in source_files():
            target = run / "frozen_repo" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (run / "installed-packages.txt").write_text(freeze)
        write(
            run / "host.json",
            {
                "python": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "docker_version": subprocess.check_output(
                    ["docker", "version", "--format", "{{json .}}"], text=True
                ),
            },
        )
    # One process per run; OS lock releases on crash. Lock file remains intentionally.
    import fcntl

    lock = (run / "process.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from frontis_local.sandbox.service import JobStore, handler_factory

    store = JobStore(sandbox_config(env, run, task, identity))
    sandbox_key = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(store, sandbox_key))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    cfg["seed"] = seed
    cfg["output_dir"] = str(run / "search")
    cfg["llm"]["base_url"] = env["api"]["base_url"]
    cfg["llm"]["model_id"] = protocol["model_id"]
    cfg["model_plus_sandbox_time_budget"] = protocol["budget_seconds"]
    cfg["solver"]["execution_timeout"] = protocol["tasks"][task]["execution_timeout"]
    if smoke:
        cfg["solver"]["num_generations"] = 3
    task_cfg = yaml.safe_load((ROOT / "tasks" / task / "config.yaml").read_text())
    task_cfg["interpreter_data_dir"] = str(Path(env["data_root"]) / "public" / task)
    task_cfg["sandbox"].update(
        base_url=endpoint,
        job_timeout=cfg["solver"]["execution_timeout"],
        wait_timeout=cfg["solver"]["execution_timeout"] + 120,
        verify_tls=False,
    )
    task_dir = run / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "config.yaml").write_text(yaml.safe_dump(task_cfg, sort_keys=False))
    (run / "runner.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    child_env = os.environ.copy()
    for k in tuple(child_env):
        if (
            k.startswith("AIRA_")
            or k.startswith("OPENMLE_")
            or k.startswith("PRIMARY_KEY")
        ):
            child_env.pop(k)
    child_env.update(
        PRIMARY_KEY=key,
        OPENMLE_LLM_API_KEY=key,
        SANDBOX_CPU_API_KEY=sandbox_key,
        FRONTIS_RUN_DIR=str(run),
        AIRA_REQUEST_LEDGER_DIR=str(run / "request_ledger"),
        AIRA_TRANSPORT_ATTEMPTS=str(protocol["transport_attempts"]),
        AIRA_LITELLM_STRUCTURED_MODE="plain_json",
        LOGGING_DIR=str(run / "logs"),
    )
    previous = read(run / "status.json") if (run / "status.json").exists() else {}
    segments = previous.get("segments", [])
    segments.append({"started_at": time.time(), "resume": resume})
    status = {"state": "running", "task": task, "seed": seed, "segments": segments}
    write(run / "status.json", status)

    def request_pause(_sig, _frame):
        (run / "PAUSE").touch()
        print("Pause requested; finishing the current candidate attempt.", flush=True)

    old_handlers = {
        s: signal.signal(s, request_pause)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
    }
    try:
        with (run / f"runner-{len(segments):03d}.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/worker.py"),
                    "--task-dir",
                    str(task_dir),
                    "--output-dir",
                    str(run / "search"),
                    "--runner-config",
                    str(run / "runner.yaml"),
                    "--sample-index",
                    "0",
                ],
                cwd=ROOT,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["worker_pid"] = proc.pid
            write(run / "status.json", status)
            result = proc.wait()
        segments[-1].update(finished_at=time.time(), return_code=result)
        state = (
            "stopped_requires_investigation"
            if result
            else ("paused" if (run / "PAUSE").exists() else "complete")
        )
        if source_hashes() != fingerprint or verify_data(env["data_root"]) != data:
            state = "stopped_requires_investigation"
        status.update(state=state)
        write(run / "status.json", status)
        if state == "complete":
            from frontis_mila.analysis import analyze

            analyze(run)
        print(f"{run_id}: {state}. Results: {run}")
        if state == "stopped_requires_investigation":
            raise RuntimeError(
                "Run stopped. Preserve all files and inspect the runner log; no automatic replay."
            )
    except BaseException:
        status["state"] = "stopped_requires_investigation"
        write(run / "status.json", status)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        server.shutdown()
        store.executor.shutdown(wait=True)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", type=Path, default=ROOT / "configs/environment.yaml"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--no-download", action="store_true")
    p = sub.add_parser("preflight")
    p.add_argument(
        "--live-api", action="store_true", help="Make one small, billable API request"
    )
    for command in ("run", "resume"):
        p = sub.add_parser(command)
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--run-id", required=True)
        p.add_argument("--seed", type=int, default=2026090601)
        p.add_argument(
            "--smoke",
            action="store_true",
            help="Excluded three-generation engineering run",
        )
    for command in ("status", "pause", "analyze", "audit", "export"):
        p = sub.add_parser(command)
        p.add_argument("run", type=Path)
        if command == "export":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command in ("status", "pause", "analyze", "audit", "export"):
        run = args.run.resolve()
        from frontis_mila.analysis import analyze, audit, export, status_report

        if args.command == "status":
            print(json.dumps(status_report(run), indent=2))
        elif args.command == "pause":
            (run / "PAUSE").touch()
            print("Pause requested at next candidate boundary.")
        elif args.command == "analyze":
            print(json.dumps(analyze(run), indent=2))
        elif args.command == "audit":
            print(json.dumps(audit(run), indent=2))
        else:
            export(run, args.output.resolve(), args.allow_incomplete)
        return
    env = load_environment(args.environment)
    if args.command == "prepare":
        from frontis_mila.prepare import prepare

        prepare(Path(env["data_root"]), args.no_download)
        print("All three frozen data splits verified.")
    elif args.command == "preflight":
        preflight(env, args.live_api)
    else:
        start(
            env, args.task, args.run_id, args.seed, args.command == "resume", args.smoke
        )


if __name__ == "__main__":
    main()
