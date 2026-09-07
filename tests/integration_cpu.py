"""Opt-in full lifecycle check: fake API, real Docker; never scientific evidence.
Run: .venv/bin/python tests/integration_cpu.py
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontis_mila.common import ROOT, read, write
from frontis_mila.cli import preflight, start
from frontis_mila.analysis import audit, checkpoint, export

CODE = """import os
import pandas as pd
pd.read_csv(os.path.join(os.environ['DATA_DIR'],'sample_submission.csv')).to_csv('submission.csv',index=False)
print('Final Validation Score: 0.5')
"""


class Handler(BaseHTTPRequestHandler):
    generation_count = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = "\n".join(m["content"] for m in payload["messages"])
        if messages.startswith(
            "You are an expert machine learning experiment analyst."
        ):
            content = json.dumps(
                {
                    "method_overview": "Constant probabilities for the engineering test.",
                    "parent_comparison_experience": "The fixed program validates the execution path.",
                }
            )
        elif messages == "Reply with OK.":
            content = "OK"
        else:
            Handler.generation_count += 1
            code = (
                "raise RuntimeError('intentional engineering failure')"
                if Handler.generation_count == 1
                else CODE
            )
            content = "Engineering fixture.\n```python\n" + code + "\n```"
        data = {
            "id": f"fixture-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mimo-v2.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 100,
                "total_tokens": 200,
            },
        }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    directory = ROOT / "artifacts" / ("integration-" + str(time.time_ns()))
    directory.mkdir(parents=True)
    env = {
        "data_root": str(ROOT / "data"),
        "runs_root": str(directory),
        "api": {
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "key_env": "FRONTIS_FIXTURE_API_KEY",
        },
        "sandbox": {
            "backend": "docker",
            "image": "frontis-cpu:0.1",
            "cpu_count": 2,
            "memory": "6g",
            "pids_limit": 256,
            "tmpfs_size": "1g",
        },
    }
    os.environ["FRONTIS_FIXTURE_API_KEY"] = "test-fixture"
    preflight(env, live=True)
    run = directory / "smoke"

    def pause_after_commit():
        while not (run / "status.json").exists():
            time.sleep(0.1)
        while read(run / "status.json")["state"] == "running":
            try:
                _, bundle = checkpoint(run)
                if len(read(bundle / "population_state.json")["candidate_slots"]) >= 2:
                    (run / "PAUSE").touch()
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.1)

    thread = threading.Thread(target=pause_after_commit, daemon=True)
    thread.start()
    start(env, "uci-sms-spam", "smoke", 2026090601, smoke=True)
    assert read(run / "status.json")["state"] == "paused"
    audit(run, require_complete=False)
    _, bundle = checkpoint(run)
    prefix = read(bundle / "population_state.json")["candidate_slots"]
    start(env, "uci-sms-spam", "smoke", 2026090601, resume=True, smoke=True)
    result = audit(run)
    _, bundle = checkpoint(run)
    assert (
        read(bundle / "population_state.json")["candidate_slots"][: len(prefix)]
        == prefix
    )
    assert result["slots"] == 15
    assert result["sandbox_jobs"] == 16
    assert read(run / "analysis/summary.json")["operations"].get("crossover", 0) > 0
    export(run, directory / "complete.tar.gz")
    # Missing evidence must prevent a normal handoff.
    item = next((run / "request_ledger").glob("*.json"))
    saved = item.read_bytes()
    item.unlink()
    try:
        try:
            audit(run)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("Missing request ledger did not fail audit")
    finally:
        item.write_bytes(saved)
    write(
        directory / "RESULT.json",
        {
            "status": "passed",
            "verification": result,
            "prefix_preserved": True,
            "paid_api_calls": 0,
        },
    )
    task_reports = {"uci-sms-spam": result}
    for task in ("uci-adult-income", "uci-bike-sharing"):
        start(env, task, task, 2026090601, smoke=True)
        task_reports[task] = audit(directory / task)
        assert task_reports[task]["slots"] == 15
        export(directory / task, directory / (task + ".tar.gz"))
    write(
        directory / "ALL_TASKS_RESULT.json",
        {"status": "passed", "tasks": task_reports, "paid_api_calls": 0},
    )
    server.shutdown()
    print("Integration evidence:", directory)


if __name__ == "__main__":
    main()
