"""OpenMLE-compatible localhost job API backed by the Docker sandbox."""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import SandboxConfig
from .grading import EvaluationError, EvaluationInfrastructureError, grade_submission
from .runner import SandboxRuntimeError, run_candidate


MAX_CODE_BYTES = 2_000_000
MAX_LOG_CHARS = 200_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobStore:
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.config.jobs_root.mkdir(parents=True, exist_ok=True)
        self.records = self.config.jobs_root / "api_records"
        self.records.mkdir(exist_ok=True)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.idempotency: dict[str, str] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="frontis-sandbox"
        )

    def _persist(self, job_id):
        path = self.records / (job_id + ".json")
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self.jobs[job_id], f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def submit(self, payload: dict[str, Any]) -> tuple[str, bool]:
        code = payload.get("code")
        data_dir = str(payload.get("data_dir", ""))
        key = str(payload.get("idempotency_key", "") or "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("code exceeds the 2 MB limit")
        if data_dir not in self.config.tasks:
            raise ValueError("data_dir is not allowlisted")
        with self.lock:
            if key and key in self.idempotency:
                return self.idempotency[key], False
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": _now(),
                "data_dir": data_dir,
                "_code": code,
            }
            self.jobs[job_id]["request"] = payload
            self._persist(job_id)
            if key:
                self.idempotency[key] = job_id
        self.executor.submit(self._execute, job_id)
        return job_id, True

    def public(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                saved = self.records / (job_id + ".json")
                if not saved.exists():
                    return None
                job = json.loads(saved.read_text())
                if job["status"] in {"queued", "running"}:
                    job.update(
                        status="failed",
                        result={
                            "result": "failed",
                            "failure_type": "sandbox_failure",
                            "score": None,
                        },
                    )
            return {
                key: value
                for key, value in job.items()
                if not key.startswith("_") and key != "request"
            }

    def _finish(self, job_id: str, *, status: str, result: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job_id].update(
                {"status": status, "completed_at": _now(), "result": result}
            )
            self._persist(job_id)

    def _execute(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _now()
            self._persist(job_id)
            code = job.pop("_code")
            data_dir = str(job["data_dir"])
        task = self.config.tasks[data_dir]
        try:
            execution = run_candidate(self.config, logical_data_dir=data_dir, code=code)
        except SandboxRuntimeError:
            self._finish(
                job_id,
                status="failed",
                result={
                    "result": "failed",
                    "failure_type": "sandbox_failure",
                    "score": None,
                },
            )
            return

        base_result: dict[str, Any] = {
            "execution_job_id": execution.job_id,
            "stdout": execution.stdout[-MAX_LOG_CHARS:],
            "stderr": execution.stderr[-MAX_LOG_CHARS:],
            "run_log": (
                "--- CANDIDATE STDOUT ---\n"
                + execution.stdout[-MAX_LOG_CHARS:]
                + "\n--- CANDIDATE STDERR ---\n"
                + execution.stderr[-MAX_LOG_CHARS:]
            ),
            "score": None,
        }
        if execution.returncode != 0 or execution.submission_path is None:
            base_result.update(
                {
                    "result": "failed",
                    "failure_type": (
                        "candidate_timeout"
                        if execution.returncode == 124
                        else "candidate_runtime_failure"
                    ),
                }
            )
            self._finish(job_id, status="failed", result=base_result)
            return
        try:
            score = grade_submission(
                task.grader, execution.submission_path, task.hidden_answers
            )
        except EvaluationInfrastructureError:
            base_result.update(
                {"result": "failed", "failure_type": "evaluator_failure"}
            )
            self._finish(job_id, status="failed", result=base_result)
            return
        except EvaluationError:
            base_result.update(
                {"result": "failed", "failure_type": "candidate_submission_failure"}
            )
            self._finish(job_id, status="failed", result=base_result)
            return
        except Exception:
            base_result.update(
                {"result": "failed", "failure_type": "evaluator_failure"}
            )
            self._finish(job_id, status="failed", result=base_result)
            return
        base_result.update(
            {
                "result": "success",
                "failure_type": None,
                "score": score,
                "scores_by_split": {task.phase: score},
            }
        )
        base_result[f"{'valid' if task.phase == 'validation' else 'test'}_score"] = (
            score
        )
        self._finish(job_id, status="completed", result=base_result)


def handler_factory(store: JobStore, api_key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FrontisLocalSandbox/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return bool(api_key) and self.headers.get("X-API-Key") == api_key

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            prefix = "/api/v1/jobs/"
            if not self.path.startswith(prefix):
                self._send(404, {"error": "not found"})
                return
            job = store.public(self.path[len(prefix) :])
            self._send(
                404 if job is None else 200,
                {"error": "not found"} if job is None else job,
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/v1/jobs":
                self._send(404, {"error": "not found"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_CODE_BYTES + 100_000:
                    raise ValueError("request size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                job_id, created = store.submit(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(
                202 if created else 200, {"job_id": job_id, "deduplicated": not created}
            )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6580)
    args = parser.parse_args()
    api_key = os.environ.get("FRONTIS_SANDBOX_API_KEY", "")
    if not api_key:
        raise SystemExit("FRONTIS_SANDBOX_API_KEY must be non-empty")
    store = JobStore(SandboxConfig.from_file(args.config))
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_factory(store, api_key)
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
