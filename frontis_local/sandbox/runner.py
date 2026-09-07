"""Launch one generated candidate in a fresh, restricted Docker container."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig


class SandboxRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    job_id: str
    returncode: int
    stdout: str
    stderr: str
    submission_path: Path | None


def docker_command(
    config: SandboxConfig,
    *,
    logical_data_dir: str,
    candidate_path: Path,
    output_dir: Path,
) -> list[str]:
    if logical_data_dir not in config.tasks:
        raise SandboxRuntimeError(f"task is not allowlisted: {logical_data_dir}")
    task = config.tasks[logical_data_dir]
    public_data = task.public_data.resolve()
    candidate_path = candidate_path.resolve()
    output_dir = output_dir.resolve()
    if not public_data.is_dir():
        raise SandboxRuntimeError(
            f"public data directory does not exist: {public_data}"
        )
    if not candidate_path.is_file():
        raise SandboxRuntimeError(f"candidate file does not exist: {candidate_path}")
    if not output_dir.is_dir():
        raise SandboxRuntimeError(f"output directory does not exist: {output_dir}")

    return [
        "docker",
        "run",
        "--rm",
        "--name",
        "frontis-" + candidate_path.parent.name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--user",
        "65532:65532",
        "--cpus",
        str(config.cpu_count),
        "--memory",
        config.memory,
        "--pids-limit",
        str(config.pids_limit),
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_size},mode=1777",
        "--mount",
        f"type=bind,source={public_data},target=/datasets/input,readonly",
        "--mount",
        f"type=bind,source={candidate_path},target=/app/candidate.py,readonly",
        "--mount",
        f"type=bind,source={output_dir},target=/work",
        "--workdir",
        "/work",
        "--env",
        "DATA_DIR=/datasets/input",
        "--env",
        "SANDBOX_DATA_DIR=/datasets/input",
        config.image,
        "python",
        "-I",
        "/app/candidate.py",
    ]


def run_candidate(
    config: SandboxConfig,
    *,
    logical_data_dir: str,
    code: str,
) -> ExecutionResult:
    if len(code.encode("utf-8")) > 2_000_000:
        raise SandboxRuntimeError("candidate code exceeds the 2 MB limit")
    job_id = uuid.uuid4().hex
    job_dir = config.jobs_root / job_id
    job_dir.mkdir(parents=True, mode=0o700)
    candidate = job_dir / "candidate.py"
    candidate.write_text(code, encoding="utf-8")
    os.chmod(candidate, 0o444)
    output_dir = job_dir / "output"
    output_dir.mkdir(mode=0o777)
    os.chmod(output_dir, 0o777)

    command = docker_command(
        config,
        logical_data_dir=logical_data_dir,
        candidate_path=candidate,
        output_dir=output_dir,
    )
    (job_dir / "execution.json").write_text(
        json.dumps(
            {"job_id": job_id, "command": command, "status": "started"}, indent=2
        )
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired as exc:
        (job_dir / "timeout.json").write_text(
            json.dumps({"timeout_seconds": config.timeout_seconds})
        )
        (job_dir / "stdout.txt").write_text(
            (exc.stdout or b" ").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        (job_dir / "stderr.txt").write_text(
            (exc.stderr or b" ").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        container_name = "frontis-" + job_id
        cleanup = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if cleanup.returncode and "No such container" not in cleanup.stderr:
            raise SandboxRuntimeError("timed-out container cleanup failed") from exc
        return ExecutionResult(
            job_id=job_id,
            returncode=124,
            stdout=(
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            ),
            stderr=f"candidate timeout: exceeded {config.timeout_seconds}s wall time",
            submission_path=None,
        )
    except FileNotFoundError as exc:
        raise SandboxRuntimeError("Docker executable is unavailable") from exc

    (job_dir / "stdout.txt").write_text(completed.stdout)
    (job_dir / "stderr.txt").write_text(completed.stderr)
    (job_dir / "execution.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "command": command,
                "returncode": completed.returncode,
                "status": "complete",
            },
            indent=2,
        )
    )
    if completed.returncode == 125:
        raise SandboxRuntimeError("Docker failed to launch candidate")

    submission = output_dir / "submission.csv"
    return ExecutionResult(
        job_id=job_id,
        returncode=completed.returncode,
        stdout=completed.stdout[-200_000:],
        stderr=completed.stderr[-200_000:],
        submission_path=(
            submission if submission.is_file() and not submission.is_symlink() else None
        ),
    )
