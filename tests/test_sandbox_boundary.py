from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontis_local.sandbox.config import SandboxConfig, SandboxConfigError
from frontis_local.sandbox.runner import SandboxRuntimeError, docker_command


def write_config(tmp_path: Path, *, hidden_inside_public: bool = False) -> Path:
    public = tmp_path / "public"
    public.mkdir()
    hidden = (
        public / "answers.csv"
        if hidden_inside_public
        else tmp_path / "private/answers.csv"
    )
    hidden.parent.mkdir(parents=True, exist_ok=True)
    hidden.write_text("answer\n", encoding="utf-8")
    config_path = tmp_path / "sandbox.json"
    config_path.write_text(
        json.dumps(
            {
                "image": "frontis-local-candidate:0.1",
                "cpu_count": 2,
                "memory": "6g",
                "pids_limit": 128,
                "timeout_seconds": 30,
                "tmpfs_size": "256m",
                "jobs_root": str(tmp_path / "jobs"),
                "tasks": {
                    "/datasets/validation/task": {
                        "public_data": str(public),
                        "hidden_answers": str(hidden),
                        "grader": "test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_hidden_answers_cannot_be_inside_public_mount(tmp_path: Path):
    with pytest.raises(SandboxConfigError):
        SandboxConfig.from_file(write_config(tmp_path, hidden_inside_public=True))


def test_docker_command_contains_required_isolation_and_no_hidden_path(tmp_path: Path):
    config = SandboxConfig.from_file(write_config(tmp_path))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("print('ok')\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    command = docker_command(
        config,
        logical_data_dir="/datasets/validation/task",
        candidate_path=candidate,
        output_dir=output,
    )
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges=true" in command
    assert "--user 65532:65532" in joined
    assert "target=/datasets/input,readonly" in joined
    assert (
        f"source={config.tasks['/datasets/validation/task'].hidden_answers}"
        not in joined
    )
    assert "QWEN_API_KEY" not in joined
    assert "/var/run/docker.sock" not in joined


def test_non_allowlisted_task_is_rejected(tmp_path: Path):
    config = SandboxConfig.from_file(write_config(tmp_path))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(SandboxRuntimeError, match="not allowlisted"):
        docker_command(
            config,
            logical_data_dir="/datasets/validation/unknown",
            candidate_path=candidate,
            output_dir=output,
        )


def test_timeout_removes_specific_container_and_returns_candidate_failure(
    tmp_path, monkeypatch
):
    import subprocess
    from types import SimpleNamespace
    from frontis_local.sandbox.runner import run_candidate

    config = SandboxConfig.from_file(write_config(tmp_path))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 30, output=b"partial")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_candidate(
        config, logical_data_dir="/datasets/validation/task", code="while True: pass"
    )
    assert result.returncode == 124 and result.submission_path is None
    assert calls[1] == ["docker", "rm", "-f", "frontis-" + result.job_id]
    assert calls[0][calls[0].index("--name") + 1] == "frontis-" + result.job_id
