"""Validated configuration for the local candidate sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SandboxConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TaskMount:
    public_data: Path
    hidden_answers: Path
    grader: str
    phase: str


@dataclass(frozen=True)
class SandboxConfig:
    image: str
    cpu_count: float
    memory: str
    pids_limit: int
    timeout_seconds: int
    tmpfs_size: str
    jobs_root: Path
    tasks: dict[str, TaskMount]

    @classmethod
    def from_file(cls, path: Path) -> "SandboxConfig":
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        tasks: dict[str, TaskMount] = {}
        for logical_name, value in raw.get("tasks", {}).items():
            if not logical_name.startswith("/"):
                raise SandboxConfigError(
                    "logical task paths must be absolute container paths"
                )
            public_data = Path(value["public_data"]).expanduser().resolve()
            hidden_answers = Path(value["hidden_answers"]).expanduser().resolve()
            if public_data == hidden_answers or public_data in hidden_answers.parents:
                raise SandboxConfigError(
                    "hidden answers must not be inside the public data tree"
                )
            tasks[logical_name] = TaskMount(
                public_data=public_data,
                hidden_answers=hidden_answers,
                grader=str(value["grader"]),
                phase=str(value.get("phase", "validation")),
            )
            if tasks[logical_name].phase not in {"validation", "test"}:
                raise SandboxConfigError("task phase must be validation or test")
        config = cls(
            image=str(raw["image"]),
            cpu_count=float(raw["cpu_count"]),
            memory=str(raw["memory"]),
            pids_limit=int(raw["pids_limit"]),
            timeout_seconds=int(raw["timeout_seconds"]),
            tmpfs_size=str(raw["tmpfs_size"]),
            jobs_root=Path(raw["jobs_root"]).expanduser().resolve(),
            tasks=tasks,
        )
        if (
            config.cpu_count <= 0
            or config.pids_limit <= 0
            or config.timeout_seconds <= 0
        ):
            raise SandboxConfigError("resource limits must be positive")
        if not config.image or config.image.endswith(":latest"):
            raise SandboxConfigError(
                "candidate image must use an explicit non-latest tag or digest"
            )
        return config
