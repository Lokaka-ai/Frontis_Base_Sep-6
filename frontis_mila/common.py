from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("uci-sms-spam", "uci-adult-income", "uci-bike-sharing")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def load_environment(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("data_root", "runs_root"):
        p = Path(cfg[k]).expanduser()
        cfg[k] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if cfg["sandbox"]["backend"] != "docker":
        raise ValueError(
            "Reference backend is Docker. See docs/ADAPTING.md for the adapter contract."
        )
    if cfg["sandbox"]["cpu_count"] != 2 or cfg["sandbox"]["memory"] != "6g":
        raise ValueError(
            "Candidate resource envelope changed. Agree a new protocol before running."
        )
    from urllib.parse import urlsplit

    url = urlsplit(cfg["api"]["base_url"])
    if url.scheme not in ("https", "http") or url.username or url.password or url.query:
        raise ValueError("API URL must not contain credentials or query parameters")
    if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("Non-local model API must use HTTPS")
    return cfg


def source_files():
    roots = [
        "frontis_mila",
        "frontis_local",
        "upstream",
        "tasks",
        "configs",
        "scripts",
        "containers",
        "provenance",
        "docs",
        "slurm",
    ]
    paths = [
        ROOT / n
        for n in [
            "README.md",
            "EXPERIMENT.md",
            "UPSTREAM.md",
            "pyproject.toml",
            "requirements.lock.txt",
        ]
        if (ROOT / n).exists()
    ]
    for root in roots:
        paths += [
            p
            for p in (ROOT / root).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in (".pyc",)
            and p.name not in ("environment.yaml", ".DS_Store")
        ]
    return sorted(set(paths))


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in source_files()}


def verify_data(data_root):
    root = Path(data_root)
    expected = read(ROOT / "provenance/expected-data.json")
    for row in expected["tasks"]:
        for name, sha in row["file_sha256"].items():
            path = (
                root / "private" / row["task"] / "answers.csv"
                if name == "private/answers.csv"
                else root / "public" / row["task"] / name
            )
            if digest(path) != sha:
                raise ValueError(f'Data hash mismatch: {row["task"]}/{name}')
    return expected
