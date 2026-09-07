#!/usr/bin/env python3
"""Prepare the frozen S1 public/hidden task splits from official UCI archives."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "data" / "sources" / "uci"
PUBLIC_ROOT = REPO_ROOT / "data" / "s1"
PRIVATE_ROOT = REPO_ROOT / "private" / "s1"

SOURCES = {
    "sms": {
        "filename": "sms-spam-228.zip",
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "sha256": "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
    },
    "adult": {
        "filename": "adult-2.zip",
        "url": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "sha256": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
    },
    "bike": {
        "filename": "bike-sharing-275.zip",
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    },
}

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify_sources() -> dict[str, Path]:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, source in SOURCES.items():
        path = SOURCE_ROOT / str(source["filename"])
        if not path.exists():
            with urllib.request.urlopen(str(source["url"]), timeout=60) as response:
                path.write_bytes(response.read())
        actual = sha256_path(path)
        if actual != source["sha256"]:
            raise ValueError(
                f"source hash mismatch for {path.name}: expected {source['sha256']}, got {actual}"
            )
        paths[key] = path
    return paths


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def secure_hidden_file(path: Path) -> None:
    path.parent.chmod(0o700)
    path.chmod(0o600)


def prepare_sms(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8")
    records = []
    for source_index, line in enumerate(raw.splitlines()):
        label, text = line.split("\t", 1)
        records.append({"id": f"sms-{source_index:05d}", "text": text, "label": label})
    full = pd.DataFrame(records)
    if len(full) != 5574 or set(full["label"]) != {"ham", "spam"}:
        raise ValueError("unexpected SMS source schema or labels")
    if full.groupby("text")["label"].nunique().max() != 1:
        raise ValueError("an exact SMS text occurs with conflicting labels")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260830,
    )
    _visible_index, hidden_index = next(
        splitter.split(full, y=full["label"], groups=full["text"])
    )
    hidden_mask = np.zeros(len(full), dtype=bool)
    hidden_mask[hidden_index] = True
    visible = full.loc[~hidden_mask].reset_index(drop=True)
    hidden = full.loc[hidden_mask].reset_index(drop=True)
    if set(visible["text"]).intersection(hidden["text"]):
        raise ValueError("SMS duplicate group leaked across the hidden split")

    public_dir = PUBLIC_ROOT / "uci-sms-spam"
    answer_path = PRIVATE_ROOT / "uci-sms-spam" / "answers.csv"
    write_csv(visible, public_dir / "train.csv")
    write_csv(hidden[["id", "text"]], public_dir / "test.csv")
    sample = hidden[["id"]].copy()
    sample["ham"] = 0.5
    sample["spam"] = 0.5
    write_csv(sample, public_dir / "sample_submission.csv")
    answers = hidden[["id"]].copy()
    answers["ham"] = (hidden["label"] == "ham").astype(int)
    answers["spam"] = (hidden["label"] == "spam").astype(int)
    write_csv(answers, answer_path)
    secure_hidden_file(answer_path)
    return task_record(
        "uci-sms-spam", public_dir, answer_path, len(visible), len(hidden)
    )


def _read_adult_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = archive.read(member).decode("utf-8")
    rows = [
        line for line in raw.splitlines() if line.strip() and not line.startswith("|")
    ]
    frame = pd.read_csv(
        StringIO("\n".join(rows)),
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )
    frame["income"] = frame["income"].astype(str).str.strip().str.removesuffix(".")
    return frame


def prepare_adult(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        visible = _read_adult_member(archive, "adult.data")
        hidden = _read_adult_member(archive, "adult.test")
    if len(visible) != 32561 or len(hidden) != 16281:
        raise ValueError("unexpected Adult source row counts")
    if set(visible["income"]) != {"<=50K", ">50K"} or set(hidden["income"]) != {
        "<=50K",
        ">50K",
    }:
        raise ValueError("unexpected Adult target labels")
    visible.insert(
        0, "id", [f"adult-train-{index:05d}" for index in range(len(visible))]
    )
    hidden.insert(0, "id", [f"adult-test-{index:05d}" for index in range(len(hidden))])

    public_dir = PUBLIC_ROOT / "uci-adult-income"
    answer_path = PRIVATE_ROOT / "uci-adult-income" / "answers.csv"
    write_csv(visible, public_dir / "train.csv")
    write_csv(hidden.drop(columns=["income"]), public_dir / "test.csv")
    sample = hidden[["id"]].copy()
    sample["probability_gt_50k"] = 0.5
    write_csv(sample, public_dir / "sample_submission.csv")
    answers = hidden[["id"]].copy()
    answers["probability_gt_50k"] = (hidden["income"] == ">50K").astype(int)
    write_csv(answers, answer_path)
    secure_hidden_file(answer_path)
    return task_record(
        "uci-adult-income", public_dir, answer_path, len(visible), len(hidden)
    )


def prepare_bike(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        full = pd.read_csv(BytesIO(archive.read("hour.csv")))
    required = {"instant", "dteday", "casual", "registered", "cnt"}
    if len(full) != 17379 or not required.issubset(full.columns):
        raise ValueError("unexpected Bike Sharing source schema or row count")
    full["dteday"] = pd.to_datetime(full["dteday"], errors="raise")
    hidden_mask = full["dteday"] >= pd.Timestamp("2012-10-01")
    visible = full.loc[~hidden_mask].copy()
    hidden = full.loc[hidden_mask].copy()
    if (
        visible.empty
        or hidden.empty
        or visible["dteday"].max() >= hidden["dteday"].min()
    ):
        raise ValueError("Bike temporal split is invalid")
    public_columns = [
        column for column in full.columns if column not in {"casual", "registered"}
    ]
    visible_public = visible[public_columns].copy()
    hidden_public = hidden[public_columns].drop(columns=["cnt"]).copy()
    for frame in (visible_public, hidden_public):
        frame["dteday"] = frame["dteday"].dt.strftime("%Y-%m-%d")
    if {"casual", "registered"}.intersection(visible_public.columns) or {
        "casual",
        "registered",
        "cnt",
    }.intersection(hidden_public.columns):
        raise ValueError("Bike target or additive target components leaked")

    public_dir = PUBLIC_ROOT / "uci-bike-sharing"
    answer_path = PRIVATE_ROOT / "uci-bike-sharing" / "answers.csv"
    write_csv(visible_public, public_dir / "train.csv")
    write_csv(hidden_public, public_dir / "test.csv")
    sample = hidden[["instant"]].copy()
    sample["cnt"] = float(visible["cnt"].median())
    write_csv(sample, public_dir / "sample_submission.csv")
    write_csv(hidden[["instant", "cnt"]], answer_path)
    secure_hidden_file(answer_path)
    return task_record(
        "uci-bike-sharing", public_dir, answer_path, len(visible), len(hidden)
    )


def task_record(
    task: str,
    public_dir: Path,
    answer_path: Path,
    visible_rows: int,
    hidden_rows: int,
) -> dict:
    if public_dir.resolve() in answer_path.resolve().parents:
        raise ValueError(f"hidden answers for {task} are inside the public tree")
    files = {
        path.name: sha256_path(path)
        for path in sorted(public_dir.iterdir())
        if path.is_file()
    }
    files["private/answers.csv"] = sha256_path(answer_path)
    return {
        "task": task,
        "visible_rows": int(visible_rows),
        "hidden_rows": int(hidden_rows),
        "public_columns": list(pd.read_csv(public_dir / "test.csv", nrows=0).columns),
        "file_sha256": files,
    }


def prepare(data_root: Path, no_download: bool = False) -> dict:
    global SOURCE_ROOT, PUBLIC_ROOT, PRIVATE_ROOT
    SOURCE_ROOT = data_root / "sources"
    PUBLIC_ROOT = data_root / "public"
    PRIVATE_ROOT = data_root / "private"
    if no_download and any(
        not (SOURCE_ROOT / x["filename"]).is_file() for x in SOURCES.values()
    ):
        raise ValueError("Pinned source archives are missing from data_root/sources")
    sources = download_and_verify_sources()
    records = [
        prepare_sms(sources["sms"]),
        prepare_adult(sources["adult"]),
        prepare_bike(sources["bike"]),
    ]
    expected = json.loads((REPO_ROOT / "provenance/expected-data.json").read_text())
    if records != expected["tasks"]:
        raise ValueError(
            "Prepared data differs from the frozen historical splits; do not run"
        )
    manifest = {"sources": SOURCES, "tasks": records}
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
