"""Host-only append-only audit events. Logging never samples RNG."""

from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def event(kind, **payload):
    root = os.environ.get("FRONTIS_RUN_DIR")
    if not root:
        return
    path = Path(root) / "events.jsonl"
    record = {
        "schema_version": 1,
        "kind": kind,
        "time": time.time(),
        "slot": os.environ.get("AIRA_CANDIDATE_SLOT"),
        **payload,
    }
    with path.open("a") as f:
        f.write(json.dumps(clean(record), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
