import json
from pathlib import Path
import pytest
from frontis_mila.common import verify_data, ROOT
from frontis_mila.analysis import checkpoint, export
from frontis_local.sandbox.service import JobStore
from frontis_local.sandbox.config import SandboxConfig


def test_pinned_data_hash_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_data(tmp_path)


def test_checkpoint_rejects_corruption(tmp_path):
    cp = tmp_path / "search/aira_evo/checkpoint"
    cp.mkdir(parents=True)
    bundle = "a" * 32
    (cp / "slot_commits" / bundle).mkdir(parents=True)
    hashes = {
        x: "0" * 64 for x in ("state.json", "population_state.json", "journal.jsonl")
    }
    for name in hashes:
        (cp / "slot_commits" / bundle / name).write_text("{}")
    (cp / "slot_current.json").write_text(
        json.dumps({"bundle": bundle, "sha256": hashes})
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint(tmp_path)


def test_active_export_is_refused(tmp_path):
    (tmp_path / "status.json").write_text('{"state":"running"}')
    with pytest.raises(ValueError, match="Pause"):
        export(tmp_path, tmp_path.parent / "x.tar.gz")


def test_protocol_forbids_intervention():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    parent = cfg["solver"]["experience"]["parent_selection"]
    assert parent["selection_policy"] == "frontis"
    assert parent["novelty_policy"] == "method_family"
    assert not parent.get("forced_action")
    assert cfg["solver"]["experience"]["slot_checkpointing"]
    assert cfg["solver"]["num_islands"] == 1
    assert cfg["llm"]["generation_kwargs"]["seed"] is None
