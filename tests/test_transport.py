import json
from types import SimpleNamespace
from pathlib import Path
import litellm
import pytest
from dojo.core.solvers.llm_helpers.request_ledger import recorded_query


def test_generic_http500_is_retried_with_same_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRA_REQUEST_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("FRONTIS_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(
        "dojo.core.solvers.llm_helpers.request_ledger.time.sleep", lambda _: None
    )
    calls = []

    def query(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            raise litellm.APIError(
                status_code=500,
                message="provider failure",
                model="muse-spark-1.3-contributor",
                llm_provider="openai",
            )
        return "OK", {"total_tokens": 2}

    value, usage = recorded_query(
        SimpleNamespace(query=query),
        [{"role": "user", "content": "test"}],
        {"transport_retries": 0},
    )
    assert value == "OK" and calls[0] == calls[1]
    record = json.loads(next((tmp_path / "ledger").glob("*.json")).read_text())
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["provider_generation_may_have_occurred"]


def test_nontransport_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRA_REQUEST_LEDGER_DIR", str(tmp_path / "ledger"))

    def query(*args, **kwargs):
        raise ValueError("invalid request")

    with pytest.raises(ValueError):
        recorded_query(SimpleNamespace(query=query), [], {"transport_retries": 0})
    record = json.loads(next((tmp_path / "ledger").glob("*.json")).read_text())
    assert len(record["attempts"]) == 1 and record["status"] == "failed"
