import httpx
import litellm
import pytest
from frontis_mila.responses_api import query, request_payload


def test_response_mapping_and_raw_usage(monkeypatch):
    raw = {
        "id": "r1",
        "model": "muse-spark-1.3-contributor",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"text": "summary"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "code"}]},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    }

    def post(url, **kwargs):
        assert url == "https://example.invalid/v1/responses"
        assert kwargs["json"]["model"] == "muse-spark-1.3-contributor"
        assert kwargs["json"]["max_output_tokens"] == 8192
        assert "max_tokens" not in kwargs["json"]
        return httpx.Response(200, json=raw)

    monkeypatch.setattr(httpx, "post", post)
    text, usage = query(
        base_url="https://example.invalid/v1",
        api_key="secret",
        model="openai/muse-spark-1.3-contributor",
        messages=[{"role": "user", "content": "hello"}],
        generation_kwargs={"max_tokens": 8192, "seed": None, "transport_retries": 0},
    )
    assert text == "code" and usage["total_tokens"] == 7
    assert usage["raw_response"] == raw
    assert usage["reasoning_content"] == "summary"
    assert "secret" not in str(usage)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_response_http_errors_reach_ledger(monkeypatch, status):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(status, json={}))
    with pytest.raises(litellm.APIError if status != 429 else litellm.RateLimitError):
        query(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="muse-spark-1.3-contributor",
            messages=[],
            generation_kwargs={},
        )


def test_unknown_generation_settings_not_silently_dropped():
    with pytest.raises(ValueError, match="Unmapped"):
        request_payload(
            "muse-spark-1.3-contributor", [], {"extra_body": {"thinking": True}}
        )
