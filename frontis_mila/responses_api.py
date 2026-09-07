"""Stateless OpenCode Responses transport, with no hidden retries or tool execution."""

from __future__ import annotations
import os
import time
import httpx


def request_payload(model, messages, generation_kwargs):
    options = dict(generation_kwargs)
    allowed = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "request_timeout_seconds",
        "transport_retries",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unmapped Responses generation settings: {sorted(unknown)}")
    if options.get("seed") is not None:
        raise ValueError("This Responses adapter does not claim model seed support")
    if options.get("transport_retries", 0) != 0:
        raise ValueError("Retries must be owned by the request ledger")
    payload = {
        "model": model.removeprefix("openai/"),
        "input": messages,
        "stream": False,
        "max_output_tokens": int(options.get("max_tokens", 8192)),
    }
    for field in ("temperature", "top_p"):
        if options.get(field) is not None:
            payload[field] = options[field]
    return payload


def query(*, base_url, api_key, model, messages, generation_kwargs):
    import litellm

    payload = request_payload(model, messages, generation_kwargs)
    timeout = float(generation_kwargs.get("request_timeout_seconds", 300))
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "frontis-mila/0.1.0"}
    session = os.environ.get("FRONTIS_RUN_DIR")
    if session:
        import hashlib

        headers["x-opencode-session"] = hashlib.sha256(session.encode()).hexdigest()
    started = time.monotonic()
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/responses",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise litellm.Timeout(
            message="Responses request timed out", model=model, llm_provider="openai"
        ) from exc
    except httpx.RequestError as exc:
        raise litellm.APIConnectionError(
            message="Responses connection failed", model=model, llm_provider="openai"
        ) from exc
    if response.status_code >= 400:
        if response.status_code == 429:
            raise litellm.RateLimitError(
                message="Responses rate limit", model=model, llm_provider="openai"
            )
        raise litellm.APIError(
            status_code=response.status_code,
            message="Responses HTTP failure",
            model=model,
            llm_provider="openai",
        )
    raw = response.json()
    if raw.get("status") not in ("completed", "incomplete"):
        raise ValueError(f'Nonterminal or failed Responses result: {raw.get("status")}')
    texts = []
    reasoning = []
    for item in raw.get("output", []):
        if item.get("type") == "message":
            texts.extend(
                c["text"]
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            )
        elif item.get("type") == "reasoning":
            reasoning.extend(
                c["text"] for c in item.get("summary", []) if c.get("text")
            )
    usage = raw.get("usage") or {}
    stats = {
        "response_id": raw.get("id"),
        "response_model": raw.get("model"),
        "finish_reason": raw.get("status"),
        "incomplete_details": raw.get("incomplete_details"),
        "raw_response": raw,
        "effective_request_parameters": payload,
        "api_transport": "responses",
        "latency": time.monotonic() - started,
        "success": True,
        "token_usage_source": (
            "provider"
            if "input_tokens" in usage and "output_tokens" in usage
            else "unavailable"
        ),
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_content": "\n".join(reasoning),
        "prompt_tokens_details": usage.get("input_tokens_details"),
        "completion_tokens_details": usage.get("output_tokens_details"),
    }
    # Keep an empty/incomplete response in the ledger so extraction records the failure.
    return "".join(texts), stats
