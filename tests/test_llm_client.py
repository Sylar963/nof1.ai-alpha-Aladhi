"""Tests for the AsyncOpenRouterClient.

The client wraps OpenRouter's chat completions endpoint with an async API
that conforms to OptionsAgent's expected ``chat_json`` interface. Tests
mock the underlying HTTP transport so they're fast and offline-friendly."""

import json

import pytest

from src.backend.llm_client import AsyncOpenRouterClient


def _fake_post_returning(payload):
    """Return an async stub that records POST requests and returns a fixed body."""
    sent: list[dict] = []

    async def _stub(url, json_body, headers, timeout):
        sent.append({"url": url, "json_body": json_body, "headers": headers, "timeout": timeout})
        return payload

    _stub.sent = sent
    return _stub


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_client_constructs_with_explicit_args():
    client = AsyncOpenRouterClient(
        api_key="sk-test",
        base_url="https://example.test/api/v1",
        model="x-ai/grok-4",
    )
    assert client.api_key == "sk-test"
    assert client.base_url == "https://example.test/api/v1"
    assert client.model == "x-ai/grok-4"


def test_client_falls_back_to_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    from src.backend import config_loader
    monkeypatch.setitem(config_loader.CONFIG, "openrouter_api_key", "sk-from-env")
    monkeypatch.setitem(config_loader.CONFIG, "openrouter_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setitem(config_loader.CONFIG, "llm_model", "x-ai/grok-4")
    client = AsyncOpenRouterClient()
    assert client.api_key == "sk-from-env"
    assert client.model == "x-ai/grok-4"


# ---------------------------------------------------------------------------
# chat_json round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_json_returns_parsed_response_message():
    client = AsyncOpenRouterClient(api_key="sk-test", model="x-ai/grok-4")
    expected_payload = {"reasoning": "ok", "trade_decisions": []}
    fake = _fake_post_returning({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(expected_payload),
                }
            }
        ]
    })
    client._post = fake  # type: ignore[assignment]

    result = await client.chat_json(
        system_prompt="You are an options trader.",
        user_prompt='{"spot": 60000}',
        schema={"type": "object"},
    )

    assert result == expected_payload
    assert len(fake.sent) == 1
    sent_body = fake.sent[0]["json_body"]
    assert sent_body["model"] == "x-ai/grok-4"
    assert sent_body["messages"][0]["role"] == "system"
    assert sent_body["messages"][0]["content"] == "You are an options trader."
    assert sent_body["messages"][1]["role"] == "user"
    assert sent_body["messages"][1]["content"] == '{"spot": 60000}'


@pytest.mark.asyncio
async def test_chat_json_prefers_parsed_field_when_present():
    """OpenRouter structured outputs return a 'parsed' field — prefer it over content."""
    client = AsyncOpenRouterClient(api_key="sk-test")
    expected_payload = {"reasoning": "structured", "trade_decisions": []}
    fake = _fake_post_returning({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "parsed": expected_payload,
                    "content": "should be ignored",
                }
            }
        ]
    })
    client._post = fake  # type: ignore[assignment]

    result = await client.chat_json(
        system_prompt="x", user_prompt="y", schema={"type": "object"},
    )
    assert result == expected_payload


@pytest.mark.asyncio
async def test_chat_json_returns_empty_dict_on_invalid_json():
    """A response with non-JSON content must NOT raise — return {} so callers fall back."""
    client = AsyncOpenRouterClient(api_key="sk-test")
    fake = _fake_post_returning({
        "choices": [{"message": {"role": "assistant", "content": "not valid json"}}]
    })
    client._post = fake  # type: ignore[assignment]
    result = await client.chat_json(
        system_prompt="x", user_prompt="y", schema={"type": "object"},
    )
    assert result == {}


@pytest.mark.asyncio
async def test_chat_json_returns_empty_dict_on_http_failure():
    """Transport-level errors are swallowed and the client returns an empty dict."""
    client = AsyncOpenRouterClient(api_key="sk-test")

    async def _failing(url, json_body, headers, timeout):
        raise RuntimeError("connection refused")

    client._post = _failing  # type: ignore[assignment]
    result = await client.chat_json(
        system_prompt="x", user_prompt="y", schema={"type": "object"},
    )
    assert result == {}


@pytest.mark.asyncio
async def test_chat_json_includes_response_format_for_structured_output():
    """The OpenRouter request body must declare json_schema response format."""
    client = AsyncOpenRouterClient(api_key="sk-test", model="x-ai/grok-4")
    fake = _fake_post_returning({
        "choices": [{"message": {"role": "assistant", "content": "{}"}}]
    })
    client._post = fake  # type: ignore[assignment]

    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    await client.chat_json(system_prompt="x", user_prompt="y", schema=schema)

    body = fake.sent[0]["json_body"]
    assert "response_format" in body
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema
