"""Tests for OptionsLLMLifecycle — the cache + cleanup wrapper around the
options agent's LLM client.

The lifecycle exists to fix a resource leak: bot_engine previously called
``_options_llm_adapter()`` on every 3-hour options decision cycle, and that
method was creating a fresh ``AsyncOpenRouterClient`` per call. Each new
client lazily allocated an aiohttp session that was never explicitly
closed, so sessions accumulated forever. The lifecycle wrapper caches a
single instance for the bot's lifetime and exposes a ``close()`` method
that bot_engine.stop() awaits."""

import logging

import pytest

from src.backend.agent.options_llm_lifecycle import OptionsLLMLifecycle
from src.backend.llm_client import AsyncOpenRouterClient


_LOGGER = logging.getLogger("test_options_llm_lifecycle")


# ---------------------------------------------------------------------------
# Cached client (real LLM path)
# ---------------------------------------------------------------------------


def test_get_returns_same_instance_across_calls_when_api_key_set():
    """Calling get() repeatedly must return the SAME AsyncOpenRouterClient
    instance — that's the whole point of the cache."""
    lifecycle = OptionsLLMLifecycle(api_key="sk-test", logger=_LOGGER)
    first = lifecycle.get()
    second = lifecycle.get()
    third = lifecycle.get()
    assert first is second
    assert second is third
    assert isinstance(first, AsyncOpenRouterClient)


def test_get_returns_shim_when_api_key_missing():
    """No key → return a shim object that satisfies the chat_json contract."""
    lifecycle = OptionsLLMLifecycle(api_key=None, logger=_LOGGER)
    shim = lifecycle.get()
    assert hasattr(shim, "chat_json")


def test_get_returns_same_shim_across_calls():
    """The shim is cached too — no need to recreate it on every call."""
    lifecycle = OptionsLLMLifecycle(api_key="", logger=_LOGGER)
    first = lifecycle.get()
    second = lifecycle.get()
    assert first is second


@pytest.mark.asyncio
async def test_shim_returns_empty_decisions_with_warning(caplog):
    lifecycle = OptionsLLMLifecycle(api_key=None, logger=_LOGGER)
    shim = lifecycle.get()
    with caplog.at_level(logging.WARNING):
        result = await shim.chat_json(
            system_prompt="x", user_prompt="y", schema={"type": "object"}
        )
    assert result == {"reasoning": "shim", "trade_decisions": []}


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_awaits_real_client_close():
    """When the lifecycle owns a real client, close() must await its close."""
    lifecycle = OptionsLLMLifecycle(api_key="sk-test", logger=_LOGGER)
    client = lifecycle.get()

    closed = {"n": 0}

    async def fake_close():
        closed["n"] += 1

    client.close = fake_close  # type: ignore[assignment]

    await lifecycle.close()
    assert closed["n"] == 1


@pytest.mark.asyncio
async def test_close_is_a_noop_for_shim():
    """The shim has no resources, so close() must succeed silently."""
    lifecycle = OptionsLLMLifecycle(api_key=None, logger=_LOGGER)
    lifecycle.get()
    # Must not raise.
    await lifecycle.close()


@pytest.mark.asyncio
async def test_close_is_idempotent():
    """Calling close() twice must not error and must not double-close."""
    lifecycle = OptionsLLMLifecycle(api_key="sk-test", logger=_LOGGER)
    client = lifecycle.get()

    closed = {"n": 0}

    async def fake_close():
        closed["n"] += 1

    client.close = fake_close  # type: ignore[assignment]

    await lifecycle.close()
    await lifecycle.close()
    assert closed["n"] == 1  # second call must be a no-op


@pytest.mark.asyncio
async def test_close_swallows_close_exceptions():
    """A failure inside the underlying client's close() must be logged but not raised."""
    lifecycle = OptionsLLMLifecycle(api_key="sk-test", logger=_LOGGER)
    client = lifecycle.get()

    async def boom():
        raise RuntimeError("simulated session teardown failure")

    client.close = boom  # type: ignore[assignment]
    # Must not raise.
    await lifecycle.close()
