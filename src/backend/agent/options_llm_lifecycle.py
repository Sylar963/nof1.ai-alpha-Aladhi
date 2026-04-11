"""Cache + cleanup wrapper around the options agent's LLM client.

The bot's options decision loop runs on a 3-hour cadence. Before this
helper, ``bot_engine._options_llm_adapter()`` constructed a fresh
:class:`AsyncOpenRouterClient` on every call. Each new client lazily
allocated an aiohttp session on first request and never explicitly closed
it, so sessions accumulated forever — a slow but real resource leak that
would eventually exhaust file descriptors on a long-running deployment.

This module owns a single client instance for the bot's lifetime and
exposes a clean shutdown path so :py:meth:`bot_engine.stop` can release
the underlying aiohttp session deterministically.

Two paths:

- **Real client path**: when an OpenRouter API key is configured the
  lifecycle constructs and caches an :class:`AsyncOpenRouterClient`. Every
  ``get()`` returns the same instance. ``close()`` awaits the client's
  ``close()``, which tears down the aiohttp session.

- **Shim path**: when no key is configured the lifecycle returns a tiny
  in-memory shim with the same ``chat_json`` interface that logs a
  warning and returns an empty decision set. The shim has no resources,
  so ``close()`` is a no-op.

``close()`` is idempotent and swallows exceptions raised by the underlying
client — losing a session at shutdown should never crash the bot.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


class _ShimLLM:
    """Stand-in returned when no OPENROUTER_API_KEY is configured.

    Implements the ``chat_json`` interface :class:`OptionsAgent` expects so
    the scheduler can keep ticking without crashing on a missing key. Every
    cycle that hits this path logs a warning and returns an empty decision
    set so the bot stays idle until the operator wires a real key.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    async def chat_json(self, *, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        self._logger.warning(
            "OPENROUTER_API_KEY missing — OptionsAgent returning empty decisions"
        )
        return {"reasoning": "shim", "trade_decisions": []}


class OptionsLLMLifecycle:
    """Owns the cached LLM client (or shim) used by OptionsAgent.

    Construction is cheap and side-effect free in the shim path. The real
    path imports :class:`AsyncOpenRouterClient` lazily so test environments
    that don't have OpenRouter configured never load it.
    """

    def __init__(self, api_key: Optional[str], logger: logging.Logger) -> None:
        self._logger = logger
        self._closed = False
        if api_key:
            from src.backend.llm_client import AsyncOpenRouterClient

            self._instance: Any = AsyncOpenRouterClient()
            self._needs_close = True
        else:
            self._instance = _ShimLLM(logger)
            self._needs_close = False

    def get(self) -> Any:
        """Return the cached client or shim. Same instance on every call."""
        return self._instance

    async def close(self) -> None:
        """Tear down the underlying client. Idempotent and exception-safe.

        Calling close() twice is fine — the second call is a no-op. If the
        client's close() raises, the exception is logged at warning level
        and swallowed so a teardown failure can't crash the bot's stop()
        path.
        """
        if self._closed:
            return
        self._closed = True
        if not self._needs_close:
            return
        close_coro = getattr(self._instance, "close", None)
        if close_coro is None:
            return
        try:
            await close_coro()
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning("OptionsLLMLifecycle close failed: %s", exc)
