"""Async OpenRouter LLM client.

Thin async wrapper around the OpenRouter chat-completions endpoint that
exposes a single ``chat_json`` coroutine — the interface
:class:`OptionsAgent` expects. Sits next to the existing sync
:class:`TradingAgent` (which still drives the perps loop); future PRs can
migrate the perps agent to this client too if we want one async transport
for both pipelines.

Design notes
------------
- **Defensive**: any HTTP failure or JSON-parse error returns ``{}`` so the
  agent layer can fall back gracefully without raising.
- **Structured outputs**: builds the OpenRouter ``response_format``
  ``json_schema`` block from the supplied schema. If the response carries a
  ``parsed`` field (some providers return it directly), we prefer that over
  re-parsing the ``content`` string.
- **No SDK dependency**: uses ``aiohttp`` (already in requirements.txt for
  the rest of the bot) instead of pulling another LLM SDK.
- **Lazy session**: the aiohttp session is built on first request so
  construction is cheap and side-effect free in tests.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.backend.config_loader import CONFIG


logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class AsyncOpenRouterClient:
    """Async wrapper around OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        referer: Optional[str] = None,
        app_title: Optional[str] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or CONFIG.get("openrouter_api_key")
        self.base_url = base_url or CONFIG.get("openrouter_base_url") or _DEFAULT_BASE_URL
        self.model = model or CONFIG.get("llm_model") or "x-ai/grok-4"
        self.referer = referer or CONFIG.get("openrouter_referer")
        self.app_title = app_title or CONFIG.get("openrouter_app_title")
        self.timeout_seconds = timeout_seconds
        self._session = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:  # pylint: disable=broad-except
                pass
            self._session = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def _post(self, url: str, json_body: dict, headers: dict, timeout: float) -> dict:
        """Issue a POST against the chat-completions endpoint and return parsed JSON."""
        import aiohttp

        session = await self._ensure_session()
        async with session.post(
            url,
            json=json_body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body_text = await resp.text()
                logger.warning("OpenRouter returned HTTP %s: %s", resp.status, body_text[:500])
                return {}
            return await resp.json()

    # ------------------------------------------------------------------
    # Public API — matches OptionsAgent's expected llm.chat_json contract
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """Send a chat completion + parse the assistant's JSON response.

        Args:
            system_prompt: the system role content (agent role / rules).
            user_prompt: the user role content — typically a serialized
                context dict.
            schema: JSON schema for structured-output mode. Becomes the
                OpenRouter ``response_format.json_schema.schema`` field.

        Returns:
            Parsed dict the LLM returned. ``{}`` on any failure.
        """
        if not self.api_key:
            logger.warning("AsyncOpenRouterClient: missing OPENROUTER_API_KEY")
            return {}

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "options_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0.4,
        }

        try:
            response = await self._post(url, body, headers, self.timeout_seconds)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("OpenRouter request failed: %s", exc)
            return {}

        return _parse_response(response)


def _parse_response(response) -> dict:
    """Pull the JSON payload out of an OpenRouter chat-completions response."""
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return {}

    # Prefer the structured-output `parsed` field when the provider supplies it.
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        loaded = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        logger.warning("OpenRouter response content was not valid JSON: %s", content[:200])
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded
