"""Public Deribit JSON-RPC client (read-only).

Wraps the small subset of the public Deribit API we need for cross-venue IV
mispricing scans:

- ``public/get_instruments``         — list active BTC option instruments
- ``public/get_book_summary_by_currency`` — mark/IV/best-bid/best-ask snapshot
- ``public/get_index_price``         — BTC index price (settlement reference)

Everything is unauthenticated; the only thing we have to be careful about is
Deribit's rate limit (~20 req/s on public). To stay safely under that and to
avoid hammering the wire when the options agent runs every few minutes, all
responses go through a TTL cache (default 15 minutes — same cadence as the
vol-surface refresh).

Construction is cheap and side-effect free; the aiohttp session is built
lazily on first request and torn down on ``close()``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


DERIBIT_PROD_BASE_URL = "https://www.deribit.com/api/v2"
DERIBIT_TEST_BASE_URL = "https://test.deribit.com/api/v2"

_DEFAULT_CACHE_TTL_SECONDS = 900  # 15 minutes
_DEFAULT_TIMEOUT_SECONDS = 10.0


class DeribitPublicClient:
    """Read-only HTTP client for Deribit public market-data endpoints."""

    def __init__(
        self,
        network: str = "prod",
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = DERIBIT_TEST_BASE_URL if network == "test" else DERIBIT_PROD_BASE_URL
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._session = None
        self._cache: dict[tuple, tuple[float, Any]] = {}

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
        """Lazily build the aiohttp session on first use."""
        if self._session is None:
            import aiohttp  # local import keeps construction free of network deps in tests

            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Issue a GET against the public REST endpoint and return the parsed JSON."""
        import aiohttp

        session = await self._ensure_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.get(
                url,
                params=params or {},
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Deribit %s returned HTTP %s", path, resp.status)
                    return {}
                return await resp.json()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Deribit %s request failed: %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: tuple) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if (time.monotonic() - ts) >= self.cache_ttl_seconds:
            return None
        return payload

    def _cache_put(self, key: tuple, payload: Any) -> None:
        self._cache[key] = (time.monotonic(), payload)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
    ) -> list[dict]:
        """Return the list of active instruments matching the filter."""
        cache_key = ("instruments", currency, kind)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {"currency": currency, "kind": kind, "expired": "false"}
        payload = await self._get("/public/get_instruments", params=params)
        result = payload.get("result") if isinstance(payload, dict) else None
        instruments = result if isinstance(result, list) else []
        self._cache_put(cache_key, instruments)
        return instruments

    async def get_book_summary_by_currency(
        self,
        currency: str = "BTC",
        kind: str = "option",
    ) -> list[dict]:
        """Return the per-instrument book summary (mark, IV, bid/ask) for a currency."""
        cache_key = ("book_summary", currency, kind)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {"currency": currency, "kind": kind}
        payload = await self._get("/public/get_book_summary_by_currency", params=params)
        result = payload.get("result") if isinstance(payload, dict) else None
        summaries = result if isinstance(result, list) else []
        self._cache_put(cache_key, summaries)
        return summaries

    async def get_index_price(self, index_name: str = "btc_usd") -> float:
        """Return the current index spot price (e.g. ``btc_usd``)."""
        cache_key = ("index_price", index_name)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {"index_name": index_name}
        payload = await self._get("/public/get_index_price", params=params)
        result = payload.get("result") if isinstance(payload, dict) else {}
        index_price = float(result.get("index_price", 0.0) or 0.0) if isinstance(result, dict) else 0.0
        self._cache_put(cache_key, index_price)
        return index_price
