"""Thalex options exchange adapter.

Wraps the official ``thalex`` Python library to fit the project's
:class:`ExchangeAdapter` contract. Thalex is a JSON-RPC over WebSocket API,
which means this adapter owns:

- a persistent WS connection
- a background receive loop
- a request/response correlation table keyed by JSON-RPC ``id``
- async resolution of pending futures when responses arrive
- JWT RS512 login on every (re)connect
- an in-memory cache of the instrument list (for intent → instrument resolution)

This module deliberately does NOT import or open the WebSocket at module load.
Construction is cheap and side-effect-free; ``connect()`` is the lifecycle hook
that actually establishes the connection. That keeps unit tests fast and lets
the bot engine fail loudly with clear error messages when credentials are
missing.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from src.backend.config_loader import CONFIG
from src.backend.trading.exchange_adapter import (
    AccountState,
    ExchangeAdapter,
    OrderResult,
    PositionSnapshot,
)
from src.backend.trading.options import (
    OptionIntent,
    RiskCaps,
    find_best_instrument,
    parse_instrument_name,
    validate_options_order,
)


logger = logging.getLogger(__name__)


def _parse_underlyings(raw: str) -> list[str]:
    """Split a comma-separated underlyings string into a clean list."""
    if not raw:
        return ["BTC"]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


class ThalexAPI(ExchangeAdapter):
    """ExchangeAdapter implementation backed by the official ``thalex`` library."""

    venue = "thalex"

    def __init__(
        self,
        network: Optional[str] = None,
        key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        account: Optional[str] = None,
        risk_caps: Optional[RiskCaps] = None,
    ) -> None:
        self.network_name = (
            network
            or os.getenv("THALEX_NETWORK")
            or CONFIG.get("thalex_network")
            or "test"
        ).lower()
        if self.network_name not in {"test", "prod"}:
            raise ValueError(
                f"THALEX_NETWORK must be 'test' or 'prod', got {self.network_name!r}"
            )
        self.key_id = key_id or os.getenv("THALEX_KEY_ID") or CONFIG.get("thalex_key_id")
        self.private_key_path = (
            private_key_path
            or os.getenv("THALEX_PRIVATE_KEY_PATH")
            or CONFIG.get("thalex_private_key_path")
        )
        self.account = account or os.getenv("THALEX_ACCOUNT") or CONFIG.get("thalex_account")
        self.risk_caps = risk_caps or RiskCaps(
            max_contracts_per_trade=float(
                os.getenv("THALEX_MAX_CONTRACTS_PER_TRADE")
                or CONFIG.get("thalex_max_contracts_per_trade")
                or 0.1
            ),
            max_open_positions=int(
                os.getenv("THALEX_MAX_OPEN_POSITIONS")
                or CONFIG.get("thalex_max_open_positions")
                or 3
            ),
            allowed_underlyings=_parse_underlyings(
                os.getenv("THALEX_UNDERLYINGS")
                or ",".join(CONFIG.get("thalex_underlyings") or ["BTC"])
            ),
        )

        self._client = None  # constructed lazily in connect()
        self._id_counter = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._receiver_task: Optional[asyncio.Task] = None
        self._instruments_cache: list[dict] = []
        self._lock = asyncio.Lock()
        self.connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the WebSocket, login with JWT, and start the receive loop."""
        if self.connected:
            return
        async with self._lock:
            if self.connected:
                return
            from thalex import Thalex, Network  # local import keeps test imports light

            if not self.key_id:
                raise RuntimeError("THALEX_KEY_ID is not set")
            if not self.private_key_path or not Path(self.private_key_path).exists():
                raise RuntimeError(
                    f"THALEX_PRIVATE_KEY_PATH not set or file missing: {self.private_key_path}"
                )

            net = Network.TEST if self.network_name == "test" else Network.PROD
            self._client = Thalex(network=net)
            await self._client.connect()

            self._receiver_task = asyncio.create_task(self._receiver_loop())

            private_key = Path(self.private_key_path).read_text()
            login_id = self._next_id()
            login_future = self._make_future(login_id)
            await self._client.login(
                key_id=self.key_id,
                private_key=private_key,
                account=self.account,
                id=login_id,
            )
            await asyncio.wait_for(login_future, timeout=10.0)

            self.connected = True
            logger.info("Thalex connected on %s", self.network_name)

            await self._refresh_instruments_cache()

    async def disconnect(self) -> None:
        """Tear down the receive loop and close the WebSocket."""
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                pass
            self._receiver_task = None
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # pylint: disable=broad-except
                pass
        self.connected = False

    # ------------------------------------------------------------------
    # Internal: id allocation, futures, receive loop
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        return next(self._id_counter)

    def _make_future(self, request_id: int) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[request_id] = fut
        return fut

    async def _receiver_loop(self) -> None:
        """Forever loop pulling messages from the Thalex WS and resolving futures."""
        assert self._client is not None
        try:
            while True:
                raw = await self._client.receive()
                try:
                    message = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, json.JSONDecodeError):
                    logger.warning("Thalex receiver: undecodable message: %r", raw)
                    continue
                self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Thalex receiver loop crashed: %s", exc)
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)
            self._pending.clear()

    def _dispatch(self, message: dict) -> None:
        """Resolve a pending future or log an async event."""
        if not isinstance(message, dict):
            return
        msg_id = message.get("id")
        if msg_id is not None and msg_id in self._pending:
            fut = self._pending.pop(msg_id)
            if "error" in message and message["error"]:
                fut.set_exception(RuntimeError(f"Thalex error: {message['error']}"))
            else:
                fut.set_result(message.get("result"))
            return
        # Async event (subscription update, fill, etc.)
        method = message.get("method") or message.get("channel")
        if method:
            logger.debug("Thalex async event %s: %s", method, message)

    async def _request(self, sender, **kwargs):
        """Generic request helper used by every public RPC wrapper.

        ``sender`` is a callable that takes the kwargs (including ``id``) and
        invokes a method on the underlying thalex client.
        """
        if not self.connected:
            await self.connect()
        request_id = self._next_id()
        fut = self._make_future(request_id)
        try:
            await sender(id=request_id, **kwargs)
        except Exception:
            self._pending.pop(request_id, None)
            raise
        return await asyncio.wait_for(fut, timeout=15.0)

    async def _refresh_instruments_cache(self) -> None:
        try:
            result = await self._request(self._client.instruments)
            if isinstance(result, list):
                self._instruments_cache = result
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to refresh Thalex instruments cache: %s", exc)

    # ------------------------------------------------------------------
    # Intent → instrument resolution
    # ------------------------------------------------------------------

    async def resolve_intent(self, intent: OptionIntent) -> Optional[str]:
        """Pick the instrument that best matches an LLM-emitted OptionIntent."""
        if not self._instruments_cache:
            await self._refresh_instruments_cache()
        return find_best_instrument(self._instruments_cache, intent)

    def preflight(self, underlying: str, contracts: float, open_positions_count: int) -> tuple[bool, str]:
        """Apply hard risk caps before any order is sent to Thalex."""
        return validate_options_order(
            underlying=underlying,
            contracts=contracts,
            open_positions_count=open_positions_count,
            caps=self.risk_caps,
        )

    # ------------------------------------------------------------------
    # ExchangeAdapter implementation
    # ------------------------------------------------------------------

    async def place_buy_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        from thalex import Direction, OrderType

        result = await self._request(
            self._client.insert,
            direction=Direction.BUY,
            instrument_name=asset,
            amount=amount,
            order_type=OrderType.MARKET,
        )
        return self._make_order_result(asset, "buy", amount, result)

    async def place_sell_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        from thalex import Direction, OrderType

        result = await self._request(
            self._client.insert,
            direction=Direction.SELL,
            instrument_name=asset,
            amount=amount,
            order_type=OrderType.MARKET,
        )
        return self._make_order_result(asset, "sell", amount, result)

    async def place_take_profit(self, asset: str, is_buy: bool, amount: float, tp_price: float) -> OrderResult:
        """Thalex does not support trigger orders for options.

        For v1, the strategy layer is responsible for monitoring price and
        submitting an exit limit order. We return a sentinel OrderResult so the
        bot engine can log this gracefully without crashing.
        """
        return OrderResult(
            venue=self.venue,
            order_id="",
            asset=asset,
            side="tp",
            amount=amount,
            status="not_supported",
            error="Thalex options have no native TP triggers; handled by strategy layer",
        )

    async def place_stop_loss(self, asset: str, is_buy: bool, amount: float, sl_price: float) -> OrderResult:
        return OrderResult(
            venue=self.venue,
            order_id="",
            asset=asset,
            side="sl",
            amount=amount,
            status="not_supported",
            error="Thalex options have no native SL triggers; handled by strategy layer",
        )

    async def cancel_order(self, asset: str, order_id: Any) -> dict:
        result = await self._request(self._client.cancel, order_id=str(order_id))
        return {"status": "ok", "result": result}

    async def cancel_all_orders(self, asset: str) -> dict:
        result = await self._request(self._client.cancel_all)
        return {"status": "ok", "result": result}

    async def get_open_orders(self) -> list[dict]:
        result = await self._request(self._client.open_orders)
        return result if isinstance(result, list) else []

    async def get_recent_fills(self, limit: int = 50) -> list[dict]:
        result = await self._request(self._client.trade_history, limit=limit)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and isinstance(result.get("trades"), list):
            return result["trades"][-limit:]
        return []

    async def get_user_state(self) -> AccountState:
        summary = await self._request(self._client.account_summary)
        portfolio = await self._request(self._client.portfolio)

        balance = float((summary or {}).get("equity", 0.0) or 0.0)
        total_value = float((summary or {}).get("portfolio_value", balance) or balance)

        positions: list[PositionSnapshot] = []
        if isinstance(portfolio, list):
            for entry in portfolio:
                instrument_name = entry.get("instrument_name") or ""
                size = float(entry.get("position", entry.get("amount", 0.0)) or 0.0)
                if size == 0:
                    continue
                spec = parse_instrument_name(instrument_name)
                underlying = spec.underlying if spec else instrument_name.split("-", 1)[0]
                positions.append(
                    PositionSnapshot(
                        venue=self.venue,
                        asset=underlying,
                        instrument_name=instrument_name,
                        side="long" if size > 0 else "short",
                        size=abs(size),
                        entry_price=float(entry.get("average_price", 0.0) or 0.0),
                        current_price=float(entry.get("mark_price", 0.0) or 0.0),
                        unrealized_pnl=float(entry.get("unrealized_pnl", 0.0) or 0.0),
                        delta=float(entry.get("delta", 0.0) or 0.0) if "delta" in entry else None,
                        raw=entry,
                    )
                )

        return AccountState(
            venue=self.venue,
            balance=balance,
            total_value=total_value,
            positions=positions,
            raw={"summary": summary, "portfolio": portfolio},
        )

    async def get_current_price(self, asset: str) -> float:
        result = await self._request(self._client.ticker, instrument_name=asset)
        if isinstance(result, dict):
            for key in ("mark_price", "last_price", "best_bid", "best_ask"):
                value = result.get(key)
                if value is not None:
                    return float(value)
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_order_result(self, asset: str, side: str, amount: float, raw: Any) -> OrderResult:
        order_id = ""
        status = "ok"
        if isinstance(raw, dict):
            order_id = str(raw.get("order_id") or raw.get("id") or "")
            status = str(raw.get("status") or "ok")
        return OrderResult(
            venue=self.venue,
            order_id=order_id,
            asset=asset,
            side=side,
            amount=amount,
            status=status,
            instrument_name=asset,
            raw=raw if isinstance(raw, dict) else None,
        )
