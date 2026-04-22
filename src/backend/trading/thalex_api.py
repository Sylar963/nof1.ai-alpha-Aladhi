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
import time
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


# Greeks field paths the defensive parser tries, in order. The Thalex SDK
# does not document the exact ticker response shape, so this list covers the
# three plausible layouts (top-level, nested under "greeks", nested under
# the singular "greek"). If a future probe of the live response surfaces a
# new path, append it here — get_greeks needs no other changes.
_GREEKS_FIELD_PATHS: tuple[tuple[str, ...], ...] = (
    (),                # top level: result["delta"]
    ("greeks",),       # Deribit-style: result["greeks"]["delta"]
    ("greek",),        # singular variant
    ("data",),         # some venues wrap everything under "data"
)
_GREEKS_KEYS: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "mark_iv")
_GREEKS_TTL_SECONDS: float = 5.0
_RETRY_MAX_ATTEMPTS: int = 3
_RETRY_BACKOFF_BASE_SECONDS: float = 0.5


def _parse_underlyings(raw: str) -> list[str]:
    """Split a comma-separated underlyings string into a clean list."""
    if not raw:
        return ["BTC"]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _expand_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    expanded = os.path.expandvars(value)
    return Path(expanded).expanduser()


def _normalize_private_key(raw: str) -> str:
    cleaned = raw.lstrip("\ufeff").strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in cleaned and "\n" not in cleaned:
        cleaned = cleaned.replace("\\n", "\n")
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned


def _instrument_from_channel(channel: str) -> Optional[str]:
    """Extract the option instrument name from a Thalex ticker channel string.

    Channel formats observed across exchanges of this style:
      ``ticker.BTC-10MAY26-65000-C``
      ``ticker.BTC-10MAY26-65000-C.raw``
      ``ticker:BTC-10MAY26-65000-C``
    Returns None if no instrument can be parsed.
    """
    if not isinstance(channel, str) or not channel:
        return None
    parts = channel.replace(":", ".").split(".")
    if len(parts) < 2 or parts[0] != "ticker":
        return None
    candidate = parts[1]
    return candidate or None


def _extract_greeks(payload: Any) -> dict:
    """Pull greek fields out of a ticker payload regardless of nesting layout.

    Tries each path in :data:`_GREEKS_FIELD_PATHS` and merges any keys it
    finds. Numeric coercion is best-effort; values that don't parse as floats
    are silently dropped so a malformed sub-field doesn't poison the rest.
    """
    if not isinstance(payload, dict):
        return {}
    found: dict = {}
    for path in _GREEKS_FIELD_PATHS:
        node: Any = payload
        for segment in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(segment)
        if not isinstance(node, dict):
            continue
        for key in _GREEKS_KEYS:
            if key in found:
                continue
            value = node.get(key)
            if value is None:
                continue
            try:
                found[key] = float(value)
            except (TypeError, ValueError):
                continue
    return found


def _quote_field(payload: Any, *paths: tuple[str, ...]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for path in paths:
        node: Any = payload
        for segment in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(segment)
        if node is None:
            continue
        try:
            return float(node)
        except (TypeError, ValueError):
            continue
    return None


def _stable_asset_multiplier(currency: str, index_price: Any) -> Optional[float]:
    if index_price is not None:
        try:
            return float(index_price)
        except (TypeError, ValueError):
            return None
    if str(currency or "").upper() in {"USD", "USDC", "USDT", "DAI"}:
        return 1.0
    return None


def _summary_amount(summary: dict, *keys: str) -> Optional[float]:
    for key in keys:
        if key not in summary:
            continue
        try:
            return float(summary.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return None


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
        raw_network = (
            network
            or os.getenv("THALEX_NETWORK")
            or CONFIG.get("thalex_network")
            or "test"
        )
        self.network_name = (_clean_env_value(raw_network) or "test").lower()
        if self.network_name not in {"test", "prod"}:
            raise ValueError(
                f"THALEX_NETWORK must be 'test' or 'prod', got {self.network_name!r}"
            )
        self.key_id = _clean_env_value(
            key_id or os.getenv("THALEX_KEY_ID") or CONFIG.get("thalex_key_id")
        )
        self.private_key_path = _clean_env_value(
            private_key_path
            or os.getenv("THALEX_PRIVATE_KEY_PATH")
            or CONFIG.get("thalex_private_key_path")
        )
        self.account = _clean_env_value(
            account or os.getenv("THALEX_ACCOUNT") or CONFIG.get("thalex_account")
        )
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
        self._greeks_cache: dict[str, tuple[float, dict]] = {}
        # Async subscriptions: instrument_name → callback. The receive loop
        # routes channel notifications here in _dispatch.
        self._ticker_subscribers: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _is_client_alive(self) -> bool:
        """Check whether the underlying WS client still looks usable.

        ``self.connected`` is a bool we set during login and clear in disconnect,
        but it can drift out of sync with the real WS state (e.g. the receive
        loop crashed with 'no close frame received'). Probe the client when
        possible so callers don't push messages into a dead socket.
        """
        if self._client is None:
            return False
        for attr in ("closed", "is_closed"):
            if hasattr(self._client, attr):
                try:
                    closed = getattr(self._client, attr)
                    if callable(closed):
                        closed = closed()
                    return not bool(closed)
                except Exception:  # pylint: disable=broad-except
                    return False
        receiver = self._receiver_task
        if receiver is not None and receiver.done():
            return False
        return True

    async def connect(self) -> None:
        """Establish the WebSocket, login with JWT, and start the receive loop."""
        if self.connected and self._is_client_alive():
            return
        async with self._lock:
            if self.connected and self._is_client_alive():
                return
            # If we were flagged connected but the socket is dead, tear down
            # stale state before re-establishing. Without this, a zombie
            # connection lingers forever and every request hits the 15s timeout.
            if self.connected or self._client is not None or self._receiver_task is not None:
                await self._teardown_stale_client()

            from thalex import Thalex, Network  # local import keeps test imports light

            if not self.key_id:
                raise RuntimeError("THALEX_KEY_ID is not set")
            key_path = _expand_path(self.private_key_path)
            if not key_path or not key_path.exists():
                raise RuntimeError(
                    f"THALEX_PRIVATE_KEY_PATH not set or file missing: {self.private_key_path}"
                )

            net = Network.TEST if self.network_name == "test" else Network.PROD
            self._client = Thalex(network=net)
            try:
                await asyncio.wait_for(self._client.connect(), timeout=10.0)
            except (asyncio.TimeoutError, Exception) as exc:
                self._client = None
                raise RuntimeError(f"Thalex WebSocket connect failed: {exc}") from exc

            self._receiver_task = asyncio.create_task(self._receiver_loop())

            try:
                private_key_raw = key_path.read_text(encoding="utf-8")
                private_key = _normalize_private_key(private_key_raw)
                login_id = self._next_id()
                login_future = self._make_future(login_id)
                await self._client.login(
                    key_id=self.key_id,
                    private_key=private_key,
                    account=self.account or None,
                    id=login_id,
                )
                await asyncio.wait_for(login_future, timeout=10.0)
            except Exception:
                # Login failed — roll back the partial connect so the next
                # caller retries cleanly instead of trusting a zombie client.
                await self._teardown_stale_client()
                raise

            self.connected = True
            logger.info("Thalex connected on %s", self.network_name)

            await self._refresh_instruments_cache()
            await self._resubscribe_tickers()

    async def _teardown_stale_client(self) -> None:
        """Cancel the receiver and close the client without raising.

        Called when we detect (or suspect) the WS is dead so connect() can
        rebuild from a clean slate. Safe to call when nothing is initialized.
        """
        self.connected = False
        receiver = self._receiver_task
        self._receiver_task = None
        if receiver is not None and not receiver.done():
            receiver.cancel()
            try:
                await receiver
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                pass
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:  # pylint: disable=broad-except
                pass
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("Thalex connection torn down"))
        self._pending.clear()

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
            # Flip the connection flag so the next caller rebuilds instead of
            # trusting a dead socket. Without this, connect() short-circuits on
            # self.connected and every subsequent request times out at 15s.
            self.connected = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)
            self._pending.clear()

    def _dispatch(self, message: dict) -> None:
        """Resolve a pending future, route a subscription update, or log an event.

        Three message classes:
          1. RPC reply with matching ``id`` → resolve the pending future.
          2. JSON-RPC notification (``method == 'subscription'``) carrying a
             ``ticker.{instrument_name}`` channel → invoke the registered
             subscriber callback as a fire-and-forget task so the receive
             loop never blocks waiting on the callback.
          3. Anything else → log at debug.
        """
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

        if message.get("method") == "subscription":
            params = message.get("params") or {}
            channel = params.get("channel") or ""
            payload = params.get("notification")
            instrument_name = _instrument_from_channel(channel)
            if instrument_name and instrument_name in self._ticker_subscribers:
                self.cache_greeks_snapshot(instrument_name, payload)
                callback = self._ticker_subscribers[instrument_name]
                try:
                    asyncio.get_running_loop().create_task(callback(payload))
                except RuntimeError:
                    # No running loop (e.g. test calling _dispatch synchronously
                    # outside an event-loop context). Schedule via asyncio.run.
                    pass
                return

        method = message.get("method") or message.get("channel")
        if method:
            logger.debug("Thalex async event %s: %s", method, message)

    async def subscribe_ticker(self, instrument_name: str, callback) -> None:
        """Subscribe to live ticker updates for an option instrument.

        The callback is invoked on every notification with the parsed
        notification payload. Callbacks run on the receive loop's task, so
        they MUST be non-blocking — spawn ``asyncio.create_task`` for any I/O.
        """
        self._ticker_subscribers[instrument_name] = callback
        if self._client is None:
            return
        channel = f"ticker.{instrument_name}.raw"
        try:
            await self._request(self._client.public_subscribe, channels=[channel])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Thalex subscribe %s failed: %s", channel, exc)

    async def unsubscribe_ticker(self, instrument_name: str) -> None:
        """Cancel a previously registered ticker subscription."""
        self._ticker_subscribers.pop(instrument_name, None)
        if self._client is None:
            return
        channel = f"ticker.{instrument_name}.raw"
        try:
            await self._request(self._client.public_unsubscribe, channels=[channel])
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Thalex unsubscribe %s failed: %s", channel, exc)

    async def _request(self, sender, **kwargs):
        """Generic request helper used by every public RPC wrapper.

        ``sender`` is a callable that takes the kwargs (including ``id``) and
        invokes a method on the underlying thalex client.
        """
        if not self.connected or not self._is_client_alive():
            await self.connect()
        request_id = self._next_id()
        fut = self._make_future(request_id)
        try:
            await sender(id=request_id, **kwargs)
        except Exception:
            self._pending.pop(request_id, None)
            raise
        return await asyncio.wait_for(fut, timeout=15.0)

    async def _request_with_retry(
        self,
        sender,
        *,
        max_attempts: int = _RETRY_MAX_ATTEMPTS,
        backoff_base: float = _RETRY_BACKOFF_BASE_SECONDS,
        description: str = "Thalex request",
        **kwargs,
    ):
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return await self._request(sender, **kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                if attempt < max_attempts - 1:
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying",
                        description,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                else:
                    logger.error("%s failed after %d attempts: %s", description, max_attempts, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{description} retry exited without result or exception")

    async def _refresh_instruments_cache(self) -> None:
        try:
            result = await self._request(self._client.instruments)
            if isinstance(result, list):
                self._instruments_cache = result
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to refresh Thalex instruments cache: %s", exc)

    async def _resubscribe_tickers(self) -> None:
        """Restore ticker subscriptions after a reconnect."""
        if self._client is None or not self._ticker_subscribers:
            return
        for instrument_name in list(self._ticker_subscribers):
            channel = f"ticker.{instrument_name}.raw"
            try:
                await self._request_with_retry(
                    self._client.public_subscribe,
                    channels=[channel],
                    description=f"restore Thalex subscription {channel}",
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to restore Thalex subscription %s: %s", channel, exc)

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

    async def _entry_limit_price(self, instrument_name: str, side: str, slippage: float) -> float:
        ticker = await self._request_with_retry(
            self._client.ticker,
            instrument_name=instrument_name,
            description=f"Thalex ticker {instrument_name}",
        )
        if side == "buy":
            reference = _quote_field(
                ticker,
                ("best_ask",),
                ("ask_price",),
                ("mark_price",),
                ("last_price",),
                ("best_bid",),
            )
            if reference is None or reference <= 0:
                raise RuntimeError(f"No ask/mark quote available for {instrument_name}")
            return reference * (1.0 + max(float(slippage or 0.0), 0.0))

        reference = _quote_field(
            ticker,
            ("best_bid",),
            ("bid_price",),
            ("mark_price",),
            ("last_price",),
            ("best_ask",),
        )
        if reference is None or reference <= 0:
            raise RuntimeError(f"No bid/mark quote available for {instrument_name}")
        return max(reference * (1.0 - max(float(slippage or 0.0), 0.0)), 0.0)

    async def _submit_limit_order(
        self,
        *,
        instrument_name: str,
        amount: float,
        side: str,
        limit_price: float,
        time_in_force=None,
        reduce_only: bool | None = None,
        description: str,
    ) -> OrderResult:
        from thalex import Direction, OrderType, TimeInForce

        direction = Direction.BUY if side == "buy" else Direction.SELL
        tif = time_in_force if time_in_force is not None else TimeInForce.IOC
        result = await self._request_with_retry(
            self._client.insert,
            direction=direction,
            instrument_name=instrument_name,
            amount=amount,
            price=limit_price,
            order_type=OrderType.LIMIT,
            time_in_force=tif,
            reduce_only=reduce_only,
            description=description,
        )
        return self._make_order_result(instrument_name, side, amount, result, submitted_price=limit_price)

    async def _submit_conditional_order(
        self,
        *,
        instrument_name: str,
        amount: float,
        side: str,
        stop_price: float,
        limit_price: Optional[float] = None,
        bracket_price: Optional[float] = None,
        trailing_stop_callback_rate: Optional[float] = None,
        reduce_only: bool = True,
        description: str,
    ) -> OrderResult:
        from thalex import Direction, Target

        direction = Direction.BUY if side == "buy" else Direction.SELL
        result = await self._request_with_retry(
            self._client.create_conditional_order,
            direction=direction,
            instrument_name=instrument_name,
            amount=amount,
            stop_price=float(stop_price),
            limit_price=float(limit_price) if limit_price is not None else None,
            bracket_price=float(bracket_price) if bracket_price is not None else None,
            trailing_stop_callback_rate=(
                float(trailing_stop_callback_rate)
                if trailing_stop_callback_rate is not None
                else None
            ),
            reduce_only=reduce_only,
            target=Target.MARK,
            description=description,
        )
        return self._make_order_result(
            instrument_name,
            side,
            amount,
            result,
            submitted_price=limit_price,
        )

    async def place_buy_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        limit_price = await self._entry_limit_price(asset, "buy", slippage)
        return await self._submit_limit_order(
            instrument_name=asset,
            amount=amount,
            side="buy",
            limit_price=limit_price,
            description=f"Thalex buy {asset}",
        )

    async def place_sell_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        limit_price = await self._entry_limit_price(asset, "sell", slippage)
        return await self._submit_limit_order(
            instrument_name=asset,
            amount=amount,
            side="sell",
            limit_price=limit_price,
            description=f"Thalex sell {asset}",
        )

    async def place_take_profit(self, asset: str, is_buy: bool, amount: float, tp_price: float) -> OrderResult:
        from thalex import TimeInForce

        exit_side = "sell" if is_buy else "buy"
        return await self._submit_limit_order(
            instrument_name=asset,
            amount=amount,
            side=exit_side,
            limit_price=float(tp_price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            description=f"Thalex take-profit {asset}",
        )

    async def place_stop_loss(self, asset: str, is_buy: bool, amount: float, sl_price: float) -> OrderResult:
        exit_side = "sell" if is_buy else "buy"
        return await self._submit_conditional_order(
            instrument_name=asset,
            amount=amount,
            side=exit_side,
            stop_price=float(sl_price),
            reduce_only=True,
            description=f"Thalex stop-loss {asset}",
        )

    async def place_stop_limit_order(
        self,
        asset: str,
        is_buy: bool,
        amount: float,
        stop_price: float,
        limit_price: float,
    ) -> OrderResult:
        exit_side = "sell" if is_buy else "buy"
        return await self._submit_conditional_order(
            instrument_name=asset,
            amount=amount,
            side=exit_side,
            stop_price=float(stop_price),
            limit_price=float(limit_price),
            reduce_only=True,
            description=f"Thalex stop-limit {asset}",
        )

    async def place_bracket_order(
        self,
        asset: str,
        is_buy: bool,
        amount: float,
        stop_price: float,
        bracket_price: float,
    ) -> OrderResult:
        exit_side = "sell" if is_buy else "buy"
        return await self._submit_conditional_order(
            instrument_name=asset,
            amount=amount,
            side=exit_side,
            stop_price=float(stop_price),
            bracket_price=float(bracket_price),
            reduce_only=True,
            description=f"Thalex bracket {asset}",
        )

    async def place_trailing_stop_order(
        self,
        asset: str,
        is_buy: bool,
        amount: float,
        stop_price: float,
        callback_rate: float,
    ) -> OrderResult:
        exit_side = "sell" if is_buy else "buy"
        return await self._submit_conditional_order(
            instrument_name=asset,
            amount=amount,
            side=exit_side,
            stop_price=float(stop_price),
            trailing_stop_callback_rate=float(callback_rate),
            reduce_only=True,
            description=f"Thalex trailing-stop {asset}",
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
        if not self.connected or not self._is_client_alive():
            await self.connect()
        if self._client is None:
            raise RuntimeError("Thalex client not initialized after connect()")

        summary = await self._request(self._client.account_summary)
        portfolio = await self._request(self._client.portfolio)

        summary_dict = summary if isinstance(summary, dict) else {}
        cash_entries = summary_dict.get("cash") if isinstance(summary_dict.get("cash"), list) else []
        derived_cash_collateral = 0.0
        for entry in cash_entries:
            if not isinstance(entry, dict):
                continue
            multiplier = _stable_asset_multiplier(
                str(entry.get("currency") or entry.get("asset_name") or ""),
                entry.get("collateral_index_price"),
            )
            if multiplier is None:
                continue
            try:
                derived_cash_collateral += float(entry.get("balance") or 0.0) * multiplier
            except (TypeError, ValueError):
                continue

        cash_collateral = _summary_amount(summary_dict, "cash_collateral")
        if cash_collateral is None:
            cash_collateral = derived_cash_collateral
        unrealized_pnl = _summary_amount(summary_dict, "unrealised_pnl", "unrealized_pnl") or 0.0
        margin = _summary_amount(summary_dict, "margin", "equity", "portfolio_value")
        if margin is None:
            margin = cash_collateral + unrealized_pnl

        balance = float(margin or cash_collateral or 0.0)
        total_value = float(
            _summary_amount(summary_dict, "portfolio_value", "margin", "equity")
            or balance
        )

        positions: list[PositionSnapshot] = []
        if isinstance(portfolio, list):
            for entry in portfolio:
                instrument_name = entry.get("instrument_name") or ""
                size = float(entry.get("position", entry.get("amount", 0.0)) or 0.0)
                if size == 0:
                    continue
                spec = parse_instrument_name(instrument_name)
                underlying = spec.underlying if spec else instrument_name.split("-", 1)[0]
                entry_price = float(entry.get("average_price", 0.0) or 0.0)
                mark_price = float(entry.get("mark_price", 0.0) or 0.0)

                # Thalex uses British spelling "unrealised_pnl" in some responses
                raw_pnl = (
                    entry.get("unrealised_pnl")
                    or entry.get("unrealized_pnl")
                    or entry.get("pnl")
                )
                if raw_pnl is not None:
                    pos_pnl = float(raw_pnl)
                elif entry_price and mark_price:
                    # Compute from mark vs entry
                    pos_pnl = (mark_price - entry_price) * size  # size is signed
                else:
                    pos_pnl = 0.0

                positions.append(
                    PositionSnapshot(
                        venue=self.venue,
                        asset=underlying,
                        instrument_name=instrument_name,
                        side="long" if size > 0 else "short",
                        size=abs(size),
                        entry_price=entry_price,
                        current_price=mark_price,
                        unrealized_pnl=pos_pnl,
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

    async def get_greeks(self, instrument_name: str) -> dict:
        """Return ``{delta, gamma, vega, theta, mark_iv}`` for an option instrument.

        The Thalex SDK does not document the ticker response shape, so this
        parser tries multiple plausible field paths and returns whatever it
        finds. Unrecognized payloads return an empty dict — callers must be
        ready to fall back rather than expecting an exception.

        Results are cached per-instrument for ``_GREEKS_TTL_SECONDS`` seconds
        so the event-driven hedger can call this in a tight loop without
        spamming the WebSocket.
        """
        cached = self._greeks_cache.get(instrument_name)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < _GREEKS_TTL_SECONDS:
            return cached[1]

        result = await self._request(self._client.ticker, instrument_name=instrument_name)
        parsed = _extract_greeks(result)
        self._greeks_cache[instrument_name] = (now, parsed)
        return parsed

    def cache_greeks_snapshot(self, instrument_name: str, payload: Any) -> dict:
        """Seed the short-lived greeks cache from a subscription payload."""
        parsed = _extract_greeks(payload)
        if parsed:
            self._greeks_cache[instrument_name] = (time.monotonic(), parsed)
        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_order_result(
        self,
        asset: str,
        side: str,
        amount: float,
        raw: Any,
        *,
        submitted_price: Optional[float] = None,
    ) -> OrderResult:
        order_id = ""
        status = "ok"
        price = submitted_price
        error: Optional[str] = None
        if isinstance(raw, dict):
            order_id = str(raw.get("order_id") or raw.get("id") or "")
            raw_status = str(raw.get("status") or "ok").lower()
            # Normalize Thalex status strings to the ExchangeAdapter contract
            # ("ok | filled | resting | rejected | error"). Thalex uses "open"
            # for a live resting order and "partially_filled" for a partial
            # fill — the adapter contract has no equivalents, so we fold them
            # into the closest canonical values.
            _SUCCESS_RAW = {"open", "filled", "partially_filled", "ok", "resting"}
            _STATUS_MAP = {
                "open": "resting",
                "partially_filled": "filled",
                "filled": "filled",
                "resting": "resting",
                "ok": "ok",
                "rejected": "rejected",
                "cancelled": "rejected",
                "canceled": "rejected",
                "error": "error",
            }
            err_payload = raw.get("error") or raw.get("reason")
            if err_payload and raw_status not in _SUCCESS_RAW:
                error = str(err_payload)
                status = "rejected"
            elif raw_status in {"rejected", "cancelled", "canceled", "error"}:
                status = _STATUS_MAP.get(raw_status, "rejected")
                error = str(raw.get("reject_reason") or raw.get("error") or raw_status)
            else:
                status = _STATUS_MAP.get(raw_status, raw_status or "ok")
            price = _quote_field(
                raw,
                ("price",),
                ("average_price",),
                ("avg_price",),
                ("filled_price",),
                ("fill_price",),
                ("limit_price",),
            ) or price
        return OrderResult(
            venue=self.venue,
            order_id=order_id,
            asset=asset,
            side=side,
            amount=amount,
            status=status,
            instrument_name=asset,
            price=price,
            raw=raw if isinstance(raw, dict) else None,
            error=error,
        )

    async def margin_preflight(self, required_collateral_usd: float) -> tuple[bool, str]:
        """Cheap pre-trade margin check against Thalex account cash collateral.

        Fetches ``account_summary`` and compares available cash/margin against
        ``required_collateral_usd`` (the estimated premium to pay for a long
        or the expected margin requirement for a short, in USD). Returns
        ``(True, "")`` when the venue has enough collateral, or
        ``(False, reason)`` otherwise.

        Fails open on RPC errors — a flaky account_summary shouldn't block a
        trade; the order submit itself will still catch a true rejection.
        """
        if required_collateral_usd <= 0:
            return True, ""
        try:
            state = await self.get_user_state()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Thalex margin_preflight: get_user_state failed — allowing trade: %s", exc)
            return True, ""
        # Prefer the venue's free/available cash fields — ``balance`` on
        # AccountState is portfolio equity (cash + unrealised PnL) and can
        # overstate what's actually unencumbered when positions are eating
        # margin. Fall back through the common field names before settling
        # for ``balance`` as a last resort.
        summary: dict = {}
        raw = getattr(state, "raw", None)
        if isinstance(raw, dict):
            summary = raw.get("summary") or {}
        available: Optional[float] = None
        source = ""
        for key in ("free_collateral", "available_cash", "available_funds", "cash_collateral"):
            val = summary.get(key) if isinstance(summary, dict) else None
            if val is None:
                val = getattr(state, key, None)
            if val is None:
                continue
            try:
                available = float(val)
                source = key
                break
            except (TypeError, ValueError):
                continue
        if available is None:
            available = float(getattr(state, "balance", 0.0) or 0.0)
            source = "balance (fallback)"
        if available <= 0:
            return False, f"Thalex {source} is {available:.2f} — no collateral available"
        # 10% buffer for fees/slippage + variation margin moves during fill.
        buffered = required_collateral_usd * 1.10
        if available < buffered:
            return False, (
                f"Thalex {source} ${available:,.2f} below required ≈ ${buffered:,.2f} "
                f"(${required_collateral_usd:,.2f} +10% buffer)"
            )
        return True, ""
