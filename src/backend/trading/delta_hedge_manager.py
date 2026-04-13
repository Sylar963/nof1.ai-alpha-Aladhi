"""Portfolio-driven delta hedge manager for Thalex options.

The original implementation hedged one option instrument at a time based on the
decision that opened it. This version treats the live Thalex portfolio as the
source of truth:

- reconcile open option positions from ``thalex.get_user_state()``
- subscribe/unsubscribe ticker channels to match the live portfolio
- compute net option delta per underlying across the WHOLE book
- compare that against the current Hyperliquid perp delta
- rebalance only the residual drift above the configured threshold

This makes spreads, multi-tenor books, restarts, and manual position changes
behave correctly because the hedge target comes from venue state rather than
local bookkeeping.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Optional

from src.backend.trading.exchange_adapter import ExchangeAdapter, OrderResult
from src.backend.trading.options import parse_instrument_name
from src.backend.trading.options_strategies import DeltaHedger


logger = logging.getLogger(__name__)


@dataclass
class _OptionPosition:
    instrument_name: str
    underlying: str
    side: str
    size: float
    delta_hint: Optional[float] = None


class DeltaHedgeManager:
    """Keeps the whole options book delta-neutral via Hyperliquid perps."""

    def __init__(
        self,
        thalex: ExchangeAdapter,
        hyperliquid: ExchangeAdapter,
        hedger: Optional[DeltaHedger] = None,
        enabled: bool = True,
    ) -> None:
        self.thalex = thalex
        self.hyperliquid = hyperliquid
        self.hedger = hedger or DeltaHedger()
        self._enabled = enabled
        self._subscribed_instruments: dict[str, str] = {}
        self._reconcile_lock = asyncio.Lock()
        self._underlying_locks: dict[str, asyncio.Lock] = {}
        self.degraded_underlyings: dict[str, str] = {}
        self._metrics_by_underlying: dict[str, dict[str, Any]] = {}
        self._last_known_positions: dict[str, list[_OptionPosition]] = {}
        self._last_state_error: Optional[str] = None
        self._unknown_state_logged_for: set[str] = set()

    def is_enabled(self) -> bool:
        return self._enabled

    async def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._enabled == enabled:
            return

        self._enabled = enabled
        if not enabled:
            for instrument_name in list(self._subscribed_instruments):
                await self._drop_subscription(instrument_name)
            self.degraded_underlyings.clear()
            self._metrics_by_underlying.clear()
            self._last_known_positions.clear()
            self._last_state_error = None
            self._unknown_state_logged_for.clear()
            logger.info("DeltaHedgeManager disabled")
            return

        logger.info("DeltaHedgeManager enabled")

    async def add_position(
        self,
        instrument_name: str,
        contracts: float,
        kind: str,
        underlying: str,
    ) -> None:
        """Backwards-compatible helper used by older tests/callers.

        The live reconciler no longer depends on this registration for hedge
        math, but keeping the method lets older code request an immediate
        subscription while the live portfolio catches up.
        """
        await self._ensure_subscription(instrument_name, underlying)
        logger.info(
            "DeltaHedgeManager primed subscription for %s (%s %.4f contracts)",
            instrument_name,
            kind,
            contracts,
        )

    async def remove_position(self, instrument_name: str) -> None:
        """Unsubscribe and forget a single instrument."""
        await self._drop_subscription(instrument_name)

    async def close(self) -> None:
        """Unsubscribe every tracked instrument."""
        for instrument_name in list(self._subscribed_instruments):
            await self._drop_subscription(instrument_name)
        self.degraded_underlyings.clear()
        self._metrics_by_underlying.clear()
        self._last_known_positions.clear()

    def _cache_positions(
        self,
        positions: list[_OptionPosition],
        underlying: Optional[str] = None,
    ) -> None:
        if underlying is None:
            grouped: dict[str, list[_OptionPosition]] = {}
            for position in positions:
                grouped.setdefault(position.underlying, []).append(position)
            self._last_known_positions = grouped
            return

        key = underlying.upper()
        self._last_known_positions[key] = [
            position for position in positions if position.underlying == key
        ]

    def _cached_positions(self, underlying: Optional[str] = None) -> list[_OptionPosition]:
        if underlying is None:
            cached: list[_OptionPosition] = []
            for bucket in self._last_known_positions.values():
                cached.extend(bucket)
            return cached
        return list(self._last_known_positions.get(underlying.upper(), []))

    def get_status_snapshot(self) -> dict[str, Any]:
        """Return a UI-friendly snapshot of hedge health and metrics."""
        if not self._enabled:
            return {
                "health": "disabled",
                "enabled": False,
                "degraded_underlyings": {},
                "tracked_underlyings": 0,
                "active_underlyings": 0,
                "metrics": [],
                "state_error": None,
                "last_update": datetime.now(UTC).isoformat(),
            }

        metrics = [
            self._metrics_by_underlying[key]
            for key in sorted(self._metrics_by_underlying)
        ]
        degraded = [m for m in metrics if m.get("degraded")]
        active = [m for m in metrics if (m.get("open_option_positions") or 0) > 0]
        if self._last_state_error:
            health = "unavailable"
        elif degraded:
            health = "degraded"
        elif active:
            health = "healthy"
        else:
            health = "idle"
        return {
            "health": health,
            "enabled": True,
            "degraded_underlyings": dict(self.degraded_underlyings),
            "tracked_underlyings": len(metrics),
            "active_underlyings": len(active),
            "metrics": metrics,
            "state_error": self._last_state_error,
            "last_update": datetime.now(UTC).isoformat(),
        }

    async def reconcile(self, underlying: Optional[str] = None) -> list[OrderResult]:
        """Rebuild subscriptions and hedge target from live Thalex positions."""
        if not self._enabled:
            return []

        async with self._reconcile_lock:
            tracked_underlyings_before = {
                name_underlying
                for _, name_underlying in self._subscribed_instruments.items()
                if underlying is None or name_underlying == underlying
            }
            positions = await self._load_option_positions(underlying=underlying)
            live_state_known = positions is not None
            if live_state_known:
                self._cache_positions(positions, underlying=underlying)
                await self._sync_subscriptions(positions, underlying=underlying)
            else:
                positions = self._cached_positions(underlying=underlying)
                if not positions:
                    scope = underlying.upper() if underlying is not None else "all underlyings"
                    if scope not in self._unknown_state_logged_for:
                        logger.warning(
                            "delta hedge reconcile: skipping %s because Thalex position state is unknown",
                            scope,
                        )
                        self._unknown_state_logged_for.add(scope)
                    return []

            tracked_underlyings_after = {
                name_underlying
                for _, name_underlying in self._subscribed_instruments.items()
                if underlying is None or name_underlying == underlying
            }
            target_underlyings = {pos.underlying for pos in positions}
            if live_state_known:
                target_underlyings |= tracked_underlyings_before | tracked_underlyings_after
                if underlying is not None:
                    target_underlyings.add(underlying)

            orders: list[OrderResult] = []
            for underlying_name in sorted(target_underlyings):
                bucket = [pos for pos in positions if pos.underlying == underlying_name]
                orders.extend(await self._rebalance_underlying(underlying_name, bucket))
            return orders

    async def _load_option_positions(
        self,
        underlying: Optional[str] = None,
    ) -> Optional[list[_OptionPosition]]:
        try:
            state = await self.thalex.get_user_state()
        except Exception as exc:  # pylint: disable=broad-except
            message = str(exc)
            if message != self._last_state_error:
                logger.warning("delta hedge reconcile: thalex get_user_state failed: %s", exc)
                self._last_state_error = message
            return None

        self._last_state_error = None
        self._unknown_state_logged_for.clear()

        raw_positions = []
        if isinstance(state, dict):
            raw_positions = state.get("positions") or []
        else:
            raw_positions = getattr(state, "positions", []) or []

        parsed_positions: list[_OptionPosition] = []
        for raw in raw_positions:
            instrument_name = _field(raw, "instrument_name") or _field(raw, "asset") or ""
            spec = parse_instrument_name(instrument_name)
            if spec is None:
                continue

            underlying_name = (_field(raw, "asset") or spec.underlying or "").upper()
            if underlying is not None and underlying_name != underlying.upper():
                continue

            try:
                size = abs(float(_field(raw, "size") or 0.0))
            except (TypeError, ValueError):
                continue
            if size <= 0:
                continue

            side = str(_field(raw, "side") or "long").lower()
            delta_hint = _coerce_float(_field(raw, "delta"))
            parsed_positions.append(
                _OptionPosition(
                    instrument_name=instrument_name,
                    underlying=underlying_name,
                    side=side,
                    size=size,
                    delta_hint=delta_hint,
                )
            )

        return parsed_positions

    async def _sync_subscriptions(
        self,
        positions: list[_OptionPosition],
        underlying: Optional[str] = None,
    ) -> None:
        live_instruments = {
            pos.instrument_name: pos.underlying
            for pos in positions
        }
        current_instruments = {
            instrument_name: name_underlying
            for instrument_name, name_underlying in self._subscribed_instruments.items()
            if underlying is None or name_underlying == underlying.upper()
        }

        for instrument_name, name_underlying in live_instruments.items():
            if instrument_name not in current_instruments:
                await self._ensure_subscription(instrument_name, name_underlying)

        for instrument_name in set(current_instruments) - set(live_instruments):
            await self._drop_subscription(instrument_name)

    async def _ensure_subscription(self, instrument_name: str, underlying: str) -> None:
        if instrument_name in self._subscribed_instruments:
            return

        async def _callback(payload):
            await self._on_ticker(instrument_name, payload)

        if hasattr(self.thalex, "subscribe_ticker"):
            await self.thalex.subscribe_ticker(instrument_name, _callback)
        self._subscribed_instruments[instrument_name] = underlying.upper()
        logger.info("DeltaHedgeManager subscribed %s", instrument_name)

    async def _drop_subscription(self, instrument_name: str) -> None:
        if instrument_name not in self._subscribed_instruments:
            return
        self._subscribed_instruments.pop(instrument_name, None)
        if hasattr(self.thalex, "unsubscribe_ticker"):
            await self.thalex.unsubscribe_ticker(instrument_name)
        logger.info("DeltaHedgeManager unsubscribed %s", instrument_name)

    async def _on_ticker(self, instrument_name: str, payload) -> None:
        if not self._enabled:
            return

        if hasattr(self.thalex, "cache_greeks_snapshot"):
            try:
                self.thalex.cache_greeks_snapshot(instrument_name, payload)
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("cache_greeks_snapshot failed for %s: %s", instrument_name, exc)

        underlying = self._subscribed_instruments.get(instrument_name)
        if underlying is None:
            spec = parse_instrument_name(instrument_name)
            underlying = spec.underlying if spec else None
        if underlying is None:
            return
        await self.reconcile(underlying=underlying)

    async def _rebalance_underlying(
        self,
        underlying: str,
        positions: list[_OptionPosition],
    ) -> list[OrderResult]:
        lock = self._underlying_locks.setdefault(underlying, asyncio.Lock())
        async with lock:
            net_option_delta = 0.0
            missing_greeks: list[str] = []
            subscribed_instruments = sorted(
                instrument_name
                for instrument_name, instrument_underlying in self._subscribed_instruments.items()
                if instrument_underlying == underlying
            )

            for position in positions:
                delta = await self._resolve_position_delta(position)
                if delta is None:
                    missing_greeks.append(position.instrument_name)
                    continue
                signed_size = position.size if position.side == "long" else -position.size
                net_option_delta += signed_size * delta

            if missing_greeks:
                message = (
                    "missing live delta for " + ", ".join(sorted(missing_greeks))
                )
                previous = self.degraded_underlyings.get(underlying)
                self.degraded_underlyings[underlying] = message
                if previous != message:
                    logger.error(
                        "delta hedge degraded for %s: %s — skipping rebalance until greeks recover",
                        underlying,
                        message,
                    )
                current_perp_delta = await self._current_perp_delta(underlying)
                self._metrics_by_underlying[underlying] = {
                    "underlying": underlying,
                    "status": "degraded",
                    "degraded": True,
                    "degraded_reason": message,
                    "open_option_positions": len(positions),
                    "subscribed_instruments": subscribed_instruments,
                    "net_option_delta": round(net_option_delta, 6),
                    "target_perp_delta": None,
                    "current_perp_delta": round(current_perp_delta, 6),
                    "residual_delta": None,
                    "drift_abs": None,
                    "threshold": self.hedger.threshold,
                    "last_rebalance_side": None,
                    "last_rebalance_size": 0.0,
                    "last_rebalance_at": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                return []

            self.degraded_underlyings.pop(underlying, None)

            target_perp_delta = -net_option_delta
            current_perp_delta = await self._current_perp_delta(underlying)
            residual_before = target_perp_delta - current_perp_delta
            action = self.hedger.compute_rebalance(
                target_delta=target_perp_delta,
                current_perp_delta=current_perp_delta,
            )
            if action.side == "noop" or action.contracts_to_trade <= 0:
                self._metrics_by_underlying[underlying] = {
                    "underlying": underlying,
                    "status": "neutral" if positions else "flat",
                    "degraded": False,
                    "degraded_reason": None,
                    "open_option_positions": len(positions),
                    "subscribed_instruments": subscribed_instruments,
                    "net_option_delta": round(net_option_delta, 6),
                    "target_perp_delta": round(target_perp_delta, 6),
                    "current_perp_delta": round(current_perp_delta, 6),
                    "residual_delta": round(residual_before, 6),
                    "drift_abs": round(abs(residual_before), 6),
                    "threshold": self.hedger.threshold,
                    "last_rebalance_side": None,
                    "last_rebalance_size": 0.0,
                    "last_rebalance_at": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                logger.debug(
                    "delta hedge noop for %s (target=%.4f current=%.4f)",
                    underlying,
                    target_perp_delta,
                    current_perp_delta,
                )
                return []

            try:
                if action.side == "buy":
                    order = await self.hyperliquid.place_buy_order(underlying, action.contracts_to_trade)
                else:
                    order = await self.hyperliquid.place_sell_order(underlying, action.contracts_to_trade)

                updated_perp_delta = await self._current_perp_delta(underlying)
                residual = target_perp_delta - updated_perp_delta
                self._metrics_by_underlying[underlying] = {
                    "underlying": underlying,
                    "status": "rebalanced",
                    "degraded": False,
                    "degraded_reason": None,
                    "open_option_positions": len(positions),
                    "subscribed_instruments": subscribed_instruments,
                    "net_option_delta": round(net_option_delta, 6),
                    "target_perp_delta": round(target_perp_delta, 6),
                    "current_perp_delta": round(updated_perp_delta, 6),
                    "residual_delta": round(residual, 6),
                    "drift_abs": round(abs(residual), 6),
                    "threshold": self.hedger.threshold,
                    "last_rebalance_side": action.side,
                    "last_rebalance_size": round(action.contracts_to_trade, 6),
                    "last_rebalance_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                logger.info(
                    "delta rebalance %s %.4f %s (target=%.4f before=%.4f after=%.4f residual=%.4f)",
                    action.side,
                    action.contracts_to_trade,
                    underlying,
                    target_perp_delta,
                    current_perp_delta,
                    updated_perp_delta,
                    residual,
                )
                return [order]
            except Exception as exc:  # pylint: disable=broad-except
                self._metrics_by_underlying[underlying] = {
                    "underlying": underlying,
                    "status": "error",
                    "degraded": True,
                    "degraded_reason": str(exc),
                    "open_option_positions": len(positions),
                    "subscribed_instruments": subscribed_instruments,
                    "net_option_delta": round(net_option_delta, 6),
                    "target_perp_delta": round(target_perp_delta, 6),
                    "current_perp_delta": round(current_perp_delta, 6),
                    "residual_delta": round(residual_before, 6),
                    "drift_abs": round(abs(residual_before), 6),
                    "threshold": self.hedger.threshold,
                    "last_rebalance_side": action.side,
                    "last_rebalance_size": round(action.contracts_to_trade, 6),
                    "last_rebalance_at": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                logger.error("delta rebalance failed for %s: %s", underlying, exc)
                return []

    async def _resolve_position_delta(self, position: _OptionPosition) -> Optional[float]:
        if position.delta_hint is not None:
            return position.delta_hint

        if hasattr(self.thalex, "get_greeks"):
            try:
                greeks = await self.thalex.get_greeks(position.instrument_name)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("get_greeks failed for %s: %s", position.instrument_name, exc)
                return None
            if isinstance(greeks, dict) and greeks.get("delta") is not None:
                return _coerce_float(greeks.get("delta"))
        return None

    async def _current_perp_delta(self, underlying: str) -> float:
        try:
            state = await self.hyperliquid.get_user_state()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("get_user_state failed: %s", exc)
            return 0.0

        positions = []
        if isinstance(state, dict):
            positions = state.get("positions") or []
        else:
            positions = getattr(state, "positions", []) or []

        for pos in positions:
            asset = _field(pos, "asset")
            if asset != underlying:
                continue
            size = _coerce_float(_field(pos, "size")) or 0.0
            side = str(_field(pos, "side") or "long").lower()
            magnitude = abs(size)
            return magnitude if side == "long" else -magnitude
        return 0.0


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
