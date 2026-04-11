"""Multi-venue decision schema (Option B intent style).

This is the typed contract between the LLM output and the trading layer. It
intentionally accepts both:

- The legacy Hyperliquid spot/perp shape: ``{asset, action, allocation_usd,
  tp_price, sl_price, exit_plan, rationale}``.
- The new Thalex options shape that adds ``venue``, ``strategy``,
  ``underlying``, ``kind``, ``tenor_days``, ``target_strike``,
  ``target_delta``, ``contracts``, and ``legs``.

The Thalex adapter consumes :class:`OptionIntent` produced by
:meth:`TradeDecision.to_option_intent` and resolves it against
``public/instruments`` at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.backend.trading.options import OptionIntent


VALID_VENUES = {"hyperliquid", "thalex"}
VALID_ACTIONS = {"buy", "sell", "hold"}
VALID_STRATEGIES = {
    "credit_put",
    "credit_spread",
    "long_call_delta_hedged",
    "long_put_delta_hedged",
}
VALID_KINDS = {"call", "put"}


class DecisionParseError(ValueError):
    """Raised when an LLM payload cannot be coerced into a TradeDecision."""


@dataclass
class OptionsLeg:
    """A single leg inside a multi-leg options strategy (e.g. credit spread)."""

    kind: str  # "call" | "put"
    side: str  # "buy" | "sell"
    contracts: float
    target_strike: Optional[float] = None
    target_delta: Optional[float] = None


@dataclass
class TradeDecision:
    """Normalized trade decision routed by the bot engine to a venue adapter."""

    asset: str
    action: str
    rationale: str
    venue: str = "hyperliquid"

    # Legacy Hyperliquid spot/perp fields
    allocation_usd: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    exit_plan: Optional[str] = None

    # New Thalex options fields
    strategy: Optional[str] = None
    underlying: Optional[str] = None
    kind: Optional[str] = None
    tenor_days: Optional[int] = None
    target_strike: Optional[float] = None
    target_delta: Optional[float] = None
    contracts: Optional[float] = None
    legs: list[OptionsLeg] = field(default_factory=list)

    def to_option_intent(self) -> Optional[OptionIntent]:
        """Build an OptionIntent for single-leg options decisions, else None."""
        if self.venue != "thalex" or self.strategy is None:
            return None
        if self.legs:
            return None  # multi-leg strategies are resolved leg-by-leg by the strategy layer
        if not self.kind or not self.tenor_days:
            return None
        return OptionIntent(
            underlying=self.underlying or self.asset,
            kind=self.kind,
            tenor_days=self.tenor_days,
            target_strike=self.target_strike,
            target_delta=self.target_delta,
        )


def _coerce_legs(raw_legs) -> list[OptionsLeg]:
    if not raw_legs:
        return []
    if not isinstance(raw_legs, list):
        raise DecisionParseError("legs must be a list of objects")
    out: list[OptionsLeg] = []
    for raw in raw_legs:
        if not isinstance(raw, dict):
            raise DecisionParseError("each leg must be an object")
        kind = raw.get("kind")
        side = raw.get("side")
        if kind not in VALID_KINDS:
            raise DecisionParseError(f"leg kind must be one of {VALID_KINDS}")
        if side not in {"buy", "sell"}:
            raise DecisionParseError("leg side must be 'buy' or 'sell'")
        contracts = float(raw.get("contracts", 0.0) or 0.0)
        out.append(
            OptionsLeg(
                kind=kind,
                side=side,
                contracts=contracts,
                target_strike=_optional_float(raw.get("target_strike")),
                target_delta=_optional_float(raw.get("target_delta")),
            )
        )
    return out


def _optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionParseError(f"expected number, got {value!r}") from exc


def _optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DecisionParseError(f"expected integer, got {value!r}") from exc


def parse_decision(payload: dict) -> TradeDecision:
    """Parse a single LLM trade-decision payload into a typed TradeDecision."""
    if not isinstance(payload, dict):
        raise DecisionParseError(f"decision must be an object, got {type(payload).__name__}")

    asset = payload.get("asset")
    if not isinstance(asset, str) or not asset:
        raise DecisionParseError("decision must include a non-empty 'asset'")

    action = payload.get("action")
    if action not in VALID_ACTIONS:
        raise DecisionParseError(f"action must be one of {VALID_ACTIONS}, got {action!r}")

    venue = (payload.get("venue") or "hyperliquid").lower()
    if venue not in VALID_VENUES:
        raise DecisionParseError(f"venue must be one of {VALID_VENUES}, got {venue!r}")

    strategy = payload.get("strategy")
    if strategy is not None and strategy not in VALID_STRATEGIES:
        raise DecisionParseError(
            f"strategy must be one of {VALID_STRATEGIES} or null, got {strategy!r}"
        )

    underlying = payload.get("underlying")
    if strategy is not None and not underlying:
        raise DecisionParseError("strategy decisions require an 'underlying' field")

    kind = payload.get("kind")
    if kind is not None and kind not in VALID_KINDS:
        raise DecisionParseError(f"kind must be one of {VALID_KINDS} or null, got {kind!r}")

    return TradeDecision(
        asset=asset,
        action=action,
        rationale=str(payload.get("rationale") or ""),
        venue=venue,
        allocation_usd=_optional_float(payload.get("allocation_usd")),
        tp_price=_optional_float(payload.get("tp_price")),
        sl_price=_optional_float(payload.get("sl_price")),
        exit_plan=payload.get("exit_plan"),
        strategy=strategy,
        underlying=underlying,
        kind=kind,
        tenor_days=_optional_int(payload.get("tenor_days")),
        target_strike=_optional_float(payload.get("target_strike")),
        target_delta=_optional_float(payload.get("target_delta")),
        contracts=_optional_float(payload.get("contracts")),
        legs=_coerce_legs(payload.get("legs")),
    )
