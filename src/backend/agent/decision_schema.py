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

# Defined-risk strategies only. Single-leg short premium plays (`credit_put`,
# `credit_call`) are intentionally absent — every short-vol position must be
# wrapped as a vertical spread or an iron condor for capped downside.
VALID_STRATEGIES = {
    "credit_put_spread",
    "credit_call_spread",
    "iron_condor",
    "long_call_delta_hedged",
    "long_put_delta_hedged",
    "vol_arb",
}
VALID_KINDS = {"call", "put"}
VALID_VOL_VIEWS = {"short_vol", "long_vol", "neutral"}
VALID_ENTRY_KINDS = {
    "outright",
    "vertical",
    "calendar",
    "diagonal",
    "iron_condor",
    "vol_arb",
}


class DecisionParseError(ValueError):
    """Raised when an LLM payload cannot be coerced into a TradeDecision."""


@dataclass
class OptionsLeg:
    """A single leg inside a multi-leg options strategy (e.g. iron condor).

    ``tenor_days`` is optional — when set, this leg overrides the
    decision-level tenor (used for calendar/diagonal spreads where each leg
    has a different expiry).
    """

    kind: str  # "call" | "put"
    side: str  # "buy" | "sell"
    contracts: float
    target_strike: Optional[float] = None
    target_delta: Optional[float] = None
    tenor_days: Optional[int] = None


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

    # PR B additions: hedge-fund-grade options metadata
    entry_kind: Optional[str] = None  # outright|vertical|calendar|diagonal|iron_condor|vol_arb
    vol_view: Optional[str] = None  # short_vol|long_vol|neutral
    target_gamma_btc: Optional[float] = None  # for multi-tenor auto-distribute sizing

    risk_flags: list[str] = field(default_factory=list)

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
                tenor_days=_optional_int(raw.get("tenor_days")),
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

    # Type-safe venue resolution — a non-string venue (e.g. int from a
    # malformed LLM response) would blow up on ``.lower()`` before this
    # check could report it as a validation error.
    raw_venue = payload.get("venue")
    if raw_venue is None:
        venue = "hyperliquid"
    elif not isinstance(raw_venue, str):
        raise DecisionParseError(
            f"venue must be a string, got {type(raw_venue).__name__}: {raw_venue!r}"
        )
    else:
        venue = raw_venue.lower()
    if venue not in VALID_VENUES:
        raise DecisionParseError(f"venue must be one of {VALID_VENUES}, got {venue!r}")

    strategy = payload.get("strategy")
    if strategy is not None and strategy not in VALID_STRATEGIES:
        raise DecisionParseError(
            f"strategy must be one of {VALID_STRATEGIES} or null, got {strategy!r}"
        )

    # Options strategies can only live on the Thalex venue — enforce the
    # strategy ↔ venue contract at parse time so a cross-contaminated perps
    # decision can't slip through routing. When strategy is set but venue is
    # missing, coerce to thalex (forgiving the common LLM omission). When
    # venue is explicitly hyperliquid, reject — that's a prompt bug we want
    # to surface, not paper over.
    if strategy is not None:
        if raw_venue is None:
            venue = "thalex"
        else:
            # ``raw_venue`` is already known to be a string here; the
            # isinstance check above rejects everything else.
            if raw_venue.lower() != "thalex":
                raise DecisionParseError(
                    f"options strategy {strategy!r} requires venue='thalex', got {raw_venue!r}"
                )

    underlying = payload.get("underlying")
    if strategy is not None and not underlying:
        raise DecisionParseError("strategy decisions require an 'underlying' field")

    kind = payload.get("kind")
    if kind is not None and kind not in VALID_KINDS:
        raise DecisionParseError(f"kind must be one of {VALID_KINDS} or null, got {kind!r}")

    vol_view = payload.get("vol_view")
    if vol_view is not None and vol_view not in VALID_VOL_VIEWS:
        raise DecisionParseError(
            f"vol_view must be one of {VALID_VOL_VIEWS} or null, got {vol_view!r}"
        )

    entry_kind = payload.get("entry_kind")
    if entry_kind is not None and entry_kind not in VALID_ENTRY_KINDS:
        raise DecisionParseError(
            f"entry_kind must be one of {VALID_ENTRY_KINDS} or null, got {entry_kind!r}"
        )

    raw_flags = payload.get("risk_flags")
    if raw_flags is None:
        risk_flags: list[str] = []
    elif isinstance(raw_flags, list):
        risk_flags = [str(f) for f in raw_flags if f]
    else:
        raise DecisionParseError(
            f"risk_flags must be a list of strings, got {type(raw_flags).__name__}"
        )

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
        entry_kind=entry_kind,
        vol_view=vol_view,
        target_gamma_btc=_optional_float(payload.get("target_gamma_btc")),
        risk_flags=risk_flags,
    )
