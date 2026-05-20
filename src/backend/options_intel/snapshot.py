"""OptionsContext: the compact digest the LLM consumes.

Hard rule: this dataclass NEVER carries a raw chain. The serializer caps the
mispricings list to a sane top-N, drops verbose fields, and stays under a
strict byte budget so the prompt token cost is flat regardless of how many
instruments Thalex has listed.

The LLM's job is to reason about regime + greeks + mispricings, not to scan
500 instruments — that's what the pre-LLM intel layer is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# Hard byte budget for the serialized snapshot. Roughly aligned with a
# ~500-token prompt fragment so we leave plenty of room for the system
# prompt + reasoning around it.
BYTE_BUDGET: int = 4096

# Cap on how many entries the mispricings list can carry into the LLM.
_MAX_MISPRICINGS: int = 10


@dataclass
class OptionsContext:
    """Compact options-intel digest passed to the LLM each decision cycle."""

    timestamp_utc: str
    spot: float
    spot_24h_change_pct: float
    opening_range: dict
    keltner: dict

    # Vol surface highlights
    atm_iv_by_tenor: dict
    skew_25d_by_tenor: dict
    term_structure_slope: float
    expected_move_pct_by_tenor: dict

    # Regime
    vol_regime: str
    vol_regime_confidence: str
    realized_iv_ratio_30d: float
    straddle_test_30d: dict

    # Cross-venue
    top_mispricings_vs_deribit: list = field(default_factory=list)

    # Portfolio
    open_positions: list = field(default_factory=list)
    portfolio_greeks: dict = field(default_factory=dict)
    structures: list = field(default_factory=list)
    structure_views: list = field(default_factory=list)
    triggered_by_events: list = field(default_factory=list)

    # Risk
    capital_available: float = 0.0
    max_contracts_per_trade: float = 0.1
    max_open_positions: int = 3
    open_position_count: int = 0

    # Hedge budget — the options agent's biggest silent failure mode is
    # proposing a delta-hedged strategy when the Hyperliquid sidecar has no
    # free margin for the perp leg. We surface both the raw HL collateral
    # and a pre-computed notional cap so the LLM has no excuse to ignore it.
    hyperliquid_free_margin: float = 0.0
    hyperliquid_max_leverage: int = 1
    max_hedge_notional: float = 0.0

    # Recent activity
    recent_options_trades: list = field(default_factory=list)
    # Skip-feedback loop — entries like "proposal_skipped_insufficient_*"
    # written by bot_engine on the previous cycle(s). Having this in context
    # is what lets the LLM self-correct instead of re-proposing the same
    # unaffordable trade every 3 hours.
    recent_options_skips: list = field(default_factory=list)

    # Explicit coverage map for vol/greek data. Lets the LLM tell "the tenor
    # isn't on the chain" apart from "I forgot to look" — both otherwise
    # manifest as a missing dict key. surface_stale=True is the only
    # authoritative "vol data unavailable" signal.
    vol_data_coverage: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a dict suitable for the LLM, with hard caps on list lengths."""
        payload = {
            "timestamp_utc": self.timestamp_utc,
            "spot": self.spot,
            "spot_24h_change_pct": self.spot_24h_change_pct,
            "opening_range": self.opening_range,
            "keltner": self.keltner,
            "atm_iv_by_tenor": self.atm_iv_by_tenor,
            "skew_25d_by_tenor": self.skew_25d_by_tenor,
            "term_structure_slope": self.term_structure_slope,
            "expected_move_pct_by_tenor": self.expected_move_pct_by_tenor,
            "vol_regime": self.vol_regime,
            "vol_regime_confidence": self.vol_regime_confidence,
            "realized_iv_ratio_30d": self.realized_iv_ratio_30d,
            "straddle_test_30d": self.straddle_test_30d,
            "top_mispricings_vs_deribit": self.top_mispricings_vs_deribit[:_MAX_MISPRICINGS],
            "open_positions": self.open_positions[:self.max_open_positions],
            "portfolio_greeks": self.portfolio_greeks,
            "capital_available": self.capital_available,
            "max_contracts_per_trade": self.max_contracts_per_trade,
            "max_open_positions": self.max_open_positions,
            "open_position_count": self.open_position_count,
            "hyperliquid_free_margin": self.hyperliquid_free_margin,
            "hyperliquid_max_leverage": self.hyperliquid_max_leverage,
            "max_hedge_notional": self.max_hedge_notional,
            "recent_options_trades": self.recent_options_trades[:5],
            "recent_options_skips": self.recent_options_skips[:5],
            "vol_data_coverage": self.vol_data_coverage,
        }
        from src.backend.config_loader import CONFIG
        if CONFIG.get("options_structure_prompt"):
            if self.structure_views:
                payload["structures"] = [v.to_dict() for v in self.structure_views]
            if self.triggered_by_events:
                payload["triggered_by_events"] = [e.to_dict() for e in self.triggered_by_events]
        return payload

    def to_json(self) -> str:
        """Serialize to JSON. Compact, no indentation, sorted keys."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True, default=str)


@dataclass(frozen=True)
class StructureView:
    structure_id: str
    kind: str
    underlying: str
    tenor_days: int
    days_open: int
    legs: tuple[dict, ...]
    net_premium: float
    is_credit: bool
    max_loss: Optional[float]
    max_profit: Optional[float]
    breakevens: tuple[float, ...]
    short_leg_delta: Optional[float]
    breach_state: str
    pnl_abs: float
    pnl_pct: float
    aggregate_greeks: dict

    @classmethod
    def from_classifier_dict(
        cls,
        structure: dict,
        open_positions: list,
        days_open: int,
    ) -> "StructureView":
        positions_by_name = {pos["instrument_name"]: pos for pos in open_positions or []}
        expanded_legs: list = []
        for instrument_name in structure.get("legs", []) or []:
            pos = positions_by_name.get(instrument_name)
            if pos is None:
                expanded_legs.append({"instrument_name": instrument_name})
                continue
            delta_value = pos.get("delta")
            abs_delta = abs(float(delta_value)) if delta_value is not None else None
            expanded_legs.append({
                "instrument_name": instrument_name,
                "kind": pos.get("kind"),
                "side": pos.get("side"),
                "strike": float(pos["strike"]) if pos.get("strike") is not None else None,
                "contracts": float(pos["size"]) if pos.get("size") is not None else None,
                "abs_delta": abs_delta,
            })

        tenor_days_min = int(structure.get("tenor_days_min", 0) or 0)
        tenor_days_max = int(structure.get("tenor_days_max", tenor_days_min) or tenor_days_min)
        if tenor_days_min and tenor_days_max:
            tenor_days = min(tenor_days_min, tenor_days_max)
        else:
            tenor_days = tenor_days_min or tenor_days_max

        max_loss = structure.get("max_loss")
        max_profit = structure.get("max_profit")
        short_leg_delta = structure.get("short_leg_delta")

        return cls(
            structure_id=str(structure["structure_id"]),
            kind=str(structure["kind"]),
            underlying=str(structure.get("underlying", "")),
            tenor_days=int(tenor_days or 0),
            days_open=int(days_open),
            legs=tuple(expanded_legs),
            net_premium=float(structure.get("net_premium", 0.0) or 0.0),
            is_credit=bool(structure.get("is_credit", False)),
            max_loss=float(max_loss) if max_loss is not None else None,
            max_profit=float(max_profit) if max_profit is not None else None,
            breakevens=tuple(float(b) for b in (structure.get("breakevens") or [])),
            short_leg_delta=float(short_leg_delta) if short_leg_delta is not None else None,
            breach_state=str(structure.get("breach_state", "nominal")),
            pnl_abs=float(structure.get("pnl_abs", 0.0) or 0.0),
            pnl_pct=float(structure.get("pnl_pct", 0.0) or 0.0),
            aggregate_greeks=dict(structure.get("aggregate_greeks") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "structure_id": self.structure_id,
            "kind": self.kind,
            "underlying": self.underlying,
            "tenor_days": self.tenor_days,
            "days_open": self.days_open,
            "legs": [dict(leg) for leg in self.legs],
            "net_premium": self.net_premium,
            "is_credit": self.is_credit,
            "max_loss": self.max_loss,
            "max_profit": self.max_profit,
            "breakevens": list(self.breakevens),
            "short_leg_delta": self.short_leg_delta,
            "breach_state": self.breach_state,
            "pnl_abs": self.pnl_abs,
            "pnl_pct": self.pnl_pct,
            "aggregate_greeks": dict(self.aggregate_greeks),
        }


@dataclass(frozen=True)
class EventSummary:
    type: str
    fired_at: str
    description: str
    structure_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "fired_at": self.fired_at,
            "description": self.description,
            "structure_id": self.structure_id,
        }
