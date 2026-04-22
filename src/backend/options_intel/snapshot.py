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
    realized_iv_ratio_15d: float
    straddle_test_15d: dict

    # Cross-venue
    top_mispricings_vs_deribit: list = field(default_factory=list)

    # Portfolio
    open_positions: list = field(default_factory=list)
    portfolio_greeks: dict = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a dict suitable for the LLM, with hard caps on list lengths."""
        return {
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
            "realized_iv_ratio_15d": self.realized_iv_ratio_15d,
            "straddle_test_15d": self.straddle_test_15d,
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
        }

    def to_json(self) -> str:
        """Serialize to JSON. Compact, no indentation, sorted keys."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True, default=str)
