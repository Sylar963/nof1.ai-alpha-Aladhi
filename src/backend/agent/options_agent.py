"""OptionsAgent — separate LLM pipeline focused on vol/options reasoning.

This is the brain that turns an :class:`OptionsContext` digest into one or
more parsed :class:`TradeDecision` objects. It is **completely independent**
from the perps :class:`TradingAgent` — different system prompt, different
JSON schema, different cadence.

Why a separate agent
--------------------
Vol/options reasoning is fundamentally different from spot/perp reasoning.
The perps agent uses locally-computed structure indicators (SMA99,
Keltner, AVWAP, opening range) and reasons about trend continuation.
The options agent reasons about IV regime, vol
surface skew, term structure, and the gamma/vega/theta of multi-leg
positions. Cramming both into one prompt makes both worse — separating
them gives each agent a focused mental model.

Decision schema
---------------
The agent expects the LLM to return ``{"reasoning": str, "trade_decisions":
[...]}`` where each entry conforms to the multi-leg options shape defined in
:mod:`src.backend.agent.decision_schema`. Defined-risk only — naked credit
puts/calls are intentionally absent from the strategy enum.

LLM client interface
--------------------
The agent doesn't import an LLM SDK directly. It depends on a small
``llm.chat_json(system_prompt, user_prompt, schema) -> dict`` interface so
the test suite can swap in a fake. The real client wrapper lives outside
this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.backend.agent.decision_schema import (
    DecisionParseError,
    TradeDecision,
    parse_decision,
)
from src.backend.options_intel.snapshot import OptionsContext


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — assembled from named sections
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are a hedge-fund-grade BTC OPTIONS trader running on the Thalex venue,
hedged on Hyperliquid perps. You reason about volatility surface, greeks,
expected moves, and term structure — NOT about RSI, MACD, or moving averages.

Your job is to read the compact OptionsContext digest below and emit
zero or more trade decisions in strict JSON form.

VENUE
- All options decisions go to Thalex (venue="thalex").
- Delta hedges fire automatically on Hyperliquid for the *_delta_hedged
  strategies — you do NOT need to size them yourself.

DEFINED-RISK STRATEGIES (the only allowed values for `strategy`)
- credit_put_spread        — sell premium below support, defined risk
- credit_call_spread       — sell premium above resistance, defined risk
- iron_condor              — sell both spreads simultaneously (vol crush)
- long_call_delta_hedged   — directional long, gamma-scalped on perps
- long_put_delta_hedged    — directional short, gamma-scalped on perps
- vol_arb                  — exploit Thalex IV mispricing vs Deribit

NAKED LEGS ARE FORBIDDEN. Every short premium position MUST be wrapped as a
vertical (credit_*_spread) or as both legs of an iron condor."""


_OUTPUT_CONTRACT = """\
OUTPUT CONTRACT
Return a strict JSON object with two keys, in order:
  - "reasoning": long-form analysis (string, be verbose).
  - "trade_decisions": array of decision objects.

Each decision object MUST contain:
  - asset, action (buy|sell|hold), rationale (string)
  - venue (must be "thalex")
  - strategy (one of the defined-risk values above)
  - underlying ("BTC")
  - tenor_days (int) OR per-leg tenor_days
  - For multi-leg strategies: legs[] with kind/side/contracts/target_strike

OPTIONAL METADATA (encouraged)
- entry_kind: outright | vertical | calendar | diagonal | iron_condor | vol_arb
- vol_view:   short_vol | long_vol | neutral

SIZING
- Always express size in `contracts` (float, BTC equivalent). Min 0.001,
  max 0.1 per trade.
- For multi-tenor gamma builds, set `target_gamma_btc` instead of `contracts`
  and the system will distribute across tenors automatically.

Do not emit Markdown. Do not include extra properties. If no trade is
warranted right now, return an empty trade_decisions array."""


_OPTIONS_SYSTEM_PROMPT = "\n\n".join([
    _PREAMBLE,
    _OUTPUT_CONTRACT,
])


class OptionsAgent:
    """LLM-driven options decision pipeline."""

    def __init__(self, llm) -> None:
        """``llm`` must expose
        ``async chat_json(system_prompt, user_prompt, schema) -> dict``."""
        self.llm = llm
        # Most recent raw LLM payload ({reasoning, trade_decisions}). The GUI
        # surfaces this so the operator can inspect the options-cycle reasoning
        # text alongside the perps cycle. Reset to an empty dict on failure.
        self.last_payload: dict = {}

    async def decide(self, context: OptionsContext) -> list[TradeDecision]:
        """Run one decision cycle against the supplied options context.

        Returns the list of *successfully parsed* TradeDecision objects.
        Malformed individual decisions are dropped with a logged warning;
        the agent never raises on bad LLM output.
        """
        user_prompt = context.to_json()
        schema = self._build_schema()

        try:
            payload = await self.llm.chat_json(
                system_prompt=_OPTIONS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=schema,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("OptionsAgent LLM call failed: %s", exc)
            self.last_payload = {}
            return []

        self.last_payload = payload if isinstance(payload, dict) else {}

        raw_decisions = []
        if isinstance(payload, dict):
            raw_decisions = payload.get("trade_decisions") or []
        if not isinstance(raw_decisions, list):
            logger.warning("OptionsAgent: trade_decisions is not a list, got %r", type(raw_decisions))
            return []

        parsed: list[TradeDecision] = []
        for raw in raw_decisions:
            try:
                parsed.append(parse_decision(raw))
            except DecisionParseError as exc:
                logger.warning("OptionsAgent: dropping malformed decision: %s — %s", exc, raw)
        return parsed

    @staticmethod
    def _build_schema() -> dict:
        """JSON schema the LLM client passes to the model for structured output."""
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "trade_decisions": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["reasoning", "trade_decisions"],
        }
