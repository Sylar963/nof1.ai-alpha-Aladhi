"""Options intelligence package: vol surface, pricing, regime, sizing.

The brain of the options trading agent. Pure-logic + read-only IO modules
that build a compact :class:`OptionsContext` digest the LLM consumes. No
order execution, no bot wiring — that lives in PR B.
"""
