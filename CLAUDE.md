# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nof1.ai Alpha Arena — an autonomous AI-powered trading bot that trades cryptocurrency on Hyperliquid (perpetual futures) and Thalex (BTC options) using LLM-driven decision-making via OpenRouter. Desktop GUI built with NiceGUI.

## Commands

```bash
# Run the app (NiceGUI desktop; browser mode on WSL)
python main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_decision_schema.py

# Run a single test by name
pytest tests/test_decision_schema.py::test_parse_valid_decision -v

# Run tests matching a keyword
pytest -k "options" -v
```

There is no linter, formatter, or type-checker configured. No CI/CD pipeline exists.

## Repo-Specific LLM Reference

- For dependency version bounds and pytest fixture/testing patterns, read `.agents/skills/nof1-stack-and-testing/SKILL.md` before editing stack-sensitive code or tests.
- Treat `requirements.txt` as the compatibility source of truth.

## Architecture

```text
main.py                          # Entry point — NiceGUI desktop app (browser mode on WSL)
src/
├── backend/
│   ├── bot_engine.py            # TradingBotEngine — core event loop
│   │                            #   5-min perps cycle (Hyperliquid)
│   │                            #   3-hour options cycle (Thalex, optional)
│   ├── config_loader.py         # CONFIG dict from .env via python-dotenv
│   ├── llm_client.py            # OpenRouter HTTP client
│   ├── agent/
│   │   ├── decision_maker.py    # TradingAgent — LLM prompt → TradeDecision
│   │   ├── decision_schema.py   # TradeDecision dataclass + parse_decision() validator
│   │   ├── options_agent.py     # Options-specific LLM logic
│   │   └── options_llm_lifecycle.py
│   ├── models/
│   │   └── trade_proposal.py    # In-memory TradeProposal dataclass (UUID ids, state-machine methods)
│   ├── trading/
│   │   ├── exchange_adapter.py  # ExchangeAdapter base class (OrderResult, PositionSnapshot)
│   │   ├── exchange_factory.py  # Venue routing
│   │   ├── hyperliquid_api.py   # HyperliquidAPI — perps, wallet auth, retry w/ backoff
│   │   ├── thalex_api.py        # ThalexAPI — options, JWT/RS512 auth, WebSocket
│   │   ├── options_strategies.py# Strategy executors (spreads, delta-hedged)
│   │   ├── options_scheduler.py # Two-cadence scheduler
│   │   ├── options.py           # Options intent dataclasses
│   │   └── delta_hedge_manager.py
│   ├── utils/
│   │   ├── formatting.py        # Numeric formatting helpers
│   │   └── prompt_utils.py      # JSON serialization, safe_float, round helpers for LLM prompts
│   ├── indicators/
│   │   ├── indicator_engine.py  # Local EMA/ATR/Keltner computation (fallback from TAAPI)
│   │   ├── taapi_client.py      # TAAPI.io HTTP client
│   │   └── taapi_cache.py       # Indicator response caching
│   └── options_intel/           # Vol surface builder, mispricing detection, Deribit data
├── database/
│   ├── models.py                # SQLAlchemy ORM: Trade, Position, DiaryEntry, BotState, TradeProposal, MarketData
│   └── db_manager.py            # DatabaseManager — CRUD with session context managers
├── modules/                     # Experimental / standalone modules
│   ├── polymarket_module.py     # Polymarket prediction-market integration
│   └── trader_analytics.py      # Top-trader identification
└── gui/
    ├── app.py                   # NiceGUI app setup, page routing
    ├── pages/                   # Dashboard, Recommendations, Positions, History, Market, Reasoning, Settings
    ├── components/
    │   ├── header.py            # Top navigation bar
    │   └── sidebar.py           # Side navigation menu
    └── services/
        ├── bot_service.py       # BotService — bot lifecycle, GUI API layer
        ├── state_manager.py     # Per-client connection state
        └── ui_utils.py          # is_ui_alive() — NiceGUI element lifecycle guard
```

### Key patterns

- **Callback-driven state**: `TradingBotEngine` emits `BotState` dataclass updates via callbacks to the GUI layer. The GUI never calls into the engine directly for state.
- **Two-cadence scheduler**: Perps run on a 5-min loop; options on a separate 3-hour loop via `options_scheduler.py`.
- **Multi-venue adapter**: All exchange interactions go through the `ExchangeAdapter` base class. `exchange_factory.py` routes by venue name (`hyperliquid` | `thalex`).
- **LLM → structured output**: `TradingAgent.decide()` calls OpenRouter, then `parse_decision()` validates the JSON into a `TradeDecision` dataclass. Invalid payloads raise `DecisionParseError`.
- **Dual trade-proposal model**: In-memory `TradeProposal` dataclass (`src/backend/models/trade_proposal.py`, UUID ids, state-machine methods) lives alongside the SQLAlchemy ORM model (`src/database/models.py`, auto-increment ids). The engine uses the dataclass; persistence goes through the ORM.
- **UI lifecycle guard**: All GUI timer callbacks and async handlers call `is_ui_alive()` (from `ui_utils.py`) before touching NiceGUI elements to prevent stale-client errors in the SPA.
- **DB session discipline**: Methods in `db_manager.py` that need to return data to callers must convert ORM objects to dicts **inside** the `session_scope()` block. Returning raw ORM objects causes "not bound to a Session" errors once the context manager closes.
- **Config**: All runtime config comes from `.env` → `config_loader.py` → `CONFIG` dict. Access via `CONFIG.get("key")`.
- **Imports**: Always use absolute imports from the repo root (e.g., `from src.backend.agent.decision_maker import TradingAgent`). The test `conftest.py` adds the repo root to `sys.path`.

### Venues

| Venue | Asset | Exchange class | Auth |
|-------|-------|---------------|------|
| `hyperliquid` | BTC, ETH, SOL (perps) | `HyperliquidAPI` | Private key or mnemonic |
| `thalex` | BTC (options only) | `ThalexAPI` | JWT RS512 with RSA keypair |

### Valid domain constants

```python
VALID_VENUES = {"hyperliquid", "thalex"}
VALID_ACTIONS = {"buy", "sell", "hold"}
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
```

## Development notes

- AGENTS.md contains detailed step-by-step procedures for adding exchanges, strategies, DB models, GUI pages, and config options. Read it before making structural changes.
- `.env.example` documents all environment variables. `.env` is gitignored and contains secrets.
- Thalex private keys live in `secrets/` (gitignored).
- Logs: `bot.log` (trading engine), `llm_requests.log` (LLM API calls).
- The database is SQLite by default, stored in `data/`.
- Async throughout: the trading engine, exchange adapters, and LLM client all use `asyncio`/`aiohttp`. Tests use `pytest-asyncio`.
- Two virtual environments exist: `venv` (Python 3.14, main) and `venv-gtk` (Python 3.12, GTK desktop launcher via `run-desktop.sh`).



# CODE POLICY
Do not add a footer with the trailer 'Co-authored-by'.
DO NOT WRITE COMMENTS UNLESS THEY ARE EXTREMELY NECESSARY