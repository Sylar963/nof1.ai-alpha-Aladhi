# nof1.ai Alpha Arena - Development Skill Document

## Overview

This skill provides comprehensive guidance for interacting with, modifying, and extending the nof1.ai Alpha Arena trading bot codebase. It covers architecture, conventions, patterns, and step-by-step procedures for common development tasks.

## LLM Reference Files

- For dependency/version guidance and pytest fixture conventions, consult `.agents/skills/nof1-stack-and-testing/SKILL.md` before changing stack-level code or tests.
- Treat `requirements.txt` as the source of truth for supported dependency ranges.

---

## 1. Architecture Overview

### 1.1 Clean Architecture Layers

The codebase follows a layered architecture with Clear separation of concerns:

```
src/
├── backend/          # Application + Domain Layer
│   ├── agent/       # AI Decision Making (LLM integration)
│   ├── trading/     # Exchange Adapters (Infrastructure)
│   ├── indicators/  # Technical Analysis (External API)
│   ├── models/      # Data Transfer Objects
│   └── utils/       # Utilities
├── gui/             # UI Layer (NiceGUI)
├── database/        # Data Access Layer (SQLAlchemy)
└── llm_engine.py   # LLM Integration
```

### 1.2 Dependency Direction

- **Domain**: `src/backend/models/`, `src/backend/agent/decision_schema.py`
- **Application**: `src/backend/bot_engine.py`, `src/backend/agent/decision_maker.py`
- **Infrastructure**: `src/backend/trading/` (Exchange APIs), `src/backend/indicators/`
- **UI**: `src/gui/` (NiceGUI pages and services)

### 1.3 Core Entry Points

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 77 | App entry, NiceGUI desktop mode |
| `src/backend/bot_engine.py` | ~1005 | Core trading engine |
| `src/gui/app.py` | 130 | GUI app setup, navigation |
| `src/gui/services/bot_service.py` | 630 | Bot lifecycle management |

---

## 2. Key Modules and Responsibilities

### 2.1 Trading Engine (`src/backend/bot_engine.py`)

**Class**: `TradingBotEngine` (lines 45-1005)

- Main trading loop (`_main_loop()`, lines 214-661)
- Multi-venue routing: Hyperliquid perps + Thalex options
- Manual/Auto trading modes
- Bot state management with callbacks
- Risk management (position sizing, TP/SL orders)

**Key Methods**:
- `start()` - Start the trading bot
- `stop()` - Stop the trading bot
- `_main_loop()` - Main trading cycle
- `_execute_proposal()` - Execute trade proposals

### 2.2 AI Agent (`src/backend/agent/decision_maker.py`)

**Class**: `TradingAgent` (lines 10-485)

- LLM integration via OpenRouter
- Multi-venue decision schema generation
- Technical indicator tool calls to TAAPI
- Structured output parsing with JSON schema validation

**Key Methods**:
- `decide()` - Generate trading decision
- `_call_llm()` - Call LLM with prompts
- `_parse_response()` - Parse LLM response

### 2.3 Decision Schema (`src/backend/agent/decision_schema.py`)

**Class**: `TradeDecision` (lines 51-91)

- Normalized trade decision object
- Multi-venue support: `hyperliquid` | `thalex`
- Options strategies: `credit_put`, `credit_spread`, `delta_hedged`

**Validation Function**:
- `parse_decision()` (lines 139-187) - Validates and parses LLM decisions

### 2.4 Exchange Adapters

**Base Class**: `ExchangeAdapter` (`src/backend/trading/exchange_adapter.py`, lines 69-125)

**Hyperliquid Integration** (`src/backend/trading/hyperliquid_api.py`, 422 lines):
- Class `HyperliquidAPI`
- Wallet authentication
- Order placement: market, limit, TP, SL
- User state and positions

**Thalex Options** (`src/backend/trading/thalex_api.py`, 442 lines):
- Class `ThalexAPI`
- WebSocket connection with JWT auth
- Options instrument resolution
- Risk caps validation

### 2.5 Database Models (`src/database/models.py`)

**SQLAlchemy ORM Models**:
| Model | Lines | Purpose |
|-------|-------|---------|
| `Trade` | 30-93 | Executed trades history |
| `Position` | 95-143 | Open positions (multi-venue) |
| `DiaryEntry` | 145-183 | AI decision log |
| `BotState` | 186-240 | Performance snapshots |
| `TradeProposal` | 242-297 | Manual mode proposals |
| `MarketData` | 300-341 | Historical OHLCV |

**Database Manager**: `DatabaseManager` (`src/database/db_manager.py`, 634 lines)
- Full CRUD operations
- Session management with context managers

---

## 3. Code Conventions and Patterns

### 3.1 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `TradingBotEngine`, `HyperliquidAPI` |
| Functions/Methods | snake_case | `place_buy_order()`, `_execute_proposal()` |
| Constants | UPPER_SNAKE_CASE | `VALID_VENUES`, `HYPERLIQUID_NETWORK` |
| Private Methods | Leading underscore | `_main_loop()`, `_on_state_update()` |
| Files | snake_case | `bot_engine.py`, `exchange_adapter.py` |

### 3.2 Dataclass Patterns

```python
from dataclasses import dataclass, field

@dataclass
class TradeDecision:
    asset: str
    action: str
    rationale: str
    venue: str = "hyperliquid"  # Default values at end
    legs: list[OptionsLeg] = field(default_factory=list)
```

### 3.3 Error Handling Patterns

**Custom Exceptions** (`src/backend/agent/decision_schema.py`, lines 36-38):
```python
class DecisionParseError(ValueError):
    """Raised when an LLM payload cannot be coerced into a TradeDecision."""
```

**Broad Exception Handling** (optional dependencies):
```python
try:
    from src.backend.trading.thalex_api import ThalexAPI
    self.thalex = ThalexAPI()
except Exception as exc:
    self.logger.warning("Thalex venue not initialized: %s", exc)
```

**Retry Logic with Backoff** (`src/backend/trading/hyperliquid_api.py`, lines 107-151):
```python
async def _retry(self, fn, *args, max_attempts: int = 3, backoff_base: float = 0.5, ...):
    # Exponential backoff retry logic
```

### 3.4 Configuration Management

**Pattern** (`src/backend/config_loader.py`):
```python
from src.backend.config_loader import CONFIG

# Access config values
api_key = CONFIG.get("openrouter_api_key")
assets = CONFIG.get("assets") or "BTC,ETH"
```

**Environment Variables** (`.env`):
- API keys: `TAAPI_API_KEY`, `OPENROUTER_API_KEY`, `HYPERLIQUID_PRIVATE_KEY`
- Runtime: `ASSETS`, `INTERVAL`, `TRADING_MODE` (auto/manual)
- Network: `HYPERLIQUID_NETWORK`, `THALEX_NETWORK`

### 3.5 Logging Pattern

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

---

## 4. Adding New Features

### 4.1 Adding a New Exchange Adapter

**Step 1**: Create the adapter class in `src/backend/trading/`

```python
# src/backend/trading/new_exchange_api.py
from src.backend.trading.exchange_adapter import ExchangeAdapter, OrderResult, PositionSnapshot

class NewExchangeAPI(ExchangeAdapter):
    def __init__(self, api_key: str, secret: str, network: str = "mainnet"):
        self.api_key = api_key
        self.secret = secret
        self.network = network
        self.logger = logging.getLogger(__name__)
    
    async def place_market_order(self, asset: str, side: str, size: float) -> OrderResult:
        # Implementation
        pass
    
    async def place_limit_order(self, asset: str, side: str, size: float, price: float) -> OrderResult:
        # Implementation
        pass
    
    async def get_positions(self) -> List[PositionSnapshot]:
        # Implementation
        pass
    
    async def get_account_state(self) -> AccountState:
        # Implementation
        pass
```

**Step 2**: Register in Exchange Factory (`src/backend/trading/exchange_factory.py`)

**Step 3**: Update Config (`src/backend/config_loader.py`)

**Step 4**: Add tests in `tests/test_exchange_adapter.py`

---

### 4.2 Adding a New Trading Strategy

**Step 1**: Define Strategy Intent (`src/backend/trading/options.py`)

```python
@dataclass
class NewStrategyIntent:
    """New strategy parameters."""
    asset: str
    target_size: float
    # Add strategy-specific parameters
```

**Step 2**: Implement Executor (`src/backend/trading/options_strategies.py`)

```python
class NewStrategyExecutor:
    def __init__(self, exchange: ExchangeAdapter):
        self.exchange = exchange
    
    async def execute(self, intent: NewStrategyIntent) -> OrderResult:
        # Implementation
        pass
```

**Step 3**: Update Decision Schema (`src/backend/agent/decision_schema.py`)

```python
VALID_STRATEGIES = {
    # Existing...
    "new_strategy",
}
```

**Step 4**: Add to Decision Maker (`src/backend/agent/decision_maker.py`)

- Update prompts
- Add tool definitions

---

### 4.3 Adding a New Database Model

**Step 1**: Define Model (`src/database/models.py`)

```python
class NewEntity(Base):
    __tablename__ = 'new_entities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
```

**Step 2**: Add CRUD Methods (`src/database/db_manager.py`)

```python
def create_new_entity(self, name: str) -> NewEntity:
    entity = NewEntity(name=name)
    self.session.add(entity)
    self.session.commit()
    return entity

def get_new_entities(self) -> List[NewEntity]:
    return self.session.query(NewEntity).all()
```

---

### 4.4 Adding a New GUI Page

**Step 1**: Create Page (`src/gui/pages/new_page.py`)

```python
from nicegui import ui

def create_new_page():
    with ui.column():
        ui.label("New Page")
        # Add components
```

**Step 2**: Register in App (`src/gui/app.py`)

```python
from src.gui.pages.new_page import create_new_page

# Add to navigation
app.add_page("/new", create_new_page, ...)
```

---

### 4.5 Adding a New Configuration Option

**Step 1**: Add to Config Loader (`src/backend/config_loader.py`)

```python
CONFIG = {
    # Existing...
    "new_option": _get_env("NEW_OPTION", "default_value"),
}
```

**Step 2**: Document in `.env.example` and `README.md`

**Step 3**: Use in Code

```python
from src.backend.config_loader import CONFIG

value = CONFIG.get("new_option")
```

---

## 5. Removing Features

### 5.1 Removing an Exchange Adapter

1. **Remove from Exchange Factory** (`src/backend/trading/exchange_factory.py`)
2. **Remove imports** from `bot_engine.py`
3. **Remove configuration options** from `config_loader.py`
4. **Update tests**

### 5.2 Removing a Strategy

1. **Remove from VALID_STRATEGIES** (`src/backend/agent/decision_schema.py`)
2. **Remove from Decision Maker** prompts
3. **Remove executor** if standalone

---

## 6. Testing Guidelines

### 6.1 Test Structure

```
tests/
├── conftest.py           # Pytest configuration
├── test_decision_schema.py
├── test_options_strategies.py
├── test_exchange_adapter.py
└── test_database_schema.py
```

### 6.2 Running Tests

```bash
pytest                    # Run all tests
pytest -v                # Verbose output
pytest tests/test_*.py    # Specific test file
```

### 6.3 Writing Tests

```python
# tests/test_decision_schema.py
import pytest
from src.backend.agent.decision_schema import TradeDecision, parse_decision

def test_parse_valid_decision():
    payload = {
        "asset": "BTC",
        "action": "buy",
        "venue": "hyperliquid",
        "allocation_usd": 1000.0,
    }
    decision = parse_decision(payload)
    assert decision.asset == "BTC"
    assert decision.action == "buy"
```

---

## 7. Common Development Tasks

### 7.1 Update LLM Model

Edit `src/backend/config_loader.py`:
```python
"llm_model": _get_env("LLM_MODEL", "x-ai/grok-4"),
```

Or set environment variable:
```bash
export LLM_MODEL="anthropic/claude-sonnet-4-5"
```

### 7.2 Add New Asset

1. Edit `.env`:
```bash
export ASSETS="BTC,ETH,SOL,NEW_ASSET"
```

2. Verify in Exchange API (`hyperliquid_api.py`)

### 7.3 Change Trading Interval

```bash
export INTERVAL="1h"  # Options: 5m, 15m, 1h, 4h, 1d
```

### 7.4 Enable/Disable Features

```bash
export TRADING_MODE="auto"  # or "manual"
export REASONING_ENABLED="true"
```

---

## 8. Important File Locations

| Component | File Path |
|-----------|-----------|
| Trading Engine | `src/backend/bot_engine.py` |
| AI Agent | `src/backend/agent/decision_maker.py` |
| Decision Schema | `src/backend/agent/decision_schema.py` |
| Config Loader | `src/backend/config_loader.py` |
| Hyperliquid API | `src/backend/trading/hyperliquid_api.py` |
| Thalex API | `src/backend/trading/thalex_api.py` |
| DB Models | `src/database/models.py` |
| DB Manager | `src/database/db_manager.py` |
| GUI App | `src/gui/app.py` |
| Main Entry | `main.py` |

---

## 9. Quick Reference

### Import Pattern
```python
# Local imports
from src.backend.agent.decision_maker import TradingAgent
from src.backend.config_loader import CONFIG
from src.database.db_manager import DatabaseManager

# External imports
import aiohttp
import pandas as pd
from nicegui import ui
```

### Key Constants
```python
VALID_VENUES = {"hyperliquid", "thalex"}
VALID_ACTIONS = {"buy", "sell", "hold"}
VALID_STRATEGIES = {"credit_put", "credit_spread", "long_call_delta_hedged", "long_put_delta_hedged"}
```

### Callback Pattern
```python
# Bot engine callback
on_state_update: Optional[Callable[[BotState], None]]
on_trade_executed: Optional[Callable[[Dict], None]]
on_error: Optional[Callable[[str], None]]
```

---

*This skill was generated from analyzing the nof1.ai Alpha Arena codebase. Last updated: April 2026.*
