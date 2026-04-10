"""Abstract base class and shared dataclasses for all exchange/venue adapters.

Every venue (Hyperliquid perps, Thalex options, future TastyTrade/IBKR) implements
this contract. The bot engine never imports a venue-specific class directly — it
holds an `ExchangeAdapter` reference and dispatches to the venue resolved from
each trade decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OrderResult:
    """Normalized response returned by every adapter after submitting an order."""

    venue: str
    order_id: str
    asset: str
    side: str  # "buy" | "sell" | "tp" | "sl"
    amount: float
    status: str  # "ok" | "filled" | "resting" | "rejected" | "error"
    instrument_name: Optional[str] = None
    price: Optional[float] = None
    raw: Optional[dict] = None  # original venue payload for debugging
    error: Optional[str] = None


@dataclass
class PositionSnapshot:
    """Normalized position view across spot/perp/options venues.

    `asset` is the underlying ticker (BTC, ETH). `instrument_name` is the full
    venue-specific instrument identifier (e.g. `BTC-27JUN25-100000-C` for an
    option). For spot/perp venues `instrument_name` defaults to `asset`.
    """

    venue: str
    asset: str
    side: str  # "long" | "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    instrument_name: Optional[str] = None
    delta: Optional[float] = None  # for options; per-contract delta
    notional: Optional[float] = None
    raw: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.instrument_name is None:
            self.instrument_name = self.asset


@dataclass
class AccountState:
    """Normalized account snapshot returned by every adapter."""

    venue: str
    balance: float
    total_value: float
    positions: list[PositionSnapshot] = field(default_factory=list)
    raw: Optional[dict] = None


class ExchangeAdapter(ABC):
    """Contract every exchange adapter must implement.

    Subclasses must set the class attribute :attr:`venue` to a unique short name
    (e.g. ``"hyperliquid"``, ``"thalex"``) used by the factory and routing layer.
    """

    venue: str = "unknown"

    @abstractmethod
    async def place_buy_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        """Submit a market buy / open-long order. Implementations should round size."""

    @abstractmethod
    async def place_sell_order(self, asset: str, amount: float, slippage: float = 0.01) -> OrderResult:
        """Submit a market sell / open-short order. Implementations should round size."""

    @abstractmethod
    async def place_take_profit(self, asset: str, is_buy: bool, amount: float, tp_price: float) -> OrderResult:
        """Submit a reduce-only TP trigger order on an existing position."""

    @abstractmethod
    async def place_stop_loss(self, asset: str, is_buy: bool, amount: float, sl_price: float) -> OrderResult:
        """Submit a reduce-only SL trigger order on an existing position."""

    @abstractmethod
    async def cancel_order(self, asset: str, order_id: Any) -> dict:
        """Cancel a single order by venue-native id."""

    @abstractmethod
    async def cancel_all_orders(self, asset: str) -> dict:
        """Cancel every open order on the given asset."""

    @abstractmethod
    async def get_open_orders(self) -> list[dict]:
        """Return all open orders for the account."""

    @abstractmethod
    async def get_recent_fills(self, limit: int = 50) -> list[dict]:
        """Return up to `limit` most recent fills."""

    @abstractmethod
    async def get_user_state(self) -> AccountState:
        """Return the normalized account snapshot (balance + positions)."""

    @abstractmethod
    async def get_current_price(self, asset: str) -> float:
        """Return latest mid-price for an underlying or instrument."""

    async def connect(self) -> None:
        """Optional lifecycle hook — adapters with persistent connections override this."""
        return None

    async def disconnect(self) -> None:
        """Optional lifecycle hook — adapters with persistent connections override this."""
        return None
