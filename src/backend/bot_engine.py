"""
Trading Bot Engine - Core trading logic separated from UI
Refactored from ai-trading-agent/src/main.py
"""

import asyncio
import json
import logging
import os
import tempfile
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from src.backend.agent.decision_maker import TradingAgent
from src.backend.agent.decision_schema import DecisionParseError, parse_decision
from src.backend.agent.options_llm_lifecycle import OptionsLLMLifecycle
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_client import TAAPIClient
from src.backend.indicators.indicator_engine import build_indicator_bundle
from src.backend.models.trade_proposal import TradeProposal
from src.backend.trading.delta_hedge_manager import DeltaHedgeManager
from src.backend.trading.hyperliquid_api import HyperliquidAPI
from src.backend.trading.options_scheduler import (
    OptionsScheduler,
    OptionsSchedulerConfig,
)
from src.backend.trading.options_strategies import DeltaHedger, OptionsExecutor
from src.backend.utils.prompt_utils import json_default


# Failed proposals linger in pending_proposals so the UI can offer Retry.
# After this many seconds the engine drops them so state doesn't accumulate
# indefinitely and stale market context isn't retried against fresh prices.
FAILED_PROPOSAL_TTL_SECONDS = int(CONFIG.get("FAILED_PROPOSAL_TTL_SECONDS") or 30 * 60)

# Circuit breaker: auto-pauses trading after repeated execution failures or a
# session drawdown. Resets on manual resume. These are guardrails for the
# autonomous loop, distinct from the hard bot-stop at MAX_CONSECUTIVE_ERRORS
# (which trips on any cycle-level exception, e.g. LLM/RPC failures).
CIRCUIT_BREAKER_CONSECUTIVE_FAILS = int(CONFIG.get("CIRCUIT_BREAKER_CONSECUTIVE_FAILS") or 3)
CIRCUIT_BREAKER_DRAWDOWN_PCT = float(CONFIG.get("CIRCUIT_BREAKER_DRAWDOWN_PCT") or 5.0)


@dataclass
class BotState:
    """Bot state for UI updates"""
    is_running: bool = False
    balance: float = 0.0
    total_value: float = 0.0
    balance_breakdown: Dict[str, float] = field(default_factory=dict)
    total_value_breakdown: Dict[str, float] = field(default_factory=dict)
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    positions: List[Dict] = field(default_factory=list)
    active_trades: List[Dict] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    recent_fills: List[Dict] = field(default_factory=list)
    market_data: List[Dict] = field(default_factory=list)  # Market data for dashboard
    hedge_status: Dict = field(default_factory=dict)
    hedge_metrics: List[Dict] = field(default_factory=list)
    pending_proposals: List[Dict] = field(default_factory=list)  # Pending trade proposals (manual mode)
    last_reasoning: Dict = field(default_factory=dict)  # Composite view (perps + options) consumed by GUI
    last_perps_reasoning: Dict = field(default_factory=dict)
    last_options_reasoning: Dict = field(default_factory=dict)
    last_update: str = ""
    error: Optional[str] = None
    invocation_count: int = 0
    # Circuit-breaker / kill-switch state surfaced to the UI
    is_paused: bool = False
    pause_reason: Optional[str] = None
    peak_account_value: float = 0.0
    drawdown_pct: float = 0.0
    execution_failure_streak: int = 0


class TradingBotEngine:
    """
    Core trading bot engine independent of UI.
    Communicates with GUI via callback system.
    """

    def __init__(
        self,
        assets: List[str],
        interval: str,
        delta_hedge_enabled: bool = True,
        on_state_update: Optional[Callable[[BotState], None]] = None,
        on_trade_executed: Optional[Callable[[Dict], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize trading bot engine.

        Args:
            assets: List of trading assets (e.g., ["BTC", "ETH"])
            interval: Trading interval (e.g., "5m", "1h")
            on_state_update: Callback for state updates
            on_trade_executed: Callback when trade is executed
            on_error: Callback for errors
        """
        self.assets = assets
        self.interval = interval
        self.on_state_update = on_state_update
        self.on_trade_executed = on_trade_executed
        self.on_error = on_error

        # Logging (initialize first!)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("bot.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize trading components
        self.taapi = TAAPIClient()
        self.hyperliquid = HyperliquidAPI()
        self.agent = TradingAgent()

        # Optional Thalex options venue. Constructed lazily so a missing
        # config doesn't break Hyperliquid-only deployments.
        self.thalex = None
        self.options_executor: Optional[OptionsExecutor] = None
        self.hedge_manager: Optional[DeltaHedgeManager] = None
        self.options_scheduler: Optional[OptionsScheduler] = None
        self._delta_hedge_enabled = bool(delta_hedge_enabled)
        self._hedge_audit_task: Optional[asyncio.Task] = None
        self._hedge_reconcile_interval_seconds = int(
            CONFIG.get("thalex_hedge_reconcile_interval_seconds") or 15
        )
        self.options_llm_lifecycle: Optional[OptionsLLMLifecycle] = None
        self._latest_options_context = None  # populated by the 15m surface refresh task
        if CONFIG.get("thalex_key_id") and CONFIG.get("thalex_private_key_path"):
            try:
                from src.backend.trading.thalex_api import ThalexAPI
                self.thalex = ThalexAPI()
                self.options_executor = OptionsExecutor(
                    thalex=self.thalex,
                    hyperliquid=self.hyperliquid,
                    delta_hedge_enabled=self._delta_hedge_enabled,
                )
                threshold = float(CONFIG.get("thalex_delta_threshold") or 0.02)
                self.hedge_manager = DeltaHedgeManager(
                    thalex=self.thalex,
                    hyperliquid=self.hyperliquid,
                    hedger=DeltaHedger(threshold=threshold),
                    enabled=self._delta_hedge_enabled,
                )
                self.logger.info(
                    "Thalex options venue enabled (network=%s, hedge_threshold=%.4f BTC)",
                    self.thalex.network_name, threshold,
                )

                # Cache one LLM client for the bot's lifetime so we don't
                # leak an aiohttp session every 3-hour decision cycle.
                self.options_llm_lifecycle = OptionsLLMLifecycle(
                    api_key=CONFIG.get("openrouter_api_key"),
                    logger=self.logger,
                )

                # Cache one Deribit client for the bot's lifetime.  Creating
                # a fresh client on every 15-min vol surface refresh meant a
                # brand-new aiohttp session each time — ~200ms overhead plus
                # a small FD leak if close() ever failed.
                self._deribit_client = None

                # Wire the two-cadence scheduler. Disabled by default — set
                # OPTIONS_SCHEDULER_ENABLED=1 to turn on the live decision loop.
                if CONFIG.get("options_scheduler_enabled"):
                    scheduler_config = OptionsSchedulerConfig(
                        vol_surface_interval_seconds=float(
                            CONFIG.get("options_vol_surface_interval_seconds") or 900
                        ),
                        options_decision_interval_seconds=float(
                            CONFIG.get("options_decision_interval_seconds") or 10800
                        ),
                    )
                    self.options_scheduler = OptionsScheduler(
                        config=scheduler_config,
                        refresh_vol_surface=self._refresh_options_surface,
                        run_options_decision=self._run_options_decision_cycle,
                    )
                    self.logger.info(
                        "OptionsScheduler enabled (surface=%.0fs, decision=%.0fs)",
                        scheduler_config.vol_surface_interval_seconds,
                        scheduler_config.options_decision_interval_seconds,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Thalex venue not initialized: %s", exc)

        # Bot state
        self.state = BotState()
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Internal state tracking (from original main.py)
        self.start_time: Optional[datetime] = None
        self.invocation_count = 0
        self.trade_log: List[float] = []  # For Sharpe calculation
        self._previous_total_value: Optional[float] = None
        self.active_trades: List[Dict] = []  # Local tracking of open positions
        self.recent_events: deque = deque(maxlen=200)
        self.initial_account_value: Optional[float] = None
        self._last_thalex_execution: Dict[str, Any] = {}
        self.price_history: Dict[str, deque] = {asset: deque(maxlen=60) for asset in assets}
        # Per-asset max leverage rarely changes between cycles; cache to avoid
        # re-querying the venue meta on every decision + preflight call.
        self._max_leverage_cache: Dict[str, int] = {}

        # Circuit breaker / kill switch. ``is_paused`` skips the trading cycle
        # (market data + LLM + execution) until the user resumes via the UI.
        # The streak counter is incremented on trade-execution failures only —
        # not on decision-level errors (which have their own handling).
        self.is_paused: bool = False
        self.pause_reason: Optional[str] = None
        self.peak_account_value: float = 0.0
        self._execution_failure_streak: int = 0

        # Manual trading mode
        self.trading_mode = CONFIG.get("trading_mode", "auto").lower()
        self.pending_proposals: List[TradeProposal] = []
        self.logger.info(f"Trading mode: {self.trading_mode.upper()}")

        # File paths
        self.diary_path = Path("data/diary.jsonl")
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)
        # Persisted AI-opened trade list. Without this, a bot restart would
        # lose the record of which positions were opened by the agent, and
        # the GUI would label them as "External" even though the AI opened them.
        self.active_trades_path = Path("data/active_trades.json")
        self._load_active_trades()

    def _load_active_trades(self) -> None:
        """Rehydrate ``self.active_trades`` from disk if the file exists."""
        try:
            if self.active_trades_path.exists():
                with open(self.active_trades_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.active_trades = data
                    self.logger.info(
                        "Loaded %d active trades from %s",
                        len(self.active_trades), self.active_trades_path,
                    )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to load active_trades.json: %s", exc)

    def _save_active_trades(self) -> None:
        """Persist ``self.active_trades`` to disk so it survives restarts.

        Writes to a temp file in the same directory, fsyncs, then atomically
        renames into place. This avoids the zero-length-file hazard where a
        crash mid-write would leave ``active_trades.json`` truncated and
        wipe the bot's memory of open positions on the next startup.
        """
        try:
            path = self.active_trades_path
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.active_trades, f, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to save active_trades.json: %s", exc)

    async def _get_max_leverage_cached(self, asset: str) -> int:
        """Return cached max leverage for ``asset``. Falls back to 1 on failure."""
        if asset in self._max_leverage_cache:
            return self._max_leverage_cache[asset]
        try:
            lev = await self.hyperliquid.get_max_leverage(asset)
            self._max_leverage_cache[asset] = max(int(lev or 1), 1)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("get_max_leverage(%s) failed, defaulting to 1: %s", asset, exc)
            self._max_leverage_cache[asset] = 1
        return self._max_leverage_cache[asset]

    async def _get_buying_power_snapshot(self) -> Dict[str, Any]:
        """Build the buying-power block the LLM needs to size trades responsibly.

        Returns withdrawable / free_margin / account_value plus per-asset max
        leverage and the derived cap ``max_new_notional_by_asset`` that mirrors
        ``_hl_margin_preflight`` (available × leverage / 1.05 buffer). If the
        collateral lookup fails, returns a conservative all-zeros dict so the
        LLM sees "no buying power" rather than a missing field.
        """
        if not self.hyperliquid:
            return {}
        try:
            info = await self.hyperliquid.get_free_margin_info()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("buying_power snapshot: get_free_margin_info failed: %s", exc)
            info = {"withdrawable": 0.0, "free_margin": 0.0, "account_value": 0.0, "total_margin_used": 0.0}

        withdrawable = float(info.get("withdrawable") or 0.0)
        free_margin = float(info.get("free_margin") or 0.0)
        # Mirror preflight: pick the conservative positive value as "what we can open with".
        candidates = [v for v in (withdrawable, free_margin) if v and v > 0]
        available = min(candidates) if candidates else 0.0

        max_leverage_by_asset: Dict[str, int] = {}
        max_new_notional_by_asset: Dict[str, float] = {}
        for asset in self.assets:
            lev = await self._get_max_leverage_cached(asset)
            max_leverage_by_asset[asset] = lev
            # available × leverage is the gross notional cap; /1.05 matches the
            # 5% buffer preflight demands for slippage + fees, so any number
            # the LLM picks at-or-below this will pass the guard.
            max_new_notional_by_asset[asset] = round((available * lev) / 1.05, 2) if available > 0 else 0.0

        return {
            "withdrawable": round(withdrawable, 2),
            "free_margin": round(free_margin, 2),
            "account_value": round(float(info.get("account_value") or 0.0), 2),
            "total_margin_used": round(float(info.get("total_margin_used") or 0.0), 2),
            "max_leverage_by_asset": max_leverage_by_asset,
            "max_new_notional_by_asset": max_new_notional_by_asset,
        }

    async def _hl_margin_preflight(self, asset: str, notional_usd: float) -> None:
        """Raise :class:`RuntimeError` if Hyperliquid can't cover the notional.

        Cross-margin cost of a perp is roughly ``notional / maxLeverage``. We
        add a 5% buffer for entry slippage + taker fees + any variation
        between quote-time and fill-time.

        Fail-open vs fail-closed policy:
        - ``get_free_margin_info`` failures → propagate (fail CLOSED). We
          have no collateral number so refusing the trade is the only safe
          move — the alternative (skipping the check) would re-open the
          silent-failure bug this preflight exists to close.
        - ``get_max_leverage`` failures → fail OPEN. Max leverage is a
          nice-to-have sizing hint; falling back to a conservative 1× would
          reject many legitimate trades on a transient meta hiccup. Treat
          it as unbounded (``inf``) so the leverage divisor vanishes and the
          check only trips on truly missing collateral.
        """
        if notional_usd <= 0 or not self.hyperliquid:
            return
        info = await self.hyperliquid.get_free_margin_info()
        try:
            max_leverage = await self.hyperliquid.get_max_leverage(asset)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                "Margin preflight: get_max_leverage(%s) failed, treating as unbounded: %s",
                asset, exc,
            )
            max_leverage = float("inf")
        effective_leverage = max_leverage if max_leverage and max_leverage > 0 else 1
        required = (notional_usd / effective_leverage) * 1.05
        # Use the conservative of withdrawable vs. free_margin — both model
        # "what can I open with right now" but the venue's own ``withdrawable``
        # is the ground truth whenever it's populated.
        available = min(
            v for v in (info.get("withdrawable"), info.get("free_margin")) if v and v > 0
        ) if any((info.get("withdrawable"), info.get("free_margin"))) else 0.0
        if available < required:
            raise RuntimeError(
                f"Insufficient Hyperliquid margin for {asset}: "
                f"required ≈ ${required:,.2f} "
                f"(notional ${notional_usd:,.2f} / {max_leverage}x + 5% buffer), "
                f"available ${available:,.2f} "
                f"(withdrawable={info.get('withdrawable', 0):,.2f}, "
                f"free_margin={info.get('free_margin', 0):,.2f})"
            )

    @staticmethod
    def _hl_validate_response(order_result, context: str) -> None:
        """Raise :class:`RuntimeError` when the Hyperliquid SDK response indicates rejection.

        Hyperliquid's insufficient-margin path returns a top-level ``"ok"`` with
        a per-status ``{"error": "..."}`` entry — so the naive "no exception =
        success" read is wrong. ``context`` is embedded in the error for
        trace-ability (``"buy BTC x0.01"``).
        """
        ok, reason = HyperliquidAPI.parse_order_response(order_result)
        if not ok:
            raise RuntimeError(f"Hyperliquid {context} rejected: {reason}")

    def _create_thalex_proposal(self, decision_payload: Dict) -> TradeProposal:
        """Build a manual-approval proposal for a Thalex decision payload.

        Size is derived in priority order: top-level ``contracts`` →
        ``target_gamma_btc`` → sum of ``legs[].contracts`` (multi-leg
        strategies carry their size on the legs, not at the top level).
        """
        decision = parse_decision(decision_payload)
        legs_total = sum(float(leg.contracts or 0.0) for leg in decision.legs)
        size = float(
            decision.contracts
            or decision.target_gamma_btc
            or legs_total
            or 0.0
        )
        if size <= 0:
            raise ValueError(
                f"Thalex proposal has zero size: contracts={decision.contracts}, "
                f"target_gamma_btc={decision.target_gamma_btc}, legs_total={legs_total}"
            )
        return TradeProposal(
            venue="thalex",
            asset=decision.asset,
            action=decision.action,
            confidence=float(decision_payload.get("confidence") or 75.0),
            entry_price=0.0,
            size=size,
            allocation=0.0,
            rationale=decision.rationale,
            market_conditions={
                "venue": "thalex",
                "strategy": decision.strategy,
                "underlying": decision.underlying,
                "kind": decision.kind,
                "tenor_days": decision.tenor_days,
                "target_strike": decision.target_strike,
                "target_delta": decision.target_delta,
                "contracts": decision.contracts,
                "target_gamma_btc": decision.target_gamma_btc,
                "legs": [
                    {
                        "kind": leg.kind,
                        "side": leg.side,
                        "contracts": leg.contracts,
                        "target_strike": leg.target_strike,
                        "target_delta": leg.target_delta,
                        "tenor_days": leg.tenor_days,
                    }
                    for leg in decision.legs
                ],
                "decision_payload": decision_payload,
            },
        )

    async def _execute_thalex_decision(self, decision_payload: Dict) -> tuple[bool, str]:
        """Route a Thalex options decision through the OptionsExecutor.

        Skips silently with a logged warning when Thalex is not configured. The
        executor is responsible for risk-cap preflight, intent → instrument
        resolution, leg orders, and the perp delta hedge.
        """
        self._last_thalex_execution = {}
        if self.options_executor is None or self.thalex is None:
            message = f"Thalex venue is not configured: {decision_payload}"
            self.logger.warning(message)
            return False, message
        try:
            decision = parse_decision(decision_payload)
        except DecisionParseError as exc:
            message = f"Invalid Thalex decision payload: {exc}"
            self.logger.error("%s — %s", message, decision_payload)
            return False, message

        try:
            # Ensure WS is connected before the first request.
            await self.thalex.connect()
        except Exception as exc:  # pylint: disable=broad-except
            message = f"Thalex connect failed: {exc}"
            self.logger.error(message)
            return False, message

        open_count = await self._live_thalex_open_positions_count()

        # Pre-trade margin check against Thalex cash collateral. We don't know
        # the exact premium without another ticker RPC, so we use a rough
        # proxy: contracts × underlying_spot × 1% (typical BTC option premium
        # fraction) as the required collateral. If this fails the executor
        # never hits the wire and the reason propagates to the caller.
        #
        # ``getattr``s throughout make this tolerant of the test doubles that
        # build the engine via ``TradingBotEngine.__new__`` and set only the
        # attributes they care about; the production adapter defines both
        # ``margin_preflight`` and ``get_current_price``.
        margin_preflight = getattr(self.thalex, "margin_preflight", None)
        hl = getattr(self, "hyperliquid", None)
        if callable(margin_preflight) and hl is not None:
            try:
                contracts_est = float(
                    decision.contracts
                    or decision.target_gamma_btc
                    or sum((getattr(leg, "contracts", 0.0) or 0.0) for leg in (decision.legs or []))
                )
            except Exception:  # pylint: disable=broad-except
                contracts_est = 0.0
            if contracts_est > 0:
                underlying = decision.underlying or decision.asset or "BTC"
                # Spot fetch must fail CLOSED — if we can't size the
                # collateral requirement, silently skipping the check lets
                # margin-insolvent trades slip through. Surface the error
                # and refuse the trade until pricing recovers.
                try:
                    spot_est = await hl.get_current_price(underlying)
                except Exception as exc:  # pylint: disable=broad-except
                    message = (
                        f"Thalex margin preflight aborted: spot price lookup for "
                        f"{underlying} failed ({exc}); refusing trade"
                    )
                    self.logger.error(message)
                    return False, message
                if spot_est is None or float(spot_est) <= 0:
                    message = (
                        f"Thalex margin preflight aborted: spot price for {underlying} "
                        f"unavailable (got {spot_est!r}); refusing trade"
                    )
                    self.logger.error(message)
                    return False, message
                required_usd = contracts_est * float(spot_est) * 0.01
                ok_margin, reason = await margin_preflight(required_usd)
                if not ok_margin:
                    message = f"Thalex margin preflight failed: {reason}"
                    self.logger.error(message)
                    return False, message

        result = await self.options_executor.execute(decision, open_positions_count=open_count)
        if result.ok:
            hedge_orders = []
            if self.hedge_manager is not None:
                try:
                    hedge_orders = await self.hedge_manager.reconcile(
                        underlying=(decision.underlying or decision.asset)
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("Delta hedge reconcile failed after %s: %s", decision.strategy, exc)

            self.logger.info(
                "Thalex %s executed: thalex=%d hl=%d",
                decision.strategy,
                len(result.thalex_orders),
                len(result.hyperliquid_orders) + len(hedge_orders),
            )
            instrument_name = ""
            instrument_names = []
            if result.thalex_orders:
                first = result.thalex_orders[0]
                instrument_name = getattr(first, "instrument_name", None) or getattr(first, "asset", "") or ""
                instrument_names = [
                    getattr(order, "instrument_name", None) or getattr(order, "asset", "") or ""
                    for order in result.thalex_orders
                ]
            execution_price = await self._resolve_thalex_execution_price(
                [name for name in instrument_names if name]
            )
            single_instrument_names = [name for name in instrument_names if name]
            exit_amount = 0.0
            if len(single_instrument_names) == 1:
                exit_amount = float(
                    getattr(result.thalex_orders[0], "amount", None)
                    or decision.contracts
                    or 0.0
                )

            if decision.tp_price is not None and decision.sl_price is not None and len(single_instrument_names) == 1:
                try:
                    await self.thalex.place_bracket_order(
                        single_instrument_names[0],
                        decision.action == "buy",
                        exit_amount,
                        float(decision.sl_price),
                        float(decision.tp_price),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("Failed to place Thalex bracket exit for %s: %s", single_instrument_names[0], exc)
            elif decision.tp_price is not None and len(single_instrument_names) == 1:
                try:
                    await self.thalex.place_take_profit(
                        single_instrument_names[0],
                        decision.action == "buy",
                        exit_amount,
                        float(decision.tp_price),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("Failed to place Thalex take-profit for %s: %s", single_instrument_names[0], exc)
            elif decision.sl_price is not None and len(single_instrument_names) == 1:
                try:
                    await self.thalex.place_stop_loss(
                        single_instrument_names[0],
                        decision.action == "buy",
                        exit_amount,
                        float(decision.sl_price),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("Failed to place Thalex stop-loss for %s: %s", single_instrument_names[0], exc)
            elif (decision.tp_price is not None or decision.sl_price is not None) and len(single_instrument_names) != 1:
                self.logger.warning(
                    "Skipping native Thalex TP/SL placement for multi-leg strategy %s (instruments=%s)",
                    decision.strategy,
                    single_instrument_names,
                )

            self._last_thalex_execution = {
                "instrument_names": [name for name in instrument_names if name],
                "execution_price": execution_price,
                "open_positions_count": open_count,
            }
            self.active_trades.append({
                "venue": "thalex",
                "asset": decision.asset,
                "instrument_name": instrument_name,
                "instrument_names": [name for name in instrument_names if name],
                "strategy": decision.strategy,
                "rationale": decision.rationale,
                "thalex_orders": [o.order_id for o in result.thalex_orders],
                "hyperliquid_orders": [
                    o.order_id for o in (result.hyperliquid_orders + hedge_orders)
                ],
                "execution_price": execution_price,
                "opened_at": datetime.now(UTC).isoformat(),
            })
            self._save_active_trades()
            return True, ""
        else:
            self.logger.warning("Thalex decision rejected: %s", result.reason)
            return False, result.reason

    def _handle_execution_failure(self, venue: str, asset: str, reason: str) -> None:
        """Propagate an execution failure through the bot's normal error path."""
        message = f"{venue} execution failed for {asset}: {reason or 'unknown reason'}"
        self.logger.error(message)
        self.state.error = message
        if self.on_error:
            self.on_error(message)

    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            self.logger.warning("Bot already running")
            return

        self.is_running = True
        self.state.is_running = True
        self.start_time = datetime.now(UTC)
        self.invocation_count = 0

        # Get initial account value.
        # Run Hyperliquid + Thalex fetches in parallel and bound the Thalex call
        # so a flaky options venue can never stall the UI on "Starting...". The
        # 5s cap is well over normal round-trip time (~1s) but short enough
        # that auth/whitelist failures don't block the transition to Running.
        try:
            hl_task = asyncio.create_task(self.hyperliquid.get_user_state())
            if self.thalex is not None:
                thalex_task = asyncio.create_task(
                    asyncio.wait_for(self.thalex.get_user_state(), timeout=5.0)
                )
            else:
                thalex_task = None

            user_state = await hl_task
            thalex_state = None
            if thalex_task is not None:
                try:
                    thalex_state = await thalex_task
                except (asyncio.TimeoutError, Exception) as exc:  # pylint: disable=broad-except
                    self.logger.warning("Failed to load Thalex initial account value: %s", exc)
            snapshot = self._account_snapshot(user_state, thalex_state)
            self.initial_account_value = snapshot['total_value']
            if self.initial_account_value == 0.0:
                self.initial_account_value = snapshot['balance'] or 10000.0
        except Exception as e:
            self.logger.error(f"Failed to get initial account value: {e}")
            self.initial_account_value = 10000.0

        self._task = asyncio.create_task(self._main_loop())

        # Spin up the options scheduler in parallel with the main perps loop
        # if it was wired in __init__. The scheduler runs its two background
        # tasks (vol surface refresh + options decision) independently of the
        # 5m perps cadence.
        if self.options_scheduler is not None:
            try:
                await self.options_scheduler.start()
                self.logger.info("OptionsScheduler started")
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error("Failed to start OptionsScheduler: %s", exc)

        if self.hedge_manager is not None:
            try:
                await asyncio.wait_for(self.hedge_manager.reconcile(), timeout=8.0)
            except (asyncio.TimeoutError, Exception) as exc:  # pylint: disable=broad-except
                self.logger.warning("Initial delta hedge reconcile failed: %s", exc)
            self._hedge_audit_task = asyncio.create_task(self._hedge_audit_loop())

        self.logger.info(f"Bot started - Assets: {self.assets}, Interval: {self.interval}")
        self._notify_state_update()

    async def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return

        self.is_running = False
        self.state.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._hedge_audit_task is not None:
            self._hedge_audit_task.cancel()
            try:
                await self._hedge_audit_task
            except asyncio.CancelledError:
                pass
            self._hedge_audit_task = None

        if self.options_scheduler is not None:
            try:
                await self.options_scheduler.stop()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("OptionsScheduler stop failed: %s", exc)

        if self.hedge_manager is not None:
            try:
                await self.hedge_manager.close()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("DeltaHedgeManager close failed: %s", exc)

        # Release the cached aiohttp session inside the LLM client. close()
        # is idempotent and exception-safe so this is always cheap.
        if self.options_llm_lifecycle is not None:
            try:
                await self.options_llm_lifecycle.close()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("OptionsLLMLifecycle close failed: %s", exc)

        # Release the cached Deribit client's aiohttp session.
        if self._deribit_client is not None:
            try:
                await self._deribit_client.close()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Deribit client close failed: %s", exc)
            self._deribit_client = None

        self.logger.info("Bot stopped")
        self._notify_state_update()

    def supports_delta_hedge(self) -> bool:
        return self.hedge_manager is not None

    def is_delta_hedge_enabled(self) -> bool:
        if self.hedge_manager is None:
            return False
        return self.hedge_manager.is_enabled()

    async def set_delta_hedge_enabled(self, enabled: bool) -> bool:
        if self.hedge_manager is None:
            return False

        enabled = bool(enabled)
        await self.hedge_manager.set_enabled(enabled)
        self._delta_hedge_enabled = enabled
        if self.options_executor is not None:
            self.options_executor.set_delta_hedge_enabled(enabled)

        if enabled:
            try:
                await self.hedge_manager.reconcile()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Delta hedge enable reconcile failed: %s", exc)

        self._notify_state_update()
        return True

    # ------------------------------------------------------------------
    # Options scheduler callbacks
    # ------------------------------------------------------------------

    async def _hedge_audit_loop(self) -> None:
        """Slow safety poll that reconciles the live options book and hedge."""
        while self.is_running and self.hedge_manager is not None:
            try:
                await self.hedge_manager.reconcile()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Delta hedge audit reconcile failed: %s", exc)
            await asyncio.sleep(self._hedge_reconcile_interval_seconds)

    async def _refresh_options_surface(self) -> None:
        """Background task: rebuild the OptionsContext snapshot.

        Pulls the latest Thalex chain + Deribit data, computes the vol
        surface, runs the regime classifier, and caches the resulting
        snapshot on ``self._latest_options_context`` so the decision task
        can read it on its own cadence. Spot history is fetched from
        Deribit's mark price history endpoint so the realized vol calc has
        real BTC daily closes instead of a placeholder.
        """
        if self.thalex is None:
            return
        try:
            from src.backend.options_intel.builder import build_options_context
            from src.backend.options_intel.deribit_client import DeribitPublicClient
            from src.backend.options_intel.iv_history_store import IVHistoryStore

            # Make sure Thalex is connected and the instrument cache is fresh.
            await self.thalex.connect()

            # Reuse one Deribit client across refreshes — its aiohttp session
            # is lazily built on first call and reused thereafter.
            if self._deribit_client is None:
                self._deribit_client = DeribitPublicClient()
            deribit = self._deribit_client
            try:
                store = IVHistoryStore(db_path="data/iv_history.sqlite")
                spot_history = await self._fetch_btc_daily_closes(deribit, days=16)
                # Pull a longer 30-day series so the Keltner channel (EMA20 +
                # ATR14) has enough data to populate.
                keltner_closes = await self._fetch_btc_daily_closes(deribit, days=30)
                if not keltner_closes or len(set(keltner_closes)) == 1:
                    daily_closes_for_keltner = None
                else:
                    daily_closes_for_keltner = keltner_closes
                # First-hour minute data for the opening-range signal.
                intraday_minutes = await self._fetch_btc_first_hour_minutes(deribit)

                # Recent options trade history from the bot's main DB. Lazily
                # opened in a scoped session so the connection is released
                # before the surface refresh returns. Failures degrade to an
                # empty history list rather than blanking the whole snapshot.
                db_session = None
                db_session_ctx = None
                try:
                    from src.database.db_manager import get_db_manager
                    db_session_ctx = get_db_manager().session_scope()
                    db_session = db_session_ctx.__enter__()
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("options trade-history session unavailable: %s", exc)
                    db_session = None
                    db_session_ctx = None

                try:
                    self._latest_options_context = await build_options_context(
                        thalex=self.thalex,
                        deribit=deribit,
                        iv_history=store,
                        spot_history=spot_history,
                        use_interpolation=True,
                        intraday_minute_prices=intraday_minutes,
                        daily_closes_for_keltner=daily_closes_for_keltner,
                        db_session=db_session,
                    )
                finally:
                    if db_session_ctx is not None:
                        try:
                            db_session_ctx.__exit__(None, None, None)
                        except Exception as exc:  # pylint: disable=broad-except
                            self.logger.debug("db session close failed: %s", exc)

                self.logger.info(
                    "OptionsContext refreshed (regime=%s confidence=%s, spot_history=%d closes, "
                    "intraday=%d minutes, positions=%d, history=%d)",
                    self._latest_options_context.vol_regime,
                    self._latest_options_context.vol_regime_confidence,
                    len(spot_history),
                    len(intraday_minutes),
                    self._latest_options_context.open_position_count,
                    len(self._latest_options_context.recent_options_trades),
                )
            finally:
                # Deribit client is reused across refreshes; closed in stop().
                pass
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("vol surface refresh failed: %s", exc)

    async def _fetch_btc_daily_closes(self, deribit, days: int = 16) -> List[float]:
        """Fetch ``days`` daily BTC mark prices from Deribit for realized-vol calc.

        Falls back to the bot's intraday ``price_history`` deque, then to a
        flat 60k placeholder, when Deribit is unreachable. The placeholder is
        intentional — a flat series gives RV ≈ 0, which the regime classifier
        treats as 'unknown' rather than emitting bad signal.
        """
        try:
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            start_ms = now_ms - (days * 86_400_000)
            closes = await deribit.get_mark_price_history(
                instrument_name="BTC-PERPETUAL",
                start_timestamp_ms=start_ms,
                end_timestamp_ms=now_ms,
                resolution_seconds=86400,
            )
            if closes:
                return closes
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Deribit mark price history fetch failed: %s", exc)

        # Fallbacks: bot's intraday history, then a flat placeholder.
        intraday = list(self.price_history.get("BTC", []))
        if intraday:
            return intraday[-days:]
        return [60000.0] * days

    async def _fetch_btc_first_hour_minutes(self, deribit) -> list:
        """Fetch BTC mark prices at 1-minute resolution for the first hour of today UTC.

        Returns a list of ``(unix_seconds, price)`` tuples — the shape the
        opening-range helper expects. Returns ``[]`` on any failure so the
        snapshot's opening_range stays ``unknown`` rather than crashing.
        """
        try:
            now = datetime.now(UTC)
            today_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
            start_ms = int(today_start.timestamp() * 1000)
            end_ms = start_ms + 60 * 60 * 1000  # +1 hour
            closes = await deribit.get_mark_price_history(
                instrument_name="BTC-PERPETUAL",
                start_timestamp_ms=start_ms,
                end_timestamp_ms=end_ms,
                resolution_seconds=60,
            )
            # The Deribit client only returns the price column; we need
            # timestamps too. Reconstruct an evenly-spaced minute grid
            # starting at today_start to attach timestamps to each close.
            if not closes:
                return []
            start_seconds = int(today_start.timestamp())
            return [(start_seconds + i * 60, float(price)) for i, price in enumerate(closes)]
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Deribit first-hour minute fetch failed: %s", exc)
            return []

    async def _run_options_decision_cycle(self) -> None:
        """Background task: run the options agent against the cached snapshot.

        Reads ``self._latest_options_context`` (populated by the vol surface
        refresh), calls the OptionsAgent, parses decisions, and routes each
        one through ``_execute_thalex_decision``. The scheduler's bootstrap
        sequence guarantees the surface is fetched before the first decision,
        but we still guard against None in case the refresh itself failed.
        """
        if self._latest_options_context is None:
            self.logger.info("OptionsContext not yet available; skipping decision cycle")
            return
        try:
            # Lazy import keeps the LLM client off the import path until needed.
            from src.backend.agent.options_agent import OptionsAgent

            # The real LLM client wrapper isn't built yet — for now we
            # construct an OptionsAgent with the bot's existing TradingAgent
            # as the LLM transport. PR C will swap this for a dedicated
            # async LLM client.
            agent = OptionsAgent(llm=self._options_llm_adapter())
            decisions = await agent.decide(self._latest_options_context)
            self.logger.info("OptionsAgent emitted %d decisions", len(decisions))

            # Surface options reasoning + decisions on the GUI alongside perps.
            # ``getattr`` keeps tests that swap in a minimal fake OptionsAgent
            # (no ``last_payload`` attribute) green.
            options_payload = getattr(agent, "last_payload", None) or {}
            options_decision_payloads: List[Dict[str, Any]] = []

            for decision in decisions:
                decision_payload = {
                    "asset": decision.asset,
                    "action": decision.action,
                    "rationale": decision.rationale,
                    "venue": decision.venue,
                    "strategy": decision.strategy,
                    "underlying": decision.underlying,
                    "kind": decision.kind,
                    "tenor_days": decision.tenor_days,
                    "target_strike": decision.target_strike,
                    "target_delta": decision.target_delta,
                    "contracts": decision.contracts,
                    "legs": [
                        {
                            "kind": leg.kind,
                            "side": leg.side,
                            "contracts": leg.contracts,
                            "target_strike": leg.target_strike,
                            "target_delta": leg.target_delta,
                            "tenor_days": leg.tenor_days,
                        }
                        for leg in decision.legs
                    ],
                    "entry_kind": decision.entry_kind,
                    "vol_view": decision.vol_view,
                    "target_gamma_btc": decision.target_gamma_btc,
                }
                options_decision_payloads.append(decision_payload)
                # Hold decisions surface on the reasoning page but don't
                # create proposals — nothing to execute.
                if decision.action == "hold":
                    continue
                if self.trading_mode == "manual":
                    try:
                        proposal = self._create_thalex_proposal(decision_payload)
                        self.pending_proposals.append(proposal)
                        self.logger.info(
                            "[PROPOSAL] Created: %s %s %s (ID: %s)",
                            (decision.strategy or "thalex").upper(),
                            decision.action.upper(),
                            decision.asset,
                            proposal.id[:8],
                        )
                        self._sync_pending_proposals_state()
                    except Exception as exc:  # pylint: disable=broad-except
                        self.logger.error(
                            "Error creating scheduled Thalex proposal for %s: %s",
                            decision.asset,
                            exc,
                        )
                    continue
                await self._execute_thalex_decision(decision_payload)

            # Always publish — even an empty trade_decisions array carries the
            # LLM's reasoning text, which the operator wants to see on the
            # Reasoning page.
            self.state.last_options_reasoning = {
                "reasoning": (options_payload.get("reasoning") if isinstance(options_payload, dict) else "") or "",
                "trade_decisions": options_decision_payloads,
                "cycle_at": datetime.now(UTC).isoformat(),
            }
            self._compose_last_reasoning()
            self._notify_state_update()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("options decision cycle failed: %s", exc)

    def _options_llm_adapter(self):
        """Return the cached LLM client (or shim) used by OptionsAgent.

        The lifecycle wrapper owns a single client instance for the bot's
        whole lifetime so the underlying aiohttp session is reused across
        every 3-hour decision cycle. The session is released by
        :py:meth:`stop` via ``options_llm_lifecycle.close()``.

        When the lifecycle wasn't constructed (Thalex not configured) we
        fall back to a one-shot shim instance so callers always get a
        usable ``chat_json`` interface.
        """
        if self.options_llm_lifecycle is not None:
            return self.options_llm_lifecycle.get()

        # Defensive fallback: if the lifecycle wasn't built (e.g. Thalex
        # disabled but the scheduler somehow ran), return an in-place shim.
        # This branch should not normally be reached.
        from src.backend.agent.options_llm_lifecycle import _ShimLLM

        return _ShimLLM(self.logger)

    @staticmethod
    def _position_field(position: Any, *names: str) -> Any:
        if isinstance(position, dict):
            for name in names:
                if name in position:
                    return position.get(name)
            return None
        for name in names:
            if hasattr(position, name):
                return getattr(position, name)
        return None

    def _position_opened_by(self, venue: str, asset: str, instrument_name: str) -> str:
        venue_name = (venue or "hyperliquid").lower()
        for trade in self.active_trades:
            trade_venue = (trade.get("venue") or "hyperliquid").lower()
            if trade_venue != venue_name:
                continue
            if venue_name == "thalex":
                tracked_instruments = {
                    str(name)
                    for name in (trade.get("instrument_names") or [])
                    if name
                }
                legacy_name = trade.get("instrument_name")
                if legacy_name:
                    tracked_instruments.add(str(legacy_name))
                if instrument_name and instrument_name in tracked_instruments:
                    return "AI"
                continue
            if asset and trade.get("asset") == asset:
                return "AI"
        return "External"

    def _build_positions_view(
        self,
        hyperliquid_positions: List[Dict],
        thalex_positions: Optional[List[Any]] = None,
    ) -> List[Dict]:
        rows: List[Dict] = []

        for pos in hyperliquid_positions:
            asset = str(pos.get("symbol") or pos.get("coin") or "")
            instrument_name = asset
            rows.append({
                "row_id": f"hyperliquid:{instrument_name}",
                "symbol": asset,
                "asset": asset,
                "instrument_name": instrument_name,
                "venue": "hyperliquid",
                "quantity": float(pos.get("quantity", 0) or 0),
                "entry_price": float(pos.get("entry_price", 0) or 0),
                "current_price": float(pos.get("current_price", 0) or 0),
                "liquidation_price": float(pos.get("liquidation_price", 0) or 0),
                "unrealized_pnl": float(pos.get("unrealized_pnl", 0) or 0),
                "leverage": pos.get("leverage", 1) or 1,
                "opened_by": self._position_opened_by("hyperliquid", asset, instrument_name),
                "closable": True,
            })

        for pos in thalex_positions or []:
            asset = str(self._position_field(pos, "asset", "symbol") or "")
            instrument_name = str(self._position_field(pos, "instrument_name") or asset)
            side = str(self._position_field(pos, "side") or "long").lower()
            size = abs(float(self._position_field(pos, "size", "quantity") or 0) or 0)
            quantity = size if side == "long" else -size
            rows.append({
                "row_id": f"thalex:{instrument_name}",
                "symbol": instrument_name,
                "asset": asset,
                "instrument_name": instrument_name,
                "venue": "thalex",
                "quantity": quantity,
                "entry_price": float(self._position_field(pos, "entry_price") or 0),
                "current_price": float(self._position_field(pos, "current_price") or 0),
                "liquidation_price": 0.0,
                "unrealized_pnl": float(self._position_field(pos, "unrealized_pnl") or 0),
                "leverage": 1,
                "opened_by": self._position_opened_by("thalex", asset, instrument_name),
                "closable": False,
            })

        return rows

    async def _live_thalex_open_positions_count(self) -> int:
        if self.thalex is None:
            return 0
        try:
            state = await self.thalex.get_user_state()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to load live Thalex position count: %s", exc)
            return 0

        positions = getattr(state, "positions", None)
        if positions is None and isinstance(state, dict):
            positions = state.get("positions")
        return len(list(positions or []))

    async def _resolve_thalex_execution_price(self, instrument_names: List[str]) -> float:
        if self.thalex is None or not instrument_names:
            return 0.0

        tracked = set(instrument_names)
        try:
            fills = await self.thalex.get_recent_fills(limit=20)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to load Thalex fills for execution price: %s", exc)
            fills = []

        for fill in reversed(fills or []):
            if not isinstance(fill, dict):
                continue
            fill_instrument = str(fill.get("instrument_name") or fill.get("asset") or "")
            if fill_instrument not in tracked:
                continue
            for key in ("price", "px", "avg_price", "average_price"):
                value = fill.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        try:
            return float(await self.thalex.get_current_price(instrument_names[0]) or 0.0)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to load Thalex mark price fallback: %s", exc)
            return 0.0

    def _account_snapshot(self, hyperliquid_state: Dict, thalex_state: Optional[Any] = None) -> Dict[str, Any]:
        hl_balance = float(hyperliquid_state.get('balance', 0.0) or 0.0)
        hl_total_value = float(hyperliquid_state.get('total_value', hl_balance) or hl_balance)
        balance_breakdown = {'hyperliquid': hl_balance}
        total_value_breakdown = {'hyperliquid': hl_total_value}

        if thalex_state is not None:
            if isinstance(thalex_state, dict):
                thalex_balance = float(thalex_state.get('balance', 0.0) or 0.0)
                thalex_total_value = float(thalex_state.get('total_value', thalex_balance) or thalex_balance)
            else:
                thalex_balance = float(getattr(thalex_state, 'balance', 0.0) or 0.0)
                thalex_total_value = float(getattr(thalex_state, 'total_value', thalex_balance) or thalex_balance)
            balance_breakdown['thalex'] = thalex_balance
            total_value_breakdown['thalex'] = thalex_total_value

        return {
            'balance': sum(balance_breakdown.values()),
            'total_value': sum(total_value_breakdown.values()),
            'balance_breakdown': balance_breakdown,
            'total_value_breakdown': total_value_breakdown,
        }

    async def _fetch_indicators_hl_first(self, asset: str, interval: str, current_price: float) -> dict:
        """Fetch indicators using Hyperliquid candles (primary) with TAAPI fallback.

        Pulls OHLCV candles from Hyperliquid's ``candles_snapshot`` endpoint and
        computes SMA99, Keltner(130,130,4), anchored VWAP, and opening range
        locally.  Falls back to the TAAPI bulk API only when Hyperliquid candle
        fetch fails.
        """
        from datetime import timezone

        AVWAP_ANCHOR_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1000

        try:
            now = datetime.now(timezone.utc)
            now_ms = int(now.timestamp() * 1000)

            # We need enough 5m candles for Keltner(130,130,4) + SMA(99):
            # ~131 candles × 5min = ~11 hours of 5m data
            candles_needed_5m = 200  # generous buffer
            start_5m_ms = now_ms - candles_needed_5m * 5 * 60 * 1000

            # Daily candles from the AVWAP anchor (2026-01-01)
            start_daily_ms = AVWAP_ANCHOR_MS

            # Long-term candles: need ~131 bars at the configured interval
            interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            lt_bar_mins = interval_minutes.get(interval, 240)
            start_long_ms = now_ms - 200 * lt_bar_mins * 60 * 1000

            # Fetch all three candle sets in parallel from Hyperliquid
            candles_5m, candles_daily, candles_long = await asyncio.gather(
                self.hyperliquid.get_candles(asset, "5m", start_5m_ms, now_ms),
                self.hyperliquid.get_candles(asset, "1d", start_daily_ms, now_ms),
                self.hyperliquid.get_candles(asset, interval, start_long_ms, now_ms),
            )

            if not candles_5m:
                raise ValueError(f"Hyperliquid returned no 5m candles for {asset}")

            self.logger.info(
                "Hyperliquid candles for %s: 5m=%d, 1d=%d, %s=%d",
                asset, len(candles_5m), len(candles_daily), interval, len(candles_long),
            )

            indicators = build_indicator_bundle(
                candles_5m=candles_5m,
                candles_daily=candles_daily,
                candles_long=candles_long,
                long_interval=interval,
                current_spot=current_price,
            )
            return indicators

        except Exception as e:
            self.logger.warning(
                "Hyperliquid candle fetch failed for %s, falling back to TAAPI: %s", asset, e,
            )
            # Fall back to TAAPI (sync, run in thread)
            loop = asyncio.get_running_loop()

            def _pause_for_taapi_rate_limit() -> None:
                future = asyncio.run_coroutine_threadsafe(asyncio.sleep(15), loop)
                future.result()

            indicators = await asyncio.to_thread(
                self.taapi.fetch_asset_indicators,
                asset,
                current_spot=current_price,
                request_pause=_pause_for_taapi_rate_limit,
            )
            return indicators

    _MAX_CONSECUTIVE_ERRORS = 5

    async def _main_loop(self):
        """
        Main trading loop.
        Adapted from ai-trading-agent/src/main.py lines 88-455
        """
        consecutive_errors = 0
        try:
            while self.is_running:
                self.invocation_count += 1
                self.state.invocation_count = self.invocation_count

                try:
                    # ===== PHASE 1: Fetch Account State =====
                    state = await self.hyperliquid.get_user_state()
                    thalex_state = None
                    if self.thalex is not None:
                        try:
                            thalex_state = await self.thalex.get_user_state()
                        except Exception as exc:  # pylint: disable=broad-except
                            self.logger.warning("Failed to load Thalex portfolio state: %s", exc)

                    account_snapshot = self._account_snapshot(state, thalex_state)
                    balance = account_snapshot['balance']
                    total_value = account_snapshot['total_value']

                    # Calculate total return from the actual session baseline.
                    initial_balance = float(self.initial_account_value or total_value or 0.0)
                    total_return_pct = 0.0
                    if initial_balance > 0:
                        total_return_pct = ((total_value - initial_balance) / initial_balance) * 100

                    if self._previous_total_value is not None and self._previous_total_value > 0:
                        period_return = (total_value - self._previous_total_value) / self._previous_total_value
                        self.trade_log.append(period_return)
                    self._previous_total_value = total_value

                    sharpe_ratio = self._calculate_sharpe(self.trade_log)
                    
                    self.logger.debug(f"  Balance: ${balance:,.2f} | Return: {total_return_pct:+.2f}% | Sharpe: {sharpe_ratio:.2f}")

                    # Update bot state
                    self.state.balance = balance
                    self.state.total_value = total_value
                    self.state.balance_breakdown = account_snapshot['balance_breakdown']
                    self.state.total_value_breakdown = account_snapshot['total_value_breakdown']
                    self.state.total_return_pct = total_return_pct
                    self.state.sharpe_ratio = sharpe_ratio

                    # ===== PHASE 2: Enrich Hyperliquid Positions =====
                    enriched_hyperliquid_positions = []
                    for pos in state['positions']:
                        symbol = pos.get('coin')
                        try:
                            current_price = await self.hyperliquid.get_current_price(symbol)
                            enriched_hyperliquid_positions.append({
                                'symbol': symbol,
                                'quantity': float(pos.get('szi', 0) or 0),
                                'entry_price': float(pos.get('entryPx', 0) or 0),
                                'current_price': current_price,
                                'liquidation_price': float(pos.get('liquidationPx', 0) or 0),
                                'unrealized_pnl': pos.get('unrealized_pnl') or pos.get('pnl', 0.0),
                                'leverage': pos.get('leverage', {}).get('value', 1) if isinstance(pos.get('leverage'), dict) else pos.get('leverage', 1)
                            })
                        except Exception as e:
                            self.logger.error(f"Error enriching position for {symbol}: {e}")

                    # ===== PHASE 3: Load Recent Diary =====
                    recent_diary = self._load_recent_diary(limit=10)

                    # ===== PHASE 4: Fetch Open Orders =====
                    open_orders_raw = await self.hyperliquid.get_open_orders()
                    open_orders = []
                    for o in open_orders_raw:
                        order_type_obj = o.get('orderType', {})
                        trigger_price = None
                        order_type_str = 'limit'

                        if isinstance(order_type_obj, dict) and 'trigger' in order_type_obj:
                            order_type_str = 'trigger'
                            trigger_data = order_type_obj.get('trigger', {})
                            if 'triggerPx' in trigger_data:
                                trigger_price = float(trigger_data['triggerPx'])

                        open_orders.append({
                            'coin': o.get('coin'),
                            'oid': o.get('oid'),
                            'is_buy': o.get('side') == 'B',
                            'size': float(o.get('sz', 0)),
                            'price': float(o.get('limitPx', 0)),
                            'trigger_price': trigger_price,
                            'order_type': order_type_str
                        })

                    self.state.open_orders = open_orders

                    thalex_positions = []
                    if thalex_state is not None:
                        thalex_positions = list(getattr(thalex_state, 'positions', []) or [])

                    # ===== PHASE 5: Reconcile Active Trades =====
                    await self._reconcile_active_trades(
                        state['positions'],
                        open_orders_raw,
                        thalex_positions,
                    )

                    combined_positions = self._build_positions_view(
                        enriched_hyperliquid_positions,
                        thalex_positions,
                    )
                    self.state.positions = combined_positions
                    self.state.active_trades = list(self.active_trades)

                    # ===== PHASE 6: Fetch Recent Fills =====
                    fills_raw = await self.hyperliquid.get_recent_fills(limit=50)
                    recent_fills = []
                    for fill in fills_raw[-20:]:
                        ts = fill.get('time')
                        if ts and ts > 1_000_000_000_000:
                            ts = ts / 1000
                        ts_str = datetime.fromtimestamp(ts, UTC).isoformat() if ts else ""

                        recent_fills.append({
                            'timestamp': ts_str,
                            'coin': fill.get('coin'),
                            'is_buy': fill.get('side') == 'B',
                            'size': float(fill.get('sz', 0)),
                            'price': float(fill.get('px', 0))
                        })

                    self.state.recent_fills = recent_fills

                    # ===== PHASE 7: Build Dashboard =====
                    # Buying-power snapshot — the LLM uses ``max_new_notional_by_asset``
                    # as the hard cap on ``allocation_usd`` so it won't propose trades
                    # the account can't afford. Without this, the only margin check was
                    # at execution time, which produced silent "Insufficient margin"
                    # failures with no feedback to the agent.
                    buying_power = await self._get_buying_power_snapshot()

                    dashboard = {
                        'total_return_pct': total_return_pct,
                        'balance': balance,
                        'account_value': total_value,
                        'buying_power': buying_power,
                        'sharpe_ratio': sharpe_ratio,
                        'positions': [
                            {k: v for k, v in pos.items() if k not in ("closable", "row_id")}
                            for pos in combined_positions
                        ],
                        'active_trades': self.active_trades,
                        'open_orders': open_orders,
                        'recent_diary': recent_diary,
                        'recent_fills': recent_fills
                    }

                    # ===== PHASE 8: Gather Market Data =====
                    market_sections = []
                    for idx, asset in enumerate(self.assets):
                        try:
                            # Current price
                            current_price = await self.hyperliquid.get_current_price(asset)

                            # Store price history
                            self.price_history[asset].append({
                                't': datetime.now(UTC).isoformat(),
                                'mid': current_price
                            })

                            # Open interest and funding
                            oi = await self.hyperliquid.get_open_interest(asset)
                            funding = await self.hyperliquid.get_funding_rate(asset)
                            prev_day_price = await self.hyperliquid.get_prev_day_price(asset)
                            volume_24h = await self.hyperliquid.get_daily_notional_volume(asset)

                            # --- Hyperliquid-first indicator fetch ---
                            interval = CONFIG.get("interval", "4h")
                            indicators = await self._fetch_indicators_hl_first(
                                asset, interval, current_price,
                            )
                            
                            # Extract 5m indicators
                            sma99_5m_series = indicators["5m"].get("sma99", [])
                            avwap_5m = indicators["5m"].get("avwap")
                            keltner_5m = indicators["5m"].get("keltner", {})
                            opening_range = indicators["5m"].get("opening_range", {})

                            # Extract long-term indicators (interval from config: 1h, 4h, etc.)
                            lt_indicators = indicators.get(interval, {})
                            lt_sma99_series = lt_indicators.get("sma99", [])
                            lt_avwap = lt_indicators.get("avwap")
                            lt_keltner = lt_indicators.get("keltner", {})

                            keltner_5m_middle = keltner_5m.get("middle", [])
                            keltner_5m_upper = keltner_5m.get("upper", [])
                            keltner_5m_lower = keltner_5m.get("lower", [])
                            lt_keltner_middle = lt_keltner.get("middle", [])
                            lt_keltner_upper = lt_keltner.get("upper", [])
                            lt_keltner_lower = lt_keltner.get("lower", [])
                            recent_price_points = list(self.price_history[asset])[-10:]
                            recent_timestamps = [p["t"] for p in recent_price_points]

                            price_candles_5m = indicators["5m"].get("price_candles", {}) or {}
                            price_candles_lt = lt_indicators.get("price_candles", {}) or {}
                            # Prefer candle close timestamps so indicator & price charts share an x-axis
                            candle_times_5m = price_candles_5m.get("time") or []
                            candle_times_lt = price_candles_lt.get("time") or []
                            keltner_5m_timestamps = (
                                candle_times_5m[-len(keltner_5m_middle):]
                                if keltner_5m_middle and len(candle_times_5m) >= len(keltner_5m_middle)
                                else (recent_timestamps[-len(keltner_5m_middle):] if keltner_5m_middle else [])
                            )
                            lt_keltner_timestamps = (
                                candle_times_lt[-len(lt_keltner_middle):]
                                if lt_keltner_middle and len(candle_times_lt) >= len(lt_keltner_middle)
                                else (recent_timestamps[-len(lt_keltner_middle):] if lt_keltner_middle else [])
                            )

                            def _latest(series):
                                return series[-1] if series else None

                            def _keltner_snapshot(middle_series, upper_series, lower_series):
                                middle = _latest(middle_series)
                                upper = _latest(upper_series)
                                lower = _latest(lower_series)
                                position = "unknown"
                                if all(v is not None for v in (upper, lower, current_price)):
                                    if current_price > upper:
                                        position = "above"
                                    elif current_price < lower:
                                        position = "below"
                                    else:
                                        position = "inside"
                                return {
                                    "middle": middle,
                                    "upper": upper,
                                    "lower": lower,
                                    "position": position,
                                }

                            # Build market data structure
                            market_sections.append({
                                "asset": asset,
                                "current_price": current_price,
                                "intraday": {
                                    "sma99": _latest(sma99_5m_series),
                                    "avwap": avwap_5m,
                                    "keltner": _keltner_snapshot(keltner_5m_middle, keltner_5m_upper, keltner_5m_lower),
                                    "opening_range": opening_range,
                                    "series": {
                                        "sma99": sma99_5m_series,
                                        "keltner_middle": keltner_5m_middle,
                                        "keltner_upper": keltner_5m_upper,
                                        "keltner_lower": keltner_5m_lower,
                                        "timestamps": keltner_5m_timestamps,
                                        "price_candles": price_candles_5m,
                                    }
                                },
                                "long_term": {
                                    "interval": interval,
                                    "sma99": _latest(lt_sma99_series),
                                    "avwap": lt_avwap,
                                    "keltner": _keltner_snapshot(lt_keltner_middle, lt_keltner_upper, lt_keltner_lower),
                                    "series": {
                                        "sma99": lt_sma99_series,
                                        "keltner_middle": lt_keltner_middle,
                                        "keltner_upper": lt_keltner_upper,
                                        "keltner_lower": lt_keltner_lower,
                                        "timestamps": lt_keltner_timestamps,
                                        "price_candles": price_candles_lt,
                                    },
                                },
                                "open_interest": oi,
                                "prev_day_price": prev_day_price,
                                "volume_24h": volume_24h,
                                "funding_rate": funding,
                                "funding_annualized_pct": funding * 24 * 365 * 100 if funding else None,
                                "recent_mid_prices": [p['mid'] for p in recent_price_points],
                                "recent_timestamps": recent_timestamps,
                            })

                        except Exception as e:
                            self.logger.error(f"Error gathering market data for {asset}: {e}")

                    # ===== PHASE 9: Build LLM Context =====
                    context_payload = OrderedDict([
                        ("invocation", {
                            "count": self.invocation_count,
                            "current_time": datetime.now(UTC).isoformat()
                        }),
                        ("account", dashboard),
                        ("market_data", market_sections),
                        ("instructions", {
                            "assets": self.assets,
                            "note": "Follow the system prompt guidelines strictly"
                        })
                    ])

                    # Inject compact options book summary when available so the
                    # perps agent can factor in the options desk's net exposure.
                    if self._latest_options_context is not None:
                        ctx = self._latest_options_context
                        context_payload["options_book_summary"] = {
                            "vol_regime": ctx.vol_regime,
                            "portfolio_greeks": ctx.portfolio_greeks,
                            "open_position_count": ctx.open_position_count,
                        }

                    context = json.dumps(context_payload, default=json_default, indent=2)

                    # Log prompt
                    with open("data/prompts.log", "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Invocation {self.invocation_count} - {datetime.now(UTC).isoformat()}\n")
                        f.write(f"{'='*80}\n")
                        f.write(context + "\n")

                    # ===== PHASE 10: Get LLM Decision =====
                    decisions = await asyncio.to_thread(
                        self.agent.decide_trade, self.assets, context
                    )

                    # Validate and retry if needed
                    if not isinstance(decisions, dict) or 'trade_decisions' not in decisions:
                        self.logger.warning("Invalid decision format, retrying with strict prefix...")
                        strict_context = (
                            "Return ONLY the JSON object per the schema. "
                            "No markdown, no explanation.\n\n" + context
                        )
                        decisions = await asyncio.to_thread(
                            self.agent.decide_trade, self.assets, strict_context
                        )

                    # Check for all-hold with parse errors
                    trade_decisions = decisions.get('trade_decisions', [])
                    if all(
                        d.get('action') == 'hold' and 'parse error' in d.get('rationale', '').lower()
                        for d in trade_decisions
                    ):
                        self.logger.warning("All holds with parse errors, retrying...")
                        decisions = await asyncio.to_thread(
                            self.agent.decide_trade, self.assets, context
                        )
                        trade_decisions = decisions.get('trade_decisions', [])

                    # Extract reasoning
                    reasoning = decisions.get('reasoning', '')
                    if reasoning:
                        self.logger.info(f"LLM Reasoning: {reasoning[:200]}...")

                    self.state.last_perps_reasoning = decisions
                    self._compose_last_reasoning()
                    self._notify_state_update()

                    # ===== PHASE 11: Execute Trades or Create Proposals =====
                    for decision in trade_decisions:
                        asset = decision.get('asset')
                        if asset not in self.assets:
                            continue

                        # Multi-venue routing: Thalex options decisions are
                        # handled by the OptionsExecutor; everything else falls
                        # through to the existing Hyperliquid path.
                        if (decision.get('venue') or 'hyperliquid').lower() == 'thalex':
                            if self.trading_mode == "manual":
                                try:
                                    proposal = self._create_thalex_proposal(decision)
                                    self.pending_proposals.append(proposal)
                                    self.logger.info(
                                        "[PROPOSAL] Created: %s %s %s (ID: %s)",
                                        (decision.get("strategy") or "thalex").upper(),
                                        decision.get("action", "hold").upper(),
                                        asset,
                                        proposal.id[:8],
                                    )
                                    self._sync_pending_proposals_state()
                                except Exception as exc:  # pylint: disable=broad-except
                                    self.logger.error("Error creating Thalex proposal for %s: %s", asset, exc)
                                continue
                            ok, reason = await self._execute_thalex_decision(decision)
                            if not ok:
                                self._handle_execution_failure("Thalex", asset, reason)
                            continue

                        action = decision.get('action')
                        rationale = decision.get('rationale', '')
                        allocation = float(decision.get('allocation_usd') or 0)
                        tp_price = decision.get('tp_price')
                        sl_price = decision.get('sl_price')
                        exit_plan = decision.get('exit_plan', '')
                        confidence = decision.get('confidence', 75.0)

                        if action in ['buy', 'sell']:
                            # MANUAL MODE: Create proposal instead of executing
                            if self.trading_mode == "manual":
                                try:
                                    current_price = await self.hyperliquid.get_current_price(asset)
                                    if not current_price or current_price <= 0:
                                        self.logger.error(f"Skipping proposal for {asset}: invalid price {current_price}")
                                        continue
                                    size = allocation / current_price

                                    # Pre-proposal margin guard — refuse to create a
                                    # proposal the account can't afford. Without this,
                                    # the LLM's unaffordable trade would sit in the
                                    # Recommendations UI until the user clicked Execute
                                    # and hit the same error at execution time.
                                    try:
                                        await self._hl_margin_preflight(asset, allocation)
                                    except RuntimeError as margin_err:
                                        self.logger.warning(
                                            f"[SKIP-PROPOSAL] {action.upper()} {asset}: {margin_err}"
                                        )
                                        self._write_diary_entry({
                                            'timestamp': datetime.now(UTC).isoformat(),
                                            'asset': asset,
                                            'action': 'proposal_skipped_insufficient_margin',
                                            'proposed_action': action,
                                            'proposed_allocation_usd': allocation,
                                            'reason': str(margin_err),
                                            'rationale': rationale,
                                        })
                                        continue

                                    # Calculate risk/reward
                                    risk_reward = None
                                    if tp_price and sl_price and current_price:
                                        potential_gain = abs(tp_price - current_price) / current_price
                                        potential_loss = abs(sl_price - current_price) / current_price
                                        if potential_loss > 0:
                                            risk_reward = potential_gain / potential_loss

                                    proposal = TradeProposal(
                                        asset=asset,
                                        action=action,
                                        confidence=confidence,
                                        risk_reward=risk_reward,
                                        entry_price=current_price,
                                        tp_price=tp_price,
                                        sl_price=sl_price,
                                        size=size,
                                        allocation=allocation,
                                        rationale=rationale,
                                        market_conditions={
                                            'current_price': current_price,
                                            'exit_plan': exit_plan
                                        }
                                    )
                                    
                                    self.pending_proposals.append(proposal)
                                    self.logger.info(f"[PROPOSAL] Created: {action.upper()} {asset} @ ${current_price:,.2f} (ID: {proposal.id[:8]})")
                                    
                                    # Update state with proposals
                                    self._sync_pending_proposals_state()

                                except Exception as e:
                                    self.logger.error(f"Error creating proposal for {asset}: {e}")
                                    
                                continue  # Skip execution in manual mode
                            
                            # AUTO MODE: Execute immediately (position-aware)
                            try:
                                current_price = await self.hyperliquid.get_current_price(asset)
                                if not current_price or current_price <= 0:
                                    self.logger.error(f"Skipping {action} {asset}: invalid price {current_price}")
                                    continue
                                desired_size = allocation / current_price

                                # --- Cancel stale orders for this asset ---
                                cancel_result = await self.hyperliquid.cancel_all_orders(asset)
                                if cancel_result.get('cancelled_count', 0) > 0:
                                    self.logger.info(f"Cancelled {cancel_result['cancelled_count']} stale order(s) for {asset}")

                                # --- Position-aware sizing ---
                                # Look up existing position so we account for
                                # closing it before opening the new direction.
                                existing_pos = 0.0
                                for pos in self.state.positions:
                                    if pos.get('symbol') == asset:
                                        existing_pos = float(pos.get('quantity', 0) or 0)
                                        break

                                # Determine required order size.
                                # BUY with existing short (-0.165): must buy
                                #   |short| to flatten + desired_size for
                                #   the new long.
                                # BUY with existing long: just add desired_size.
                                # (mirror logic for SELL)
                                if action == 'buy' and existing_pos < 0:
                                    # Close the short first, then go long with remainder
                                    amount = abs(existing_pos) + desired_size
                                    self.logger.info(
                                        f"{asset}: closing short ({existing_pos:.6f}) + new long ({desired_size:.6f}) = total buy {amount:.6f}"
                                    )
                                elif action == 'sell' and existing_pos > 0:
                                    # Close the long first, then go short with remainder
                                    amount = existing_pos + desired_size
                                    self.logger.info(
                                        f"{asset}: closing long ({existing_pos:.6f}) + new short ({desired_size:.6f}) = total sell {amount:.6f}"
                                    )
                                else:
                                    amount = desired_size

                                # The resulting position size is the new
                                # directional exposure (after closing the old
                                # position).  TP/SL should protect only this.
                                net_new_amount = desired_size
                                order_result = None
                                filled = False

                                if amount > 0:
                                    # Pre-trade margin check — fails loud if
                                    # the venue can't cover this notional so
                                    # we never record a phantom fill.
                                    await self._hl_margin_preflight(asset, amount * current_price)

                                    # Place market order
                                    if action == 'buy':
                                        order_result = await self.hyperliquid.place_buy_order(asset, amount)
                                    else:
                                        order_result = await self.hyperliquid.place_sell_order(asset, amount)

                                    # Validate the SDK response BEFORE we
                                    # record anything. Insufficient-margin
                                    # rejections come back with top-level
                                    # status="ok" but a per-status error —
                                    # so skipping this check silently books
                                    # a trade the exchange never accepted.
                                    self._hl_validate_response(
                                        order_result,
                                        context=f"{action} {asset} x{amount:.6f}",
                                    )

                                    self.logger.info(f"Executed {action} {asset}: {amount:.6f} @ {current_price}")

                                    # Wait and check fills
                                    await asyncio.sleep(1)
                                    recent_fills_check = await self.hyperliquid.get_recent_fills(limit=5)
                                    filled = any(
                                        f.get('coin') == asset and
                                        abs(float(f.get('sz', 0)) - amount) < 0.0001
                                        for f in recent_fills_check
                                    )

                                # Place TP/SL orders on the NEW position only
                                tp_oid = None
                                sl_oid = None

                                if tp_price and net_new_amount > 0:
                                    try:
                                        is_buy = (action == 'buy')
                                        tp_order = await self.hyperliquid.place_take_profit(
                                            asset, is_buy, net_new_amount, tp_price
                                        )
                                        oids = self.hyperliquid.extract_oids(tp_order)
                                        tp_oid = oids[0] if oids else None
                                        self.logger.info(f"Placed TP order for {asset} @ {tp_price}")
                                    except Exception as e:
                                        self.logger.error(f"Failed to place TP: {e}")

                                if sl_price and net_new_amount > 0:
                                    try:
                                        is_buy = (action == 'buy')
                                        sl_order = await self.hyperliquid.place_stop_loss(
                                            asset, is_buy, net_new_amount, sl_price
                                        )
                                        oids = self.hyperliquid.extract_oids(sl_order)
                                        sl_oid = oids[0] if oids else None
                                        self.logger.info(f"Placed SL order for {asset} @ {sl_price}")
                                    except Exception as e:
                                        self.logger.error(f"Failed to place SL: {e}")

                                # Update active trades
                                self.active_trades = [
                                    t for t in self.active_trades if t['asset'] != asset
                                ]
                                if net_new_amount > 0:
                                    self.active_trades.append({
                                        'venue': 'hyperliquid',
                                        'asset': asset,
                                        'instrument_name': asset,
                                        'instrument_names': [asset],
                                        'is_long': (action == 'buy'),
                                        'amount': net_new_amount,
                                        'entry_price': current_price,
                                        'tp_oid': tp_oid,
                                        'sl_oid': sl_oid,
                                        'exit_plan': exit_plan,
                                        # Deep-memory fields: preserve the full thesis so the
                                        # LLM sees why this trade was opened on the next cycle
                                        # (even after a bot restart). Without these, the agent
                                        # has to re-derive the thesis from only the exit_plan.
                                        'rationale': rationale,
                                        'tp_price': tp_price,
                                        'sl_price': sl_price,
                                        'confidence': confidence,
                                        'opened_at': datetime.now(UTC).isoformat()
                                    })
                                self._save_active_trades()

                                # Write to diary
                                self._write_diary_entry({
                                    'timestamp': datetime.now(UTC).isoformat(),
                                    'asset': asset,
                                    'action': action,
                                    'allocation_usd': allocation,
                                    'amount': amount,
                                    'existing_position': existing_pos,
                                    'net_new_amount': net_new_amount,
                                    'entry_price': current_price,
                                    'tp_price': tp_price,
                                    'tp_oid': tp_oid,
                                    'sl_price': sl_price,
                                    'sl_oid': sl_oid,
                                    'exit_plan': exit_plan,
                                    'rationale': rationale,
                                    'order_result': str(order_result) if order_result else 'no-order',
                                    'opened_at': datetime.now(UTC).isoformat(),
                                    'filled': filled
                                })

                                # Notify GUI of trade
                                if self.on_trade_executed:
                                    self.on_trade_executed({
                                        'asset': asset,
                                        'action': action,
                                        'amount': amount,
                                        'price': current_price,
                                        'timestamp': datetime.now(UTC).isoformat()
                                    })

                            except Exception as e:
                                self.logger.error(f"Error executing {action} for {asset}: {e}")
                                if self.on_error:
                                    self.on_error(f"Trade execution error: {e}")
                                # Persist the skip/failure so the next cycle's
                                # ``recent_diary`` tells the LLM the trade didn't happen.
                                # Margin-preflight raises RuntimeError with a distinctive
                                # prefix so we tag those separately from other failures.
                                err_text = str(e)
                                diary_action = (
                                    'execution_skipped_insufficient_margin'
                                    if err_text.startswith("Insufficient Hyperliquid margin")
                                    else 'execution_failed'
                                )
                                self._write_diary_entry({
                                    'timestamp': datetime.now(UTC).isoformat(),
                                    'asset': asset,
                                    'action': diary_action,
                                    'proposed_action': action,
                                    'proposed_allocation_usd': allocation,
                                    'reason': err_text,
                                    'rationale': rationale,
                                })

                        elif action == 'hold':
                            self.logger.info(f"{asset}: HOLD - {rationale}")
                            self._write_diary_entry({
                                'timestamp': datetime.now(UTC).isoformat(),
                                'asset': asset,
                                'action': 'hold',
                                'rationale': rationale
                            })

                    # Update market data in state for dashboard
                    self.state.market_data = market_sections

                    # Update state timestamp
                    self.state.last_update = datetime.now(UTC).isoformat()
                    self._notify_state_update()
                    consecutive_errors = 0  # successful iteration

                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(
                        "Error in main loop iteration (%d/%d consecutive): %s",
                        consecutive_errors, self._MAX_CONSECUTIVE_ERRORS, e,
                        exc_info=True,
                    )
                    self.state.error = str(e)
                    self.state.last_update = datetime.now(UTC).isoformat()
                    self._notify_state_update()
                    if self.on_error:
                        self.on_error(str(e))
                    if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(
                            "Stopping bot after %d consecutive failures", consecutive_errors,
                        )
                        self.is_running = False
                        self.state.is_running = False
                        self.state.error = (
                            f"Bot stopped: {consecutive_errors} consecutive failures — last: {e}"
                        )
                        self._notify_state_update()
                        break

                # ===== PHASE 12: Sleep Until Next Interval =====
                backoff = min(consecutive_errors, 3) * 30  # 0s, 30s, 60s, 90s
                await asyncio.sleep(self._get_interval_seconds() + backoff)

        except asyncio.CancelledError:
            self.logger.info("Bot loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
            self.state.error = str(e)
            if self.on_error:
                self.on_error(str(e))

    async def _reconcile_active_trades(
        self,
        positions: List[Dict],
        open_orders: List[Dict],
        thalex_positions: Optional[List[Any]] = None,
    ):
        """
        Reconcile local active_trades with exchange state.
        Remove stale entries that no longer exist on exchange.
        """
        exchange_assets = {pos.get('coin') for pos in positions}
        order_assets = {o.get('coin') for o in open_orders}
        tracked_assets = exchange_assets | order_assets
        thalex_instruments = {
            str(self._position_field(pos, 'instrument_name') or '')
            for pos in (thalex_positions or [])
            if self._position_field(pos, 'instrument_name')
        }
        thalex_assets = {
            str(self._position_field(pos, 'asset') or '')
            for pos in (thalex_positions or [])
            if self._position_field(pos, 'asset')
        }

        removed = []
        for trade in self.active_trades[:]:
            venue = (trade.get('venue') or 'hyperliquid').lower()
            if venue == 'thalex':
                tracked_instruments = {
                    str(name)
                    for name in (trade.get('instrument_names') or [])
                    if name
                }
                legacy_name = trade.get('instrument_name')
                if legacy_name:
                    tracked_instruments.add(str(legacy_name))
                is_live = bool(tracked_instruments & thalex_instruments)
                if not tracked_instruments:
                    is_live = trade.get('asset') in thalex_assets
            else:
                is_live = trade['asset'] in tracked_assets

            if not is_live:
                self.active_trades.remove(trade)
                removed.append(trade.get('instrument_name') or trade['asset'])

        if removed:
            self.logger.info(f"Reconciled: removed stale trades for {removed}")
            self._save_active_trades()
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'action': 'reconcile',
                'removed_assets': removed,
                'note': 'Position no longer exists on exchange'
            })

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate naive Sharpe ratio from returns list"""
        if len(returns) < 2:
            return 0.0

        try:
            import statistics
            mean = statistics.mean(returns)
            stdev = statistics.stdev(returns)
            return mean / stdev if stdev > 0 else 0.0
        except Exception:
            return 0.0

    def _get_interval_seconds(self) -> int:
        """Convert interval string to seconds"""
        if self.interval.endswith('m'):
            return int(self.interval[:-1]) * 60
        elif self.interval.endswith('h'):
            return int(self.interval[:-1]) * 3600
        elif self.interval.endswith('d'):
            return int(self.interval[:-1]) * 86400
        return 300  # default 5 minutes

    def _compose_last_reasoning(self) -> None:
        """Rebuild ``state.last_reasoning`` as the merged perps + options view.

        The Reasoning page reads ``state.last_reasoning`` directly. Perps fire
        every 5 min while options fire every 3 h, so a wholesale assignment in
        either cycle would erase the other. We instead keep both halves on
        their own fields and recompose the GUI-facing dict here.

        Trade decisions from both venues are concatenated; each item already
        carries a ``venue`` field so the timeline can colour and filter them
        appropriately. Reasoning text is exposed as a ``per_venue`` mapping
        plus a top-level ``reasoning`` summary that prefers the more recent
        cycle so the JSON editor isn't empty.
        """
        perps = getattr(self.state, "last_perps_reasoning", None) or {}
        options = getattr(self.state, "last_options_reasoning", None) or {}

        # ``list(...)`` shallow-copies the outer list but leaves the inner
        # dicts shared with the per-venue reasoning state. setdefault on those
        # originals would mutate the persisted state every time the composite
        # is rebuilt. Copy each decision before tagging the venue.
        #
        # Cross-venue sentinel: any decision in the perps bucket that looks
        # like an options decision (explicit ``venue='thalex'`` or a
        # ``strategy`` field set) gets routed to the options bucket for
        # display. The perps agent prompt/schema is supposed to prevent
        # this, but if it regresses we still want the GUI to render
        # correctly and we log a warning so the drift is visible.
        raw_perps = [d for d in (perps.get("trade_decisions") or []) if isinstance(d, dict)]
        raw_options = [d for d in (options.get("trade_decisions") or []) if isinstance(d, dict)]

        def _looks_like_options(d: Dict) -> bool:
            return (
                (d.get("venue") or "").lower() == "thalex"
                or d.get("strategy") is not None
                or bool(d.get("legs"))
            )

        perps_decisions: List[Dict] = []
        stray_options: List[Dict] = []
        for d in raw_perps:
            if _looks_like_options(d):
                stray_options.append({**d, "venue": "thalex"})
                self.logger.warning(
                    "_compose_last_reasoning: routing stray options-shaped decision "
                    "out of perps bucket (asset=%s strategy=%s) — perps agent emitted "
                    "an options field it shouldn't",
                    d.get("asset"), d.get("strategy"),
                )
            else:
                perps_decisions.append({**d, "venue": d.get("venue") or "hyperliquid"})

        options_decisions = [
            {**d, "venue": d.get("venue") or "thalex"}
            for d in raw_options
        ] + stray_options

        perps_text = perps.get("reasoning") or ""
        options_text = options.get("reasoning") or ""
        summary_parts = []
        if perps_text:
            summary_parts.append(f"[PERPS]\n{perps_text}")
        if options_text:
            summary_parts.append(f"[OPTIONS]\n{options_text}")

        self.state.last_reasoning = {
            "reasoning": "\n\n".join(summary_parts),
            "trade_decisions": perps_decisions + options_decisions,
            "per_venue": {
                "hyperliquid": {
                    "reasoning": perps_text,
                    "trade_decisions": perps_decisions,
                },
                "thalex": {
                    "reasoning": options_text,
                    "trade_decisions": options_decisions,
                    "cycle_at": options.get("cycle_at"),
                },
            },
        }

    def _notify_state_update(self):
        """Notify GUI of state update via callback"""
        self._refresh_hedge_state()
        if self.on_state_update:
            try:
                self.on_state_update(self.state)
            except Exception as e:
                self.logger.error(f"Error in state update callback: {e}")

    def _refresh_hedge_state(self) -> None:
        """Copy live hedge-manager telemetry into BotState for the GUI."""
        if self.hedge_manager is None:
            self.state.hedge_status = {
                "health": "unavailable",
                "enabled": False,
                "available": False,
                "degraded_underlyings": {},
                "tracked_underlyings": 0,
                "active_underlyings": 0,
                "last_update": datetime.now(UTC).isoformat(),
            }
            self.state.hedge_metrics = []
            return

        snapshot = self.hedge_manager.get_status_snapshot()
        self.state.hedge_status = {
            "health": snapshot.get("health", "unknown"),
            "enabled": snapshot.get("enabled", self.hedge_manager.is_enabled()),
            "available": True,
            "degraded_underlyings": snapshot.get("degraded_underlyings", {}),
            "tracked_underlyings": snapshot.get("tracked_underlyings", 0),
            "active_underlyings": snapshot.get("active_underlyings", 0),
            "state_error": snapshot.get("state_error"),
            "last_update": snapshot.get("last_update", datetime.now(UTC).isoformat()),
        }
        self.state.hedge_metrics = snapshot.get("metrics", [])

    def _write_diary_entry(self, entry: Dict):
        """Write entry to diary.jsonl"""
        try:
            with open(self.diary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=json_default) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write diary entry: {e}")

    def _load_recent_diary(self, limit: int = 10) -> List[Dict]:
        """Load recent diary entries"""
        if not self.diary_path.exists():
            return []

        try:
            entries = []
            with open(self.diary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return entries[-limit:]
        except Exception as e:
            self.logger.error(f"Failed to load diary: {e}")
            return []

    def get_state(self) -> BotState:
        """Get current bot state"""
        return self.state

    def get_assets(self) -> List[str]:
        """Get configured assets"""
        return self.assets

    def get_interval(self) -> str:
        """Get configured interval"""
        return self.interval

    async def close_position(self, asset: str) -> bool:
        """
        Manually close a position for given asset.

        Args:
            asset: Asset symbol to close

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cancel all orders for this asset
            await self.hyperliquid.cancel_all_orders(asset)

            # Find position
            for pos in self.state.positions:
                if pos['symbol'] == asset:
                    quantity = abs(pos['quantity'])
                    if quantity > 0:
                        # Close position (reverse direction)
                        if pos['quantity'] > 0:  # Long position
                            await self.hyperliquid.place_sell_order(asset, quantity)
                        else:  # Short position
                            await self.hyperliquid.place_buy_order(asset, quantity)

                        # Remove from active trades
                        self.active_trades = [
                            t for t in self.active_trades if t['asset'] != asset
                        ]
                        self._save_active_trades()

                        self._write_diary_entry({
                            'timestamp': datetime.now(UTC).isoformat(),
                            'asset': asset,
                            'action': 'manual_close',
                            'quantity': quantity,
                            'note': 'Position closed manually via GUI'
                        })

                        self.logger.info(f"Manually closed position: {asset}")
                        return True

            self.logger.warning(f"No position found to close: {asset}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to close position {asset}: {e}")
            if self.on_error:
                self.on_error(f"Failed to close position: {e}")
            return False
    
    # ===== MANUAL TRADING MODE METHODS =====
    
    def get_pending_proposals(self) -> List[TradeProposal]:
        """Get list of pending trade proposals"""
        return [p for p in self.pending_proposals if p.is_pending]

    def _prune_stale_failed_proposals(self) -> None:
        """Drop failed proposals older than the TTL.

        Only acts on status=='failed'; in-flight pending proposals and
        retries-in-flight (also status=='pending') are never pruned here.
        """
        if FAILED_PROPOSAL_TTL_SECONDS <= 0:
            return
        cutoff = datetime.now(UTC) - timedelta(seconds=FAILED_PROPOSAL_TTL_SECONDS)
        before = len(self.pending_proposals)
        self.pending_proposals = [
            p for p in self.pending_proposals
            if p.status != "failed" or (p.executed_at is not None and p.executed_at > cutoff)
        ]
        dropped = before - len(self.pending_proposals)
        if dropped:
            self.logger.info(f"[RETRY] Pruned {dropped} stale failed proposal(s)")

    def _sync_pending_proposals_state(self) -> None:
        """Reflect visible (pending + failed) proposals into BotState for the UI."""
        self._prune_stale_failed_proposals()
        self.state.pending_proposals = [
            p.to_dict() for p in self.pending_proposals if p.is_visible_to_ui
        ]

    def retry_proposal(self, proposal_id: str) -> bool:
        """Re-execute a previously failed proposal.

        Returns True if a retry was scheduled; False if the proposal is not
        found or not in a retryable state (guards double-clicks).
        """
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        if proposal is None or not proposal.reset_for_retry():
            self.logger.warning(f"Proposal {proposal_id} not found or not retryable")
            return False

        self.logger.info(
            f"[RETRY] Re-executing proposal {proposal_id[:8]}: "
            f"{proposal.action.upper()} {proposal.asset}"
        )
        asyncio.create_task(self._execute_proposal(proposal))
        self._sync_pending_proposals_state()
        self._notify_state_update()
        return True

    def dismiss_proposal(self, proposal_id: str) -> bool:
        """Remove a failed proposal from the UI without retrying."""
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        if proposal is None or proposal.status != "failed":
            self.logger.warning(f"Proposal {proposal_id} not found or not dismissible")
            return False

        self.pending_proposals = [p for p in self.pending_proposals if p.id != proposal_id]
        self.logger.info(f"[DISMISSED] Failed proposal {proposal_id[:8]}")
        self._sync_pending_proposals_state()
        self._notify_state_update()
        return True

    def approve_proposal(self, proposal_id: str) -> bool:
        """
        Approve and execute a trade proposal.
        
        Args:
            proposal_id: ID of the proposal to approve
            
        Returns:
            True if proposal found and approved, False otherwise
        """
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        
        if not proposal or not proposal.is_pending:
            self.logger.warning(f"Proposal {proposal_id} not found or not pending")
            return False
        
        # Mark as approved
        proposal.approve()
        self.logger.info(f"[APPROVED] Proposal: {proposal.action.upper()} {proposal.asset} (ID: {proposal_id[:8]})")
        
        # Execute asynchronously
        asyncio.create_task(self._execute_proposal(proposal))

        # Update state
        self._sync_pending_proposals_state()
        self._notify_state_update()

        return True

    def reject_proposal(self, proposal_id: str, reason: Optional[str] = None) -> bool:
        """
        Reject a trade proposal.
        
        Args:
            proposal_id: ID of the proposal to reject
            reason: Optional reason for rejection
            
        Returns:
            True if proposal found and rejected, False otherwise
        """
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        
        if not proposal or not proposal.is_pending:
            self.logger.warning(f"Proposal {proposal_id} not found or not pending")
            return False
        
        # Mark as rejected
        proposal.reject(reason or "Rejected by user")
        self.logger.info(f"[REJECTED] Proposal: {proposal.action.upper()} {proposal.asset} (ID: {proposal_id[:8]})")
        
        # Write to diary
        self._write_diary_entry({
            'timestamp': datetime.now(UTC).isoformat(),
            'asset': proposal.asset,
            'action': 'proposal_rejected',
            'proposal_id': proposal_id,
            'reason': reason,
            'rationale': proposal.rationale
        })
        
        # Update state
        self._sync_pending_proposals_state()
        self._notify_state_update()

        return True

    async def _execute_proposal(self, proposal: TradeProposal):
        """
        Execute an approved trade proposal.
        
        Args:
            proposal: The approved proposal to execute
        """
        try:
            self.logger.info(f"Executing proposal: {proposal.action.upper()} {proposal.asset}")
            market_conditions = proposal.market_conditions or {}

            if (proposal.venue or market_conditions.get("venue") or "hyperliquid") == "thalex":
                decision_payload = market_conditions.get("decision_payload")
                if not isinstance(decision_payload, dict):
                    raise ValueError("Missing Thalex decision payload on proposal")

                ok, reason = await self._execute_thalex_decision(decision_payload)
                if not ok:
                    raise RuntimeError(reason or "Thalex execution failed")

                execution_price = float((self._last_thalex_execution or {}).get("execution_price") or 0.0)
                proposal.mark_executed(execution_price)
                self._write_diary_entry({
                    'timestamp': datetime.now(UTC).isoformat(),
                    'asset': proposal.asset,
                    'venue': 'thalex',
                    'action': proposal.action,
                    'strategy': market_conditions.get('strategy'),
                    'contracts': market_conditions.get('contracts'),
                    'target_gamma_btc': market_conditions.get('target_gamma_btc'),
                    'execution_price': execution_price,
                    'rationale': proposal.rationale,
                    'from_proposal': proposal.id,
                    'approved_manually': True,
                })

                if self.on_trade_executed:
                    self.on_trade_executed({
                        'asset': proposal.asset,
                        'venue': 'thalex',
                        'action': proposal.action,
                        'amount': proposal.size,
                        'price': execution_price,
                        'timestamp': datetime.now(UTC).isoformat(),
                        'from_proposal': True,
                    })

                self.logger.info(f"[SUCCESS] Proposal executed: {proposal.id[:8]}")
                return
            
            # Get fresh price
            current_price = await self.hyperliquid.get_current_price(proposal.asset)
            amount = proposal.size
            
            if amount <= 0:
                raise ValueError(f"Invalid amount: {amount}")
            
            # Pre-trade margin check before submitting — same rationale as
            # the auto-execute path: don't silently record a trade the
            # venue would reject for insufficient collateral.
            await self._hl_margin_preflight(proposal.asset, amount * (current_price or 0.0))

            # Place market order
            if proposal.action == 'buy':
                order_result = await self.hyperliquid.place_buy_order(proposal.asset, amount)
            elif proposal.action == 'sell':
                order_result = await self.hyperliquid.place_sell_order(proposal.asset, amount)
            else:
                raise ValueError(f"Invalid action: {proposal.action}")

            # Validate response BEFORE touching active_trades / state.
            self._hl_validate_response(
                order_result,
                context=f"proposal {proposal.action} {proposal.asset} x{amount:.6f}",
            )

            self.logger.info(f"Order placed: {proposal.action} {proposal.asset}: {amount:.6f} @ {current_price}")
            
            # Wait and check fills
            await asyncio.sleep(1)
            recent_fills = await self.hyperliquid.get_recent_fills(limit=5)
            filled = any(
                f.get('coin') == proposal.asset and
                abs(float(f.get('sz', 0)) - amount) < 0.0001
                for f in recent_fills
            )
            
            # Place TP/SL if specified
            tp_oid = None
            sl_oid = None
            
            if proposal.tp_price:
                try:
                    is_buy = (proposal.action == 'buy')
                    tp_order = await self.hyperliquid.place_take_profit(
                        proposal.asset, is_buy, amount, proposal.tp_price
                    )
                    oids = self.hyperliquid.extract_oids(tp_order)
                    tp_oid = oids[0] if oids else None
                    self.logger.info(f"Placed TP order @ {proposal.tp_price}")
                except Exception as e:
                    self.logger.error(f"Failed to place TP: {e}")
            
            if proposal.sl_price:
                try:
                    is_buy = (proposal.action == 'buy')
                    sl_order = await self.hyperliquid.place_stop_loss(
                        proposal.asset, is_buy, amount, proposal.sl_price
                    )
                    oids = self.hyperliquid.extract_oids(sl_order)
                    sl_oid = oids[0] if oids else None
                    self.logger.info(f"Placed SL order @ {proposal.sl_price}")
                except Exception as e:
                    self.logger.error(f"Failed to place SL: {e}")
            
            # Update active trades
            self.active_trades = [
                t for t in self.active_trades if t['asset'] != proposal.asset
            ]
            self.active_trades.append({
                'venue': 'hyperliquid',
                'asset': proposal.asset,
                'instrument_name': proposal.asset,
                'instrument_names': [proposal.asset],
                'is_long': (proposal.action == 'buy'),
                'amount': amount,
                'entry_price': current_price,
                'tp_oid': tp_oid,
                'sl_oid': sl_oid,
                'exit_plan': market_conditions.get('exit_plan', ''),
                # Deep-memory fields: preserve thesis across cycles + restarts.
                'rationale': proposal.rationale,
                'tp_price': proposal.tp_price,
                'sl_price': proposal.sl_price,
                'confidence': getattr(proposal, 'confidence', None),
                'opened_at': datetime.now(UTC).isoformat(),
                'from_proposal': proposal.id
            })
            self._save_active_trades()
            
            # Mark proposal as executed
            proposal.mark_executed(current_price)
            
            # Write to diary
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'asset': proposal.asset,
                'action': proposal.action,
                'allocation_usd': proposal.allocation,
                'amount': amount,
                'entry_price': current_price,
                'tp_price': proposal.tp_price,
                'tp_oid': tp_oid,
                'sl_price': proposal.sl_price,
                'sl_oid': sl_oid,
                'rationale': proposal.rationale,
                'order_result': str(order_result),
                'filled': filled,
                'from_proposal': proposal.id,
                'approved_manually': True
            })
            
            # Notify GUI
            if self.on_trade_executed:
                self.on_trade_executed({
                    'asset': proposal.asset,
                    'action': proposal.action,
                    'amount': amount,
                    'price': current_price,
                    'timestamp': datetime.now(UTC).isoformat(),
                    'from_proposal': True
                })
            
            self.logger.info(f"[SUCCESS] Proposal executed: {proposal.id[:8]}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute proposal {proposal.id}: {e}")
            proposal.mark_failed(str(e))
            
            if self.on_error:
                self.on_error(f"Failed to execute trade: {e}")
        
        finally:
            # Update state — failed proposals stay visible so the UI can offer Retry.
            self._sync_pending_proposals_state()
            self._notify_state_update()
