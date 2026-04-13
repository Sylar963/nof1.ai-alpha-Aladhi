"""
Trading Bot Engine - Core trading logic separated from UI
Refactored from ai-trading-agent/src/main.py
"""

import asyncio
import json
import logging
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from src.backend.agent.decision_maker import TradingAgent
from src.backend.agent.decision_schema import DecisionParseError, parse_decision
from src.backend.agent.options_llm_lifecycle import OptionsLLMLifecycle
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_client import TAAPIClient
from src.backend.models.trade_proposal import TradeProposal
from src.backend.trading.delta_hedge_manager import DeltaHedgeManager
from src.backend.trading.hyperliquid_api import HyperliquidAPI
from src.backend.trading.options_scheduler import (
    OptionsScheduler,
    OptionsSchedulerConfig,
)
from src.backend.trading.options_strategies import DeltaHedger, OptionsExecutor
from src.backend.utils.prompt_utils import json_default


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
    last_reasoning: Dict = field(default_factory=dict)
    last_update: str = ""
    error: Optional[str] = None
    invocation_count: int = 0


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
        self.active_trades: List[Dict] = []  # Local tracking of open positions
        self.recent_events: deque = deque(maxlen=200)
        self.initial_account_value: Optional[float] = None
        self._last_thalex_execution: Dict[str, Any] = {}
        self.price_history: Dict[str, deque] = {asset: deque(maxlen=60) for asset in assets}
        
        # Manual trading mode
        self.trading_mode = CONFIG.get("trading_mode", "auto").lower()
        self.pending_proposals: List[TradeProposal] = []
        self.logger.info(f"Trading mode: {self.trading_mode.upper()}")

        # File paths
        self.diary_path = Path("data/diary.jsonl")
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_thalex_proposal(self, decision_payload: Dict) -> TradeProposal:
        """Build a manual-approval proposal for a Thalex decision payload."""
        decision = parse_decision(decision_payload)
        size = float(decision.contracts or decision.target_gamma_btc or 0.0)
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

        if decision.sl_price is not None:
            message = "Thalex options do not support native stop-loss triggers; refusing unprotected SL order"
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

            if decision.tp_price is not None and len([name for name in instrument_names if name]) == 1:
                try:
                    await self.thalex.place_take_profit(
                        instrument_names[0],
                        decision.action == "buy",
                        float(decision.contracts or 0.0),
                        float(decision.tp_price),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.warning("Failed to place Thalex take-profit for %s: %s", instrument_names[0], exc)

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

        # Get initial account value
        try:
            user_state = await self.hyperliquid.get_user_state()
            thalex_state = None
            if self.thalex is not None:
                try:
                    thalex_state = await self.thalex.get_user_state()
                except Exception as exc:  # pylint: disable=broad-except
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
                await self.hedge_manager.reconcile()
            except Exception as exc:  # pylint: disable=broad-except
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

            deribit = DeribitPublicClient()
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
                await deribit.close()
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

        Reads ``self._latest_options_context`` (populated by the 15m refresh
        task), calls the OptionsAgent, parses decisions, and routes each one
        through ``_execute_thalex_decision``. Skips silently if no snapshot
        is available yet (typical on first run before the surface refresh
        has had a chance to populate it).
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
                        self.state.pending_proposals = [
                            p.to_dict() for p in self.pending_proposals if p.is_pending
                        ]
                    except Exception as exc:  # pylint: disable=broad-except
                        self.logger.error(
                            "Error creating scheduled Thalex proposal for %s: %s",
                            decision.asset,
                            exc,
                        )
                    continue
                await self._execute_thalex_decision(decision_payload)
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

    async def _main_loop(self):
        """
        Main trading loop.
        Adapted from ai-trading-agent/src/main.py lines 88-455
        """
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
                                'unrealized_pnl': pos.get('pnl', 0.0),
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
                    dashboard = {
                        'total_return_pct': total_return_pct,
                        'balance': balance,
                        'account_value': total_value,
                        'sharpe_ratio': sharpe_ratio,
                        'positions': combined_positions,
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

                            # Fetch the curated TAAPI bundle off the event loop so
                            # sync requests + pacing do not stall hedge audits or shutdown.
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
                            
                            # Add delay between assets to respect TAAPI rate limit (1 req/15s)
                            # Only wait if this is not the last asset
                            if idx < len(self.assets) - 1:
                                self.logger.info(f"Waiting 15s before fetching next asset (TAAPI rate limit)...")
                                await asyncio.sleep(15)
                            
                            # Extract 5m indicators
                            sma99_5m_series = indicators["5m"].get("sma99", [])
                            avwap_5m = indicators["5m"].get("avwap")
                            keltner_5m = indicators["5m"].get("keltner", {})
                            opening_range = indicators["5m"].get("opening_range", {})

                            # Extract long-term indicators (interval from config: 1h, 4h, etc.)
                            interval = CONFIG.get("interval", "4h")
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
                                        "timestamps": recent_timestamps[-len(keltner_5m_middle):] if keltner_5m_middle else [],
                                        "price_candles": {},
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
                                        "timestamps": recent_timestamps[-len(lt_keltner_middle):] if lt_keltner_middle else [],
                                        "price_candles": {},
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

                    self.state.last_reasoning = decisions

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
                                    self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
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
                                    size = allocation / current_price if current_price > 0 else 0
                                    
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
                                    self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
                                    
                                except Exception as e:
                                    self.logger.error(f"Error creating proposal for {asset}: {e}")
                                    
                                continue  # Skip execution in manual mode
                            
                            # AUTO MODE: Execute immediately (original behavior)
                            try:
                                current_price = await self.hyperliquid.get_current_price(asset)
                                amount = allocation / current_price if current_price > 0 else 0

                                if amount > 0:
                                    # Place market order
                                    if action == 'buy':
                                        order_result = await self.hyperliquid.place_buy_order(asset, amount)
                                    else:
                                        order_result = await self.hyperliquid.place_sell_order(asset, amount)

                                    self.logger.info(f"Executed {action} {asset}: {amount:.6f} @ {current_price}")

                                    # Wait and check fills
                                    await asyncio.sleep(1)
                                    recent_fills_check = await self.hyperliquid.get_recent_fills(limit=5)
                                    filled = any(
                                        f.get('coin') == asset and
                                        abs(float(f.get('sz', 0)) - amount) < 0.0001
                                        for f in recent_fills_check
                                    )

                                    # Place TP/SL orders
                                    tp_oid = None
                                    sl_oid = None

                                    if tp_price:
                                        try:
                                            is_buy = (action == 'buy')
                                            tp_order = await self.hyperliquid.place_take_profit(
                                                asset, is_buy, amount, tp_price
                                            )
                                            oids = self.hyperliquid.extract_oids(tp_order)
                                            tp_oid = oids[0] if oids else None
                                            self.logger.info(f"Placed TP order for {asset} @ {tp_price}")
                                        except Exception as e:
                                            self.logger.error(f"Failed to place TP: {e}")

                                    if sl_price:
                                        try:
                                            is_buy = (action == 'buy')
                                            sl_order = await self.hyperliquid.place_stop_loss(
                                                asset, is_buy, amount, sl_price
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
                                    self.active_trades.append({
                                        'venue': 'hyperliquid',
                                        'asset': asset,
                                        'instrument_name': asset,
                                        'instrument_names': [asset],
                                        'is_long': (action == 'buy'),
                                        'amount': amount,
                                        'entry_price': current_price,
                                        'tp_oid': tp_oid,
                                        'sl_oid': sl_oid,
                                        'exit_plan': exit_plan,
                                        'opened_at': datetime.now(UTC).isoformat()
                                    })

                                    # Write to diary
                                    self._write_diary_entry({
                                        'timestamp': datetime.now(UTC).isoformat(),
                                        'asset': asset,
                                        'action': action,
                                        'allocation_usd': allocation,
                                        'amount': amount,
                                        'entry_price': current_price,
                                        'tp_price': tp_price,
                                        'tp_oid': tp_oid,
                                        'sl_price': sl_price,
                                        'sl_oid': sl_oid,
                                        'exit_plan': exit_plan,
                                        'rationale': rationale,
                                        'order_result': str(order_result),
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

                                    # Track PnL for Sharpe
                                    # (Simplified - actual PnL tracked on position close)

                            except Exception as e:
                                self.logger.error(f"Error executing {action} for {asset}: {e}")
                                if self.on_error:
                                    self.on_error(f"Trade execution error: {e}")

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

                except Exception as e:
                    self.logger.error(f"Error in main loop iteration: {e}", exc_info=True)
                    self.state.error = str(e)
                    if self.on_error:
                        self.on_error(str(e))

                # ===== PHASE 12: Sleep Until Next Interval =====
                await asyncio.sleep(self._get_interval_seconds())

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
        self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
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
        self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
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
            
            # Place market order
            if proposal.action == 'buy':
                order_result = await self.hyperliquid.place_buy_order(proposal.asset, amount)
            elif proposal.action == 'sell':
                order_result = await self.hyperliquid.place_sell_order(proposal.asset, amount)
            else:
                raise ValueError(f"Invalid action: {proposal.action}")
            
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
                'opened_at': datetime.now(UTC).isoformat(),
                'from_proposal': proposal.id
            })
            
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
            # Update state
            self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
            self._notify_state_update()
