"""
Bot Service - Manages bot lifecycle and provides data access for GUI
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import UTC, datetime

from src.backend.bot_engine import TradingBotEngine, BotState
from src.backend.config_loader import CONFIG


class BotService:
    """Service layer for bot management and data access"""

    @staticmethod
    def _parse_assets(raw_assets: str | List[str] | None) -> List[str]:
        """Normalize configured assets from env, settings UI, or persisted state."""
        if raw_assets is None:
            return []
        if isinstance(raw_assets, str):
            return [asset.strip() for asset in raw_assets.replace(',', ' ').split() if asset.strip()]
        return [str(asset).strip() for asset in raw_assets if str(asset).strip()]

    def __init__(self):
        self.bot_engine: Optional[TradingBotEngine] = None
        self.state_manager = None  # Set externally after creation
        self.logger = logging.getLogger(__name__)
        self.delta_hedge_enabled = bool(CONFIG.get('delta_hedge_enabled', True))
        self._reset_session_trackers()

        # Configuration
        self.config = {
            'assets': self._parse_assets(CONFIG.get('assets')) or ['BTC', 'ETH'],
            'interval': CONFIG.get('interval', '5m'),
            'model': CONFIG.get('llm_model', 'x-ai/grok-4')
        }

    async def start(self, assets: Optional[List[str]] = None, interval: Optional[str] = None):
        """
        Start the trading bot.

        Args:
            assets: List of assets to trade (optional, uses config if not provided)
            interval: Trading interval (optional, uses config if not provided)
        """
        if self.bot_engine and self.bot_engine.is_running:
            self.logger.warning("Bot already running")
            return

        # Validate API keys before starting
        if not CONFIG.get('taapi_api_key'):
            raise ValueError("TAAPI_API_KEY not configured. Please set it in .env file.")
        if not CONFIG.get('openrouter_api_key'):
            raise ValueError("OPENROUTER_API_KEY not configured. Please set it in .env file.")
        if not CONFIG.get('hyperliquid_private_key') and not CONFIG.get('mnemonic'):
            raise ValueError("HYPERLIQUID_PRIVATE_KEY or MNEMONIC not configured. Please set it in .env file.")

        # Optional Thalex venue: only validate if any Thalex env var is present.
        if CONFIG.get('thalex_key_id') or CONFIG.get('thalex_private_key_path'):
            from pathlib import Path as _Path
            if not CONFIG.get('thalex_key_id'):
                raise ValueError("THALEX_KEY_ID is required when THALEX_PRIVATE_KEY_PATH is set.")
            pem = CONFIG.get('thalex_private_key_path')
            if not pem or not _Path(pem).exists():
                raise ValueError(f"THALEX_PRIVATE_KEY_PATH file not found: {pem}")
            net = (CONFIG.get('thalex_network') or 'test').lower()
            if net not in {'test', 'prod'}:
                raise ValueError(f"THALEX_NETWORK must be 'test' or 'prod', got {net!r}.")

        # Use provided values or fall back to config
        assets = assets or self.config['assets']
        interval = interval or self.config['interval']

        if not assets or not interval:
            raise ValueError("Assets and interval must be configured. Set ASSETS and INTERVAL in .env file.")

        try:
            # Create bot engine with callbacks
            self.bot_engine = TradingBotEngine(
                assets=assets,
                interval=interval,
                delta_hedge_enabled=self.delta_hedge_enabled,
                on_state_update=self._on_state_update,
                on_trade_executed=self._on_trade_executed,
                on_error=self._on_error
            )

            self._reset_session_trackers()

            # Start the bot
            await self.bot_engine.start()

            self.logger.info(f"Bot started successfully - Assets: {assets}, Interval: {interval}")

        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}", exc_info=True)
            raise

    def _reset_session_trackers(self):
        """Clear per-run GUI state so a fresh session starts cleanly."""
        self.equity_history = []
        self.recent_events = []
        self._last_hedge_health = None
        self._last_degraded_underlyings = {}
        self._last_hedge_state_error = None

    @staticmethod
    def _normalize_position(position: Dict, price_map: Dict[str, float | None]) -> Dict:
        """Convert raw exchange positions into the UI's normalized shape."""
        def _field(*names: str):
            if isinstance(position, dict):
                for name in names:
                    if name in position:
                        return position.get(name)
                return None
            for name in names:
                if hasattr(position, name):
                    return getattr(position, name)
            return None

        venue = str(_field('venue') or 'hyperliquid').lower()
        asset = str(_field('asset', 'symbol', 'coin') or '')
        instrument_name = str(_field('instrument_name') or asset)
        symbol = instrument_name if venue == 'thalex' else asset

        quantity = float(_field('quantity', 'szi', 'size') or 0)
        side = str(_field('side') or '').lower()
        if venue == 'thalex' and side in {'long', 'short'}:
            quantity = abs(quantity) if side == 'long' else -abs(quantity)

        entry_price = float(_field('entry_price', 'entryPx') or 0)
        current_price = _field('current_price')
        if current_price is None:
            current_price = price_map.get(asset) or price_map.get(symbol)
        current_price = float(current_price or 0)
        unrealized_pnl = _field('unrealized_pnl', 'pnl')
        liquidation_price = float(_field('liquidation_price', 'liquidationPx') or 0)
        leverage = _field('leverage') or 1
        if isinstance(leverage, dict):
            leverage = leverage.get('value', 1)

        return {
            'row_id': f'{venue}:{instrument_name}',
            'symbol': symbol,
            'asset': asset,
            'instrument_name': instrument_name,
            'venue': venue,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': float(unrealized_pnl or 0),
            'liquidation_price': liquidation_price,
            'leverage': leverage or 1,
            'opened_by': 'External',
            'closable': venue == 'hyperliquid',
        }

    @staticmethod
    def _latest(series: List[float]) -> float | None:
        return series[-1] if series else None

    @staticmethod
    def _account_field(state: object, name: str, default: float = 0.0) -> float:
        if isinstance(state, dict):
            value = state.get(name, default)
        else:
            value = getattr(state, name, default)
        return float(value or 0.0)

    @classmethod
    def _aggregate_account_state(cls, hyperliquid_state: Dict, thalex_state: object | None = None) -> Dict:
        balance_breakdown = {
            'hyperliquid': cls._account_field(hyperliquid_state, 'balance'),
        }
        total_value_breakdown = {
            'hyperliquid': cls._account_field(
                hyperliquid_state,
                'total_value',
                default=balance_breakdown['hyperliquid'],
            ),
        }

        if thalex_state is not None:
            balance_breakdown['thalex'] = cls._account_field(thalex_state, 'balance')
            total_value_breakdown['thalex'] = cls._account_field(
                thalex_state,
                'total_value',
                default=balance_breakdown['thalex'],
            )

        return {
            'balance': sum(balance_breakdown.values()),
            'total_value': sum(total_value_breakdown.values()),
            'balance_breakdown': balance_breakdown,
            'total_value_breakdown': total_value_breakdown,
        }

    @classmethod
    def _keltner_snapshot(cls, keltner: Dict, current_price: float | None) -> Dict:
        middle_series = list((keltner or {}).get('middle') or [])
        upper_series = list((keltner or {}).get('upper') or [])
        lower_series = list((keltner or {}).get('lower') or [])
        middle = cls._latest(middle_series)
        upper = cls._latest(upper_series)
        lower = cls._latest(lower_series)
        position = 'unknown'
        if all(v is not None for v in (upper, lower, current_price)):
            if current_price > upper:
                position = 'above'
            elif current_price < lower:
                position = 'below'
            else:
                position = 'inside'
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'position': position,
        }

    @classmethod
    def _build_indicator_frame(cls, frame: Dict, current_price: float | None, interval: str | None = None) -> Dict:
        frame = frame or {}
        keltner = frame.get('keltner') or {}
        shaped = {
            'sma99': cls._latest(list(frame.get('sma99') or [])),
            'avwap': frame.get('avwap'),
            'keltner': cls._keltner_snapshot(keltner, current_price),
            'opening_range': frame.get('opening_range') or {},
            'series': {
                'sma99': list(frame.get('sma99') or []),
                'keltner_middle': list(keltner.get('middle') or []),
                'keltner_upper': list(keltner.get('upper') or []),
                'keltner_lower': list(keltner.get('lower') or []),
                'timestamps': list(frame.get('timestamps') or []),
                'price_candles': frame.get('price_candles') or {},
            },
        }
        if interval:
            shaped['interval'] = interval
        return shaped

    @classmethod
    def _build_market_sections(
        cls,
        assets: List[str],
        market_data: Dict[str, Dict],
        indicator_payloads: Dict[str, Dict] | None = None,
    ) -> List[Dict]:
        """Shape manual refresh market data like the running bot state."""
        sections: List[Dict] = []
        indicator_payloads = indicator_payloads or {}
        for asset in assets:
            snapshot = market_data.get(asset, {})
            price = snapshot.get('price')
            indicators = indicator_payloads.get(asset) or {}
            long_term_interval = str(CONFIG.get('interval', '4h'))
            sections.append({
                'asset': asset,
                'current_price': price,
                'price': price,
                'funding_rate': snapshot.get('funding_rate'),
                'open_interest': snapshot.get('open_interest'),
                'prev_day_price': snapshot.get('prev_day_price'),
                'volume_24h': snapshot.get('volume_24h'),
                'timestamp': snapshot.get('timestamp'),
                'intraday': cls._build_indicator_frame(indicators.get('5m') or {}, price),
                'long_term': cls._build_indicator_frame(indicators.get(long_term_interval) or {}, price, interval=long_term_interval),
            })
        return sections

    async def stop(self):
        """Stop the trading bot"""
        if not self.bot_engine:
            return

        try:
            await self.bot_engine.stop()
            self.logger.info("Bot stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}", exc_info=True)
            raise

    def is_running(self) -> bool:
        """Check if bot is currently running"""
        return self.bot_engine is not None and self.bot_engine.is_running

    def supports_delta_hedge(self) -> bool:
        if self.bot_engine is not None:
            return self.bot_engine.supports_delta_hedge()
        return bool(CONFIG.get('thalex_key_id') and CONFIG.get('thalex_private_key_path'))

    def is_delta_hedge_enabled(self) -> bool:
        if self.bot_engine is not None and self.bot_engine.supports_delta_hedge():
            return self.bot_engine.is_delta_hedge_enabled()
        return self.supports_delta_hedge() and self.delta_hedge_enabled

    async def set_delta_hedge_enabled(self, enabled: bool) -> bool:
        if not self.supports_delta_hedge():
            return False

        enabled = bool(enabled)
        self.delta_hedge_enabled = enabled

        if self.bot_engine is not None:
            changed = await self.bot_engine.set_delta_hedge_enabled(enabled)
            if not changed:
                return False

        if self.state_manager and (self.bot_engine is None or not self.bot_engine.is_running):
            state = self.state_manager.get_state()
            state.hedge_status = {
                'health': 'idle' if enabled else 'disabled',
                'enabled': enabled,
                'available': True,
                'degraded_underlyings': {},
                'tracked_underlyings': 0,
                'active_underlyings': 0,
                'state_error': None,
                'last_update': datetime.now(UTC).isoformat(),
            }
            if not enabled:
                state.hedge_metrics = []
            self.state_manager.update(state)

        self._add_event(
            '🛡️ Delta hedge enabled' if enabled else '⏸ Delta hedge disabled',
            level='info',
        )
        return True

    def get_state(self) -> BotState:
        """Get current bot state"""
        if self.bot_engine:
            return self.bot_engine.get_state()
        return BotState()

    def get_equity_history(self, limit: int = 100) -> List[Dict]:
        """
        Get equity curve history for charting.

        Returns:
            List of dicts with 'time' and 'value' keys
        """
        return self.equity_history[-limit:]

    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """
        Get recent activity events for activity feed.

        Returns:
            List of event dicts with 'time' and 'message' keys
        """
        return self.recent_events[-limit:]

    def get_trade_history(
        self,
        asset: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get trade history from diary.jsonl with optional filtering.

        Args:
            asset: Filter by asset (optional)
            action: Filter by action (buy/sell/hold) (optional)
            limit: Maximum number of entries to return

        Returns:
            List of trade entries
        """
        diary_path = Path("data/diary.jsonl")
        if not diary_path.exists():
            return []

        try:
            entries = []
            with open(diary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)

                            # Apply filters
                            if asset and entry.get('asset') != asset:
                                continue
                            if action and entry.get('action') != action:
                                continue

                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue

            return entries[-limit:]

        except Exception as e:
            self.logger.error(f"Failed to load trade history: {e}")
            return []

    async def close_position(self, asset: str) -> bool:
        """
        Manually close a position via GUI.

        Args:
            asset: Asset symbol to close

        Returns:
            True if successful, False otherwise
        """
        if not self.bot_engine:
            self.logger.error("Bot engine not initialized")
            return False

        try:
            success = await self.bot_engine.close_position(asset)
            if success:
                self._add_event(f"Manually closed position: {asset}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return False

    def update_config(self, config: Dict):
        """
        Update bot configuration.

        Args:
            config: Dict with 'assets', 'interval', 'model' keys
        """
        if 'assets' in config:
            self.config['assets'] = config['assets']
        if 'interval' in config:
            self.config['interval'] = config['interval']
        if 'model' in config:
            self.config['model'] = config['model']

        self.logger.info(f"Configuration updated: {self.config}")

    def get_assets(self) -> List[str]:
        """Get configured assets list"""
        if self.bot_engine:
            return self.bot_engine.get_assets()
        return self.config['assets']

    async def refresh_market_data(self, include_indicators: bool = False) -> bool:
        """
        Manually refresh market data from Hyperliquid without starting the bot.
        Fetches account state, positions, and market data (prices, funding rates).
        Optionally includes the curated TAAPI indicator bundle for the market page.

        Returns:
            True if successful, False otherwise
        """
        try:
            from src.backend.trading.hyperliquid_api import HyperliquidAPI

            hyperliquid = HyperliquidAPI()
            taapi = None
            indicator_payloads: Dict[str, Dict] = {}
            rate_limit_pause = None

            if include_indicators and CONFIG.get('taapi_api_key'):
                from src.backend.indicators.taapi_client import TAAPIClient

                taapi = TAAPIClient()
                loop = asyncio.get_running_loop()

                def _pause_for_taapi_rate_limit() -> None:
                    future = asyncio.run_coroutine_threadsafe(asyncio.sleep(15), loop)
                    future.result()

                rate_limit_pause = _pause_for_taapi_rate_limit

            # Fetch account state (balance, positions)
            user_state = await hyperliquid.get_user_state()

            # Fetch current market data for all configured assets
            assets = self.get_assets()
            market_data = {}

            for idx, asset in enumerate(assets):
                try:
                    price = await hyperliquid.get_current_price(asset)
                    funding_rate = await hyperliquid.get_funding_rate(asset)
                    open_interest = await hyperliquid.get_open_interest(asset)
                    prev_day_price = await hyperliquid.get_prev_day_price(asset)
                    volume_24h = await hyperliquid.get_daily_notional_volume(asset)

                    market_data[asset] = {
                        'price': price,
                        'funding_rate': funding_rate,
                        'open_interest': open_interest,
                        'prev_day_price': prev_day_price,
                        'volume_24h': volume_24h,
                        'timestamp': datetime.now(UTC).isoformat()
                    }

                    if taapi:
                        try:
                            indicator_payloads[asset] = await asyncio.to_thread(
                                taapi.fetch_asset_indicators,
                                asset,
                                current_spot=price,
                                request_pause=rate_limit_pause,
                                include_chart_data=True,
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to fetch indicators for {asset}: {e}")
                            indicator_payloads[asset] = {}
                        if idx < len(assets) - 1:
                            await asyncio.sleep(15)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch market data for {asset}: {e}")
                    market_data[asset] = {
                        'price': None,
                        'funding_rate': None,
                        'open_interest': None,
                        'prev_day_price': None,
                        'volume_24h': None,
                        'timestamp': datetime.now(UTC).isoformat()
                    }
                    indicator_payloads[asset] = {}

            # Update bot state with fresh data (create new state if bot not running)
            if not self.bot_engine:
                # Create a temporary bot state for display
                state = BotState()
            else:
                state = self.bot_engine.get_state()

            price_map = {
                asset: snapshot.get('price')
                for asset, snapshot in market_data.items()
            }
            thalex_state = None
            normalized_positions = [
                self._normalize_position(position, price_map)
                for position in (user_state.get('positions') or [])
            ]

            if self.supports_delta_hedge():
                thalex = None
                try:
                    from src.backend.trading.thalex_api import ThalexAPI

                    thalex = ThalexAPI()
                    thalex_state = await thalex.get_user_state()
                    normalized_positions.extend(
                        self._normalize_position(position, price_map)
                        for position in (getattr(thalex_state, 'positions', []) or [])
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to refresh Thalex portfolio positions: {e}")
                finally:
                    if thalex is not None:
                        try:
                            await thalex.disconnect()
                        except Exception:
                            pass

            account_snapshot = self._aggregate_account_state(user_state, thalex_state)

            # Update with fresh market data
            state.balance = account_snapshot['balance']
            state.total_value = account_snapshot['total_value']
            state.balance_breakdown = account_snapshot['balance_breakdown']
            state.total_value_breakdown = account_snapshot['total_value_breakdown']
            state.positions = normalized_positions
            state.market_data = self._build_market_sections(assets, market_data, indicator_payloads)
            state.last_update = datetime.now(UTC).isoformat()

            # Update state manager
            if self.state_manager:
                self.state_manager.update(state)

            # Add event to activity feed
            self._add_event(
                "📊 Market data refreshed - Balance: "
                f"${state.balance:,.2f} (HL ${state.balance_breakdown.get('hyperliquid', 0.0):,.2f}"
                f", Thalex ${state.balance_breakdown.get('thalex', 0.0):,.2f})"
            )

            self.logger.info("Market data refreshed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to refresh market data: {e}", exc_info=True)
            self._add_event(f"❌ Refresh failed: {str(e)}", level="error")
            return False

    def approve_proposal(self, proposal_id: str) -> bool:
        """
        Approve and execute a trade proposal.

        Args:
            proposal_id: ID of the proposal to approve

        Returns:
            True if approval was sent (async execution), False if bot not running
        """
        if not self.bot_engine or not self.bot_engine.is_running:
            self.logger.error("Bot engine not running - cannot approve proposal")
            return False

        try:
            success = self.bot_engine.approve_proposal(proposal_id)
            if success:
                self._add_event(f"✅ Proposal {proposal_id[:8]} approved - executing trade")
                self.logger.info(f"Proposal approved: {proposal_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to approve proposal: {e}")
            self._add_event(f"❌ Approval failed: {str(e)}", level="error")
            return False

    def reject_proposal(self, proposal_id: str, reason: str = "User rejected") -> bool:
        """
        Reject a trade proposal.

        Args:
            proposal_id: ID of the proposal to reject
            reason: Reason for rejection (optional)

        Returns:
            True if rejection was sent (async execution), False if bot not running
        """
        if not self.bot_engine or not self.bot_engine.is_running:
            self.logger.error("Bot engine not running - cannot reject proposal")
            return False

        try:
            success = self.bot_engine.reject_proposal(proposal_id, reason)
            if success:
                self._add_event(f"❌ Proposal {proposal_id[:8]} rejected - {reason}")
                self.logger.info(f"Proposal rejected: {proposal_id} - Reason: {reason}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to reject proposal: {e}")
            self._add_event(f"❌ Rejection failed: {str(e)}", level="error")
            return False

    def get_pending_proposals(self) -> List[Dict]:
        """
        Get list of pending trade proposals.

        Returns:
            List of proposal dicts, or empty list if bot not running
        """
        if not self.bot_engine:
            return []

        try:
            proposals = self.bot_engine.get_pending_proposals()
            # Convert TradeProposal objects to dicts for JSON serialization
            return [
                {
                    'id': p.id,
                    'asset': p.asset,
                    'action': p.action,
                    'entry_price': p.entry_price,
                    'tp_price': p.tp_price,
                    'sl_price': p.sl_price,
                    'amount': p.amount,
                    'confidence': p.confidence,
                    'risk_reward_ratio': p.risk_reward_ratio,
                    'status': p.status,
                    'rationale': p.rationale,
                    'created_at': p.created_at.isoformat() if p.created_at else None
                }
                for p in proposals
            ]
        except Exception as e:
            self.logger.error(f"Failed to get pending proposals: {e}")
            return []

    # ===== Callback Handlers =====

    def _on_state_update(self, state: BotState):
        """
        Callback when bot state updates.
        Updates state manager and tracks equity history.
        """
        if self.state_manager:
            self.state_manager.update(state)

        # Track equity history for charting
        self.equity_history.append({
            'time': state.last_update or datetime.now(UTC).isoformat(),
            'value': state.total_value
        })

        self._track_hedge_events(state)

        # Keep only last 500 points
        if len(self.equity_history) > 500:
            self.equity_history = self.equity_history[-500:]

    def _track_hedge_events(self, state: BotState):
        """Emit activity-feed events when hedge health changes."""
        hedge_status = getattr(state, 'hedge_status', {}) or {}
        hedge_health = hedge_status.get('health')
        degraded = dict(hedge_status.get('degraded_underlyings') or {})
        state_error = hedge_status.get('state_error')

        if hedge_health and hedge_health != self._last_hedge_health:
            suppress_passive_boot_event = (
                self._last_hedge_health is None
                and not getattr(state, 'is_running', False)
                and hedge_health in {'disabled', 'idle'}
            )
            if not suppress_passive_boot_event:
                if hedge_health == 'degraded':
                    self._add_event('⚠️ Delta hedge health degraded', level='error')
                elif hedge_health == 'healthy':
                    self._add_event('✅ Delta hedge healthy', level='info')
                elif hedge_health == 'disabled':
                    self._add_event('⏸ Delta hedge disabled', level='info')
                elif hedge_health == 'idle':
                    self._add_event('ℹ️ Delta hedge idle', level='info')
            self._last_hedge_health = hedge_health

        for underlying, reason in degraded.items():
            if self._last_degraded_underlyings.get(underlying) != reason:
                self._add_event(f'⚠️ Hedge degraded for {underlying}: {reason}', level='error')

        for underlying in set(self._last_degraded_underlyings) - set(degraded):
            self._add_event(f'✅ Hedge recovered for {underlying}', level='info')

        if state_error and state_error != self._last_hedge_state_error:
            self._add_event(f'⚠️ Thalex unavailable: {state_error}', level='error')
        if self._last_hedge_state_error and not state_error:
            self._add_event('✅ Thalex hedge state restored', level='info')

        self._last_degraded_underlyings = degraded
        self._last_hedge_state_error = state_error

    def _on_trade_executed(self, trade: Dict):
        """
        Callback when trade is executed.
        Adds event to activity feed.
        """
        asset = trade.get('asset', '')
        action = trade.get('action', '').upper()
        amount = trade.get('amount', 0)
        price = trade.get('price', 0)

        message = f"{action} {amount:.6f} {asset} @ ${price:,.2f}"
        self._add_event(message)

    def _on_error(self, error: str):
        """
        Callback when error occurs.
        Adds error to activity feed.
        """
        self._add_event(f"ERROR: {error}", level="error")

    def _add_event(self, message: str, level: str = "info"):
        """Add event to recent events feed"""
        self.recent_events.append({
            'time': datetime.now(UTC).strftime("%H:%M:%S"),
            'message': message,
            'level': level
        })

        # Keep only last 200 events
        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]

    # ===== Configuration Management =====

    async def update_config(self, config_updates: Dict) -> bool:
        """Update bot configuration and save to file"""
        try:
            if 'assets' in config_updates:
                assets = self._parse_assets(config_updates['assets'])
                self.config['assets'] = list(assets)
                CONFIG['assets'] = ' '.join(self.config['assets'])

            if 'interval' in config_updates:
                self.config['interval'] = config_updates['interval']
                CONFIG['interval'] = config_updates['interval']

            model = config_updates.get('llm_model', config_updates.get('model'))
            if model is not None:
                self.config['model'] = model
                CONFIG['llm_model'] = model

            # Save to .env-like configuration
            for key, value in config_updates.items():
                if key in {'assets', 'interval', 'model', 'llm_model'}:
                    continue
                if isinstance(value, list):
                    CONFIG[key] = ' '.join(value)
                else:
                    CONFIG[key] = value

            # Also save to data/config.json for persistence
            self._save_config_file()

            self.logger.info(f"Configuration updated: {list(config_updates.keys())}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    async def get_current_config(self) -> Dict:
        """Get current configuration"""
        try:
            # Load from CONFIG dict
            return {
                'assets': self._parse_assets(CONFIG.get('assets', 'BTC ETH')),
                'interval': CONFIG.get('interval', '5m'),
                'llm_model': CONFIG.get('llm_model', 'x-ai/grok-4'),
                'reasoning_enabled': CONFIG.get('reasoning_enabled', False),
                'reasoning_effort': CONFIG.get('reasoning_effort', 'high'),
                'taapi_key': CONFIG.get('taapi_api_key', ''),
                'hyperliquid_private_key': CONFIG.get('hyperliquid_private_key', ''),
                'hyperliquid_network': CONFIG.get('hyperliquid_network', 'mainnet'),
                'openrouter_key': CONFIG.get('openrouter_api_key', ''),
                'thalex_network': CONFIG.get('thalex_network', 'test'),
                'thalex_key_id': CONFIG.get('thalex_key_id', ''),
                'thalex_private_key_path': CONFIG.get('thalex_private_key_path', ''),
                'thalex_account': CONFIG.get('thalex_account', ''),
                'max_position_size': CONFIG.get('max_position_size', 1000),
                'max_leverage': CONFIG.get('max_leverage', 5),
                'desktop_notifications': CONFIG.get('desktop_notifications', True),
                'telegram_notifications': CONFIG.get('telegram_notifications', False),
                'telegram_token': CONFIG.get('telegram_token', ''),
                'telegram_chat_id': CONFIG.get('telegram_chat_id', ''),
            }
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            return {}

    def _save_config_file(self):
        """Save configuration to data/config.json"""
        try:
            config_path = Path('data/config.json')
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'strategy': {
                    'assets': CONFIG.get('assets', 'BTC ETH'),
                    'interval': CONFIG.get('interval', '5m'),
                    'llm_model': CONFIG.get('llm_model', 'x-ai/grok-4'),
                },
                'api_keys': {
                    'taapi_api_key': CONFIG.get('taapi_api_key', ''),
                    'hyperliquid_private_key': CONFIG.get('hyperliquid_private_key', ''),
                    'openrouter_api_key': CONFIG.get('openrouter_api_key', ''),
                },
                'risk_management': {
                    'max_position_size': CONFIG.get('max_position_size', 1000),
                    'max_leverage': CONFIG.get('max_leverage', 5),
                },
                'notifications': {
                    'desktop_enabled': CONFIG.get('desktop_notifications', True),
                    'telegram_enabled': CONFIG.get('telegram_notifications', False),
                    'telegram_token': CONFIG.get('telegram_token', ''),
                    'telegram_chat_id': CONFIG.get('telegram_chat_id', ''),
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.debug(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration file: {e}")

    def _load_config_file(self):
        """Load configuration from data/config.json"""
        try:
            config_path = Path('data/config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)

                # Load strategy config
                if 'strategy' in data:
                    if 'assets' in data['strategy']:
                        CONFIG['assets'] = data['strategy']['assets']
                    if 'interval' in data['strategy']:
                        CONFIG['interval'] = data['strategy']['interval']
                    if 'llm_model' in data['strategy']:
                        CONFIG['llm_model'] = data['strategy']['llm_model']

                # Load API keys
                if 'api_keys' in data:
                    if 'taapi_api_key' in data['api_keys']:
                        CONFIG['taapi_api_key'] = data['api_keys']['taapi_api_key']
                    if 'hyperliquid_private_key' in data['api_keys']:
                        CONFIG['hyperliquid_private_key'] = data['api_keys']['hyperliquid_private_key']
                    if 'openrouter_api_key' in data['api_keys']:
                        CONFIG['openrouter_api_key'] = data['api_keys']['openrouter_api_key']

                # Load risk management
                if 'risk_management' in data:
                    if 'max_position_size' in data['risk_management']:
                        CONFIG['max_position_size'] = data['risk_management']['max_position_size']
                    if 'max_leverage' in data['risk_management']:
                        CONFIG['max_leverage'] = data['risk_management']['max_leverage']

                # Load notifications
                if 'notifications' in data:
                    if 'desktop_enabled' in data['notifications']:
                        CONFIG['desktop_notifications'] = data['notifications']['desktop_enabled']
                    if 'telegram_enabled' in data['notifications']:
                        CONFIG['telegram_notifications'] = data['notifications']['telegram_enabled']
                    if 'telegram_token' in data['notifications']:
                        CONFIG['telegram_token'] = data['notifications']['telegram_token']
                    if 'telegram_chat_id' in data['notifications']:
                        CONFIG['telegram_chat_id'] = data['notifications']['telegram_chat_id']

                self.logger.debug(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")

    async def test_api_connections(self) -> Dict[str, object]:
        """Test API connections to all services"""
        results: Dict[str, object] = {
            "TAAPI": False,
            "Hyperliquid": False,
            "OpenRouter": False,
            "Thalex": False,
        }
        errors: Dict[str, str] = {}

        try:
            # Test TAAPI
            taapi_key = (os.getenv("TAAPI_API_KEY") or CONFIG.get("taapi_api_key") or "").strip()
            if taapi_key and taapi_key != "your_taapi_key_here":
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(
                            f"https://api.taapi.io/ema?secret={taapi_key}&exchange=binance&symbol=BTC/USDT&interval=4h&period=14",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                results["TAAPI"] = True
                    except Exception as e:
                        self.logger.debug(f"TAAPI test failed: {e}")
                        errors["TAAPI"] = str(e)

            # Test Hyperliquid
            hl_key = (os.getenv("HYPERLIQUID_PRIVATE_KEY") or CONFIG.get("hyperliquid_private_key") or "").strip()
            if hl_key and hl_key != "your_private_key_here":
                try:
                    from src.backend.trading.hyperliquid_api import HyperliquidAPI
                    hl = HyperliquidAPI()
                    state = await hl.get_user_state()
                    if state:
                        results["Hyperliquid"] = True
                except Exception as e:
                    self.logger.debug(f"Hyperliquid test failed: {e}")
                    errors["Hyperliquid"] = str(e)

            # Test OpenRouter
            or_key = (os.getenv("OPENROUTER_API_KEY") or CONFIG.get("openrouter_api_key") or "").strip()
            if or_key and or_key != "your_openrouter_key_here":
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(
                            "https://openrouter.ai/api/v1/auth/key",
                            headers={"Authorization": f"Bearer {or_key}"},
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status in [200, 401]:
                                results["OpenRouter"] = True
                    except Exception as e:
                        self.logger.debug(f"OpenRouter test failed: {e}")
                        errors["OpenRouter"] = str(e)

            # Test Thalex
            thalex_key_id = (os.getenv("THALEX_KEY_ID") or CONFIG.get("thalex_key_id") or "").strip()
            thalex_pem_path = (os.getenv("THALEX_PRIVATE_KEY_PATH") or CONFIG.get("thalex_private_key_path") or "").strip()
            if thalex_key_id and thalex_pem_path:
                try:
                    from src.backend.trading.thalex_api import ThalexAPI
                    import asyncio as _asyncio

                    thalex = ThalexAPI()
                    await _asyncio.wait_for(thalex.connect(), timeout=12.0)
                    results["Thalex"] = True
                    await thalex.disconnect()
                except Exception as e:
                    self.logger.debug(f"Thalex test failed: {e}")
                    errors["Thalex"] = str(e)

        except Exception as e:
            self.logger.error(f"Error testing API connections: {e}")
            errors["_global"] = str(e)

        results["errors"] = errors

        return results
