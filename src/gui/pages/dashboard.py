"""
Dashboard Page - Main dashboard with metrics and charts
"""

import asyncio
import time

import plotly.graph_objects as go
from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager
from src.gui.services.ui_utils import is_ui_alive


def create_dashboard(bot_service: BotService, state_manager: StateManager):
    """Create dashboard page with real-time metrics, charts, and controls"""

    def _ui_ok() -> bool:
        """Helper to check if this page is still the active one."""
        return is_ui_alive(status_indicator)

    def _format_compact_number(value: float | int | None, prefix: str = '') -> str:
        if value is None:
            return '--'
        amount = float(value)
        for threshold, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
            if abs(amount) >= threshold:
                return f'{prefix}{amount / threshold:,.2f}{suffix}'
        if amount.is_integer():
            return f'{prefix}{amount:,.0f}'
        return f'{prefix}{amount:,.2f}'

    def _format_change(current_price: float | None, prev_day_price: float | None) -> tuple[str, str]:
        if current_price is None or prev_day_price is None or prev_day_price <= 0:
            return '--', 'text-gray-400'
        change_pct = ((current_price - prev_day_price) / prev_day_price) * 100
        if change_pct > 0:
            return f'+{change_pct:.2f}%', 'text-green-400'
        if change_pct < 0:
            return f'{change_pct:.2f}%', 'text-red-400'
        return '0.00%', 'text-gray-400'

    def _set_status_indicator_tone(tone: str):
        status_indicator.classes(remove='text-gray-400 text-green-500 text-red-500')
        status_indicator.classes(add=tone)

    ui.label('Dashboard').classes('text-3xl font-bold mb-4 text-white')

    # ===== METRICS CARDS =====
    with ui.grid(columns=5).classes('w-full gap-4 mb-6'):
        # Card 1: Total Balance
        with ui.card().classes('metric-card'):
            balance_value = ui.label('$0.00').classes('text-4xl font-bold text-white')
            ui.label('Total Balance').classes('text-sm text-gray-200 mt-2')
            balance_breakdown_value = ui.label('HL: -- | Thalex: --').classes('text-xs text-gray-300 mt-1')

        # Card 2: Total Return
        with ui.card().classes('metric-card'):
            return_value = ui.label('+0.00%').classes('text-4xl font-bold text-white')
            ui.label('Total Return').classes('text-sm text-gray-200 mt-2')

        # Card 3: Sharpe Ratio
        with ui.card().classes('metric-card'):
            sharpe_value = ui.label('0.00').classes('text-4xl font-bold text-white')
            ui.label('Sharpe Ratio').classes('text-sm text-gray-200 mt-2')

        # Card 4: Active Positions
        with ui.card().classes('metric-card'):
            positions_value = ui.label('0').classes('text-4xl font-bold text-white')
            ui.label('Active Positions').classes('text-sm text-gray-200 mt-2')
            positions_breakdown_value = ui.label('AI: 0 | External: 0').classes('text-xs text-gray-300 mt-1')

        # Card 5: Hedge Health
        with ui.card().classes('metric-card'):
            hedge_health_value = ui.label('IDLE').classes('text-4xl font-bold text-white')
            hedge_health_summary = ui.label('No active hedge book').classes('text-sm text-gray-200 mt-2')

    # ===== CHARTS ROW =====
    with ui.row().classes('w-full gap-4 mb-6'):
        # Equity Curve Chart (left half)
        with ui.card().classes('flex-1 p-4'):
            ui.label('Portfolio Value (3 months)').classes('text-xl font-bold text-white mb-2')

            equity_chart = ui.plotly(go.Figure(
                data=[go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='Value',
                    line=dict(color='#667eea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102,126,234,0.15)',
                )],
                layout=go.Layout(
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=50, r=20, t=20, b=40),
                    xaxis=dict(title='Time', showgrid=True, gridcolor='#374151',
                               type='date', rangeslider=dict(visible=False)),
                    yaxis=dict(title='Value ($)', showgrid=True, gridcolor='#374151'),
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb')
                )
            )).classes('w-full')

        # Asset Allocation Pie Chart (right half)
        with ui.card().classes('flex-1 p-4'):
            ui.label('Asset Allocation').classes('text-xl font-bold text-white mb-2')

            allocation_chart = ui.plotly(go.Figure(
                data=[go.Pie(
                    labels=[],
                    values=[],
                    hole=0.4,
                    marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
                )],
                layout=go.Layout(
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb'),
                    showlegend=True,
                    legend=dict(orientation='v', x=1, y=0.5)
                )
            )).classes('w-full')

    # ===== MARKET DATA =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Market Data').classes('text-xl font-bold text-white mb-2')
        with ui.row().classes('gap-2 text-xs text-gray-400 mb-3 wrap'): 
            ui.badge('Price: Hyperliquid mids').classes('bg-gray-800 text-gray-200')
            ui.badge('24h metrics: Hyperliquid metaAndAssetCtxs').classes('bg-gray-800 text-gray-200')
            ui.badge('Indicators: TAAPI').classes('bg-gray-800 text-gray-200')
        market_data_container = ui.column().classes('w-full gap-4')
        
        with market_data_container:
            ui.label('No market data available').classes('text-gray-400 text-center py-4')

    # ===== HEDGE MONITOR =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Delta Hedge Monitor').classes('text-xl font-bold text-white mb-2')
        hedge_alert_container = ui.column().classes('w-full gap-2 mb-4')
        hedge_metrics_container = ui.column().classes('w-full gap-4')

        with hedge_alert_container:
            ui.label('No hedge alerts').classes('text-gray-400 text-sm')

        with hedge_metrics_container:
            ui.label('No hedge metrics available').classes('text-gray-400 text-center py-4')

    # ===== ACTIVITY FEED =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Recent Activity').classes('text-xl font-bold text-white mb-2')

        activity_log = ui.log(max_lines=10).classes('w-full h-48 bg-gray-900 text-gray-300 p-4 rounded')
        activity_log.push('Bot initialized. Waiting to start...')

    # ===== CONTROL PANEL =====
    with ui.card().classes('w-full p-4'):
        ui.label('Bot Controls').classes('text-xl font-bold text-white mb-4')

        with ui.row().classes('gap-4 items-center'):
            # Refresh Data Button (for manual mode)
            refresh_data_btn = ui.button('🔄 Refresh Data', on_click=lambda: refresh_market_data())
            refresh_data_btn.classes('bg-blue-600 hover:bg-blue-700 text-white px-6 py-3')
            refresh_data_loading = ui.label('').classes('text-sm text-blue-400 ml-2')

            # Start Button
            start_btn = ui.button('▶ Start Bot', on_click=lambda: start_bot())
            start_btn.classes('bg-green-600 hover:bg-green-700 text-white px-6 py-3')

            # Stop Button
            stop_btn = ui.button('⏹ Stop Bot', on_click=lambda: stop_bot())
            stop_btn.classes('bg-red-600 hover:bg-red-700 text-white px-6 py-3')
            stop_btn.props('disable')  # Initially disabled

            hedge_toggle_btn = ui.button('⏸ Delta Hedge OFF', on_click=lambda: toggle_delta_hedge())
            hedge_toggle_btn.classes('bg-yellow-600 hover:bg-yellow-700 text-white px-6 py-3')

            # Status indicator
            status_indicator = ui.label('⚫ Stopped').classes('text-lg font-bold ml-4')

        # Last refresh timestamp
        with ui.row().classes('gap-4 items-center mt-4'):
            last_refresh_label = ui.label('Last refreshed: Never').classes('text-sm text-gray-400')
            refresh_timer_label = ui.label('').classes('text-xs text-gray-500')

    # ===== CONTROL FUNCTIONS =====

    async def start_bot():
        """Start the trading bot"""
        # Snapshot alive-state up front so we know whether any element writes
        # after the await are still safe to issue. If the user navigated away
        # during bot_service.start(), status_indicator is gone and any .text
        # assignment raises RuntimeError before we can reach the _ui_ok guard.
        if _ui_ok():
            status_indicator.text = '🟡 Starting...'
            activity_log.push('Starting bot...')

        try:
            await bot_service.start()
        except Exception as e:
            if not _ui_ok():
                return
            status_indicator.text = '🔴 Error'
            _set_status_indicator_tone('text-red-500')
            activity_log.push(f'❌ Error starting bot: {str(e)}')
            ui.notify(f'Failed to start: {str(e)}', type='negative')
            return

        if not _ui_ok():
            return
        status_indicator.text = '🟢 Running'
        _set_status_indicator_tone('text-green-500')
        start_btn.props('disable')
        stop_btn.props(remove='disable')
        activity_log.push('✅ Bot started successfully!')
        ui.notify('Bot started!', type='positive')

    async def stop_bot():
        """Stop the trading bot"""
        if _ui_ok():
            status_indicator.text = '🟡 Stopping...'
            activity_log.push('Stopping bot...')

        try:
            await bot_service.stop()
        except Exception as e:
            if _ui_ok():
                activity_log.push(f'❌ Error stopping bot: {str(e)}')
                ui.notify(f'Failed to stop: {str(e)}', type='negative')
            return

        if not _ui_ok():
            return
        status_indicator.text = '⚫ Stopped'
        _set_status_indicator_tone('text-gray-400')
        start_btn.props(remove='disable')
        stop_btn.props('disable')
        activity_log.push('✅ Bot stopped successfully!')
        ui.notify('Bot stopped!', type='info')

    async def toggle_delta_hedge():
        """Enable or disable live delta hedging from the dashboard."""
        if not bot_service.supports_delta_hedge():
            if _ui_ok():
                ui.notify('Delta hedge is unavailable until Thalex is configured', type='warning')
            return

        desired_state = not bot_service.is_delta_hedge_enabled()
        hedge_toggle_btn.enabled = False
        try:
            success = await bot_service.set_delta_hedge_enabled(desired_state)
            if not _ui_ok():
                return
            if success:
                ui.notify(
                    'Delta hedge enabled' if desired_state else 'Delta hedge disabled',
                    type='positive' if desired_state else 'info',
                )
                await update_dashboard()
            else:
                ui.notify('Unable to change delta hedge state', type='negative')
        except Exception as e:
            if _ui_ok():
                ui.notify(f'Failed to update delta hedge: {str(e)}', type='negative')
        finally:
            if _ui_ok():
                hedge_toggle_btn.enabled = True

    # ===== AUTO-REFRESH FUNCTIONS =====

    last_refresh_time = None
    refresh_seconds_ago = 0
    displayed_events = set()

    async def refresh_market_data():
        """Refresh market data from Hyperliquid without starting bot"""
        nonlocal last_refresh_time, refresh_seconds_ago

        try:
            refresh_data_btn.enabled = False
            refresh_data_loading.text = '⏳ Fetching...'
            activity_log.push('📊 Refreshing market data...')

            # Call bot service to refresh data
            success = await bot_service.refresh_market_data(include_indicators=True)

            if not _ui_ok():
                return

            if success:
                last_refresh_time = time.time()
                refresh_data_loading.text = '✅ Done'
                activity_log.push('✅ Market data refreshed successfully!')
                ui.notify('Market data refreshed!', type='positive')
                await update_dashboard()
            else:
                refresh_data_loading.text = '❌ Failed'
                activity_log.push('❌ Failed to refresh market data')
                ui.notify('Failed to refresh market data', type='negative')

        except RuntimeError:
            return  # client navigated away
        except Exception as e:
            if not _ui_ok():
                return
            activity_log.push(f'❌ Refresh error: {str(e)}')
            ui.notify(f'Error: {str(e)}', type='negative')
            refresh_data_loading.text = '❌ Error'
        finally:
            if _ui_ok():
                refresh_data_btn.enabled = True
                await asyncio.sleep(2.0)
                if _ui_ok():
                    refresh_data_loading.text = ''

    async def update_dashboard():
        """Update all dashboard components with latest data"""
        if not _ui_ok():
            return
        nonlocal refresh_seconds_ago

        try:
            state = state_manager.get_state()

            # Update metrics cards
            balance_value.text = f'${state.balance:,.2f}'
            balance_breakdown = getattr(state, 'balance_breakdown', {}) or {}
            balance_breakdown_value.text = (
                f"HL: ${balance_breakdown.get('hyperliquid', 0.0):,.2f} | "
                f"Thalex: ${balance_breakdown.get('thalex', 0.0):,.2f}"
            )

            # Return with color coding
            return_pct = state.total_return_pct
            return_value.text = f'{return_pct:+.2f}%'
            if return_pct >= 0:
                return_value.classes(remove='text-red-500', add='text-green-500')
            else:
                return_value.classes(remove='text-green-500', add='text-red-500')

            sharpe_value.text = f'{state.sharpe_ratio:.2f}'
            positions = state.positions or []
            positions_value.text = str(len(positions))
            ai_positions = sum(1 for position in positions if position.get('opened_by') == 'AI')
            positions_breakdown_value.text = f'AI: {ai_positions} | External: {len(positions) - ai_positions}'

            hedge_status = getattr(state, 'hedge_status', {}) or {}
            hedge_metrics = getattr(state, 'hedge_metrics', []) or []
            hedge_health = str(hedge_status.get('health', 'idle')).upper()
            hedge_enabled = bool(hedge_status.get('enabled', bot_service.is_delta_hedge_enabled()))
            hedge_available = bool(hedge_status.get('available', bot_service.supports_delta_hedge()))
            hedge_health_value.text = hedge_health
            tracked = hedge_status.get('tracked_underlyings', 0)
            active = hedge_status.get('active_underlyings', 0)
            degraded = hedge_status.get('degraded_underlyings', {}) or {}
            state_error = hedge_status.get('state_error')
            hedge_health_summary.text = f'{active} active / {tracked} tracked'
            if not hedge_available:
                hedge_health_value.text = 'UNAVAILABLE'
                hedge_health_value.classes(remove='text-green-500 text-red-500 text-yellow-400', add='text-gray-400')
                hedge_health_summary.text = 'Delta hedge unavailable'
            elif not hedge_enabled or hedge_health == 'DISABLED':
                hedge_health_value.text = 'DISABLED'
                hedge_health_value.classes(remove='text-green-500 text-red-500 text-gray-400', add='text-yellow-400')
                hedge_health_summary.text = 'Delta hedge paused from GUI'
            elif hedge_health == 'HEALTHY':
                hedge_health_value.classes(remove='text-red-500 text-yellow-400 text-gray-400', add='text-green-500')
            elif hedge_health == 'DEGRADED':
                hedge_health_value.classes(remove='text-green-500 text-gray-400', add='text-red-500')
                hedge_health_summary.text = f'{len(degraded)} degraded underlying(s)'
            elif hedge_health == 'UNAVAILABLE':
                hedge_health_value.classes(remove='text-green-500 text-red-500', add='text-gray-400')
                hedge_health_summary.text = state_error or 'Hedge state unavailable'
            else:
                hedge_health_value.classes(remove='text-green-500 text-red-500', add='text-gray-400')

            # Update equity curve chart — merge 3-month DB history with in-memory session data
            db_curve = bot_service.get_equity_curve(days=90)
            session_history = bot_service.get_equity_history(limit=500)
            merged_times = []
            merged_values = []
            if db_curve:
                merged_times.extend(d['timestamp'] for d in db_curve if d.get('total_value'))
                merged_values.extend(d['total_value'] for d in db_curve if d.get('total_value'))
            if session_history:
                db_last_ts = merged_times[-1] if merged_times else ''
                for d in session_history:
                    ts = d.get('time', '')
                    if ts > db_last_ts and d.get('value'):
                        merged_times.append(ts)
                        merged_values.append(d['value'])
            if merged_times:
                equity_chart.figure.data[0].x = merged_times
                equity_chart.figure.data[0].y = merged_values
                equity_chart.update()
            else:
                # Clear any stale points left over from a prior session;
                # otherwise the chart keeps showing yesterday's equity.
                equity_chart.figure.data[0].x = []
                equity_chart.figure.data[0].y = []
                equity_chart.update()

            # Update asset allocation chart
            if positions:
                labels = [p['symbol'] for p in positions]
                values = [abs(p['quantity'] * p['entry_price']) for p in positions]

                allocation_chart.figure.data[0].labels = labels
                allocation_chart.figure.data[0].values = values
                allocation_chart.update()

            # Update market data
            market_data = getattr(state, 'market_data', None)
            market_data_container.clear()
            
            if market_data and isinstance(market_data, list) and len(market_data) > 0:
                with market_data_container:
                    with ui.grid(columns=len(market_data)).classes('w-full gap-4'):
                        for asset_data in market_data:
                            asset = asset_data.get('asset', 'N/A')
                            price = asset_data.get('current_price')
                            prev_day_price = asset_data.get('prev_day_price')
                            volume_24h = asset_data.get('volume_24h')
                            open_interest = asset_data.get('open_interest')
                            change_text, change_tone = _format_change(price, prev_day_price)
                            
                            # Intraday data
                            intraday = asset_data.get('intraday', {})
                            sma99 = intraday.get('sma99', 0)
                            avwap = intraday.get('avwap', 0)
                            
                            # Long-term data
                            lt = asset_data.get('long_term', {})
                            lt_sma99 = lt.get('sma99', 0)
                            lt_avwap = lt.get('avwap', 0)
                            
                            with ui.card().classes('p-4 bg-gradient-to-br from-gray-700 to-gray-800'):
                                with ui.row().classes('w-full items-center justify-between mb-2'):
                                    ui.label(asset).classes('text-2xl font-bold text-white')
                                    ui.badge('Hyperliquid + TAAPI').classes('bg-gray-900 text-gray-200')
                                ui.label(f'${price:,.2f}' if price is not None else '--').classes('text-xl text-green-400 mb-2')
                                with ui.grid(columns=3).classes('w-full gap-2 mb-3'):
                                    with ui.card().classes('p-2 bg-gray-900/40'):
                                        ui.label('24h Change').classes('text-xs text-gray-400')
                                        ui.label(change_text).classes(f'text-sm font-bold {change_tone}')
                                    with ui.card().classes('p-2 bg-gray-900/40'):
                                        ui.label('24h Volume').classes('text-xs text-gray-400')
                                        ui.label(_format_compact_number(volume_24h, prefix='$')).classes('text-sm font-bold text-white')
                                    with ui.card().classes('p-2 bg-gray-900/40'):
                                        ui.label('Open Interest').classes('text-xs text-gray-400')
                                        ui.label(_format_compact_number(open_interest)).classes('text-sm font-bold text-white')
                                
                                with ui.column().classes('gap-1 text-sm'):
                                    ui.label(f'SMA99 (5m): {sma99:.2f}' if sma99 else 'SMA99 (5m): N/A').classes('text-gray-300')
                                    ui.label(f'AVWAP 2026: {avwap:.2f}' if avwap else 'AVWAP 2026: N/A').classes('text-gray-300')
                                    ui.separator()
                                    ui.label(f'SMA99 ({lt.get("interval", "HTF")}): {lt_sma99:.2f}' if lt_sma99 else f'SMA99 ({lt.get("interval", "HTF")}): N/A').classes('text-gray-400')
                                    ui.label(f'AVWAP 2026: {lt_avwap:.2f}' if lt_avwap else 'AVWAP 2026: N/A').classes('text-gray-400')
            else:
                with market_data_container:
                    ui.label('No market data available').classes('text-gray-400 text-center py-4')

            hedge_alert_container.clear()
            if state_error:
                with hedge_alert_container:
                    with ui.card().classes('w-full bg-yellow-900/40 border border-yellow-700 p-3'):
                        ui.label('Thalex unavailable').classes('text-yellow-300 font-semibold')
                        ui.label(str(state_error)).classes('text-sm text-yellow-100')
            elif degraded:
                with hedge_alert_container:
                    for underlying, reason in degraded.items():
                        with ui.card().classes('w-full bg-red-900/40 border border-red-700 p-3'):
                            ui.label(f'{underlying} hedge degraded').classes('text-red-300 font-semibold')
                            ui.label(reason).classes('text-sm text-red-200')
            else:
                with hedge_alert_container:
                    ui.label('No hedge alerts').classes('text-gray-400 text-sm')

            hedge_metrics_container.clear()
            if hedge_metrics:
                with hedge_metrics_container:
                    with ui.grid(columns=max(1, len(hedge_metrics))).classes('w-full gap-4'):
                        for metric in hedge_metrics:
                            underlying = metric.get('underlying', 'N/A')
                            status = str(metric.get('status', 'unknown')).upper()
                            residual = metric.get('residual_delta')
                            target = metric.get('target_perp_delta')
                            current = metric.get('current_perp_delta')
                            threshold = metric.get('threshold')
                            last_side = metric.get('last_rebalance_side') or 'none'
                            last_size = metric.get('last_rebalance_size') or 0.0
                            with ui.card().classes('p-4 bg-gradient-to-br from-gray-700 to-gray-800'):
                                ui.label(underlying).classes('text-2xl font-bold text-white mb-2')
                                ui.label(status).classes('text-sm font-semibold text-cyan-300 mb-3')
                                with ui.column().classes('gap-1 text-sm'):
                                    ui.label(f'Net option delta: {metric.get("net_option_delta", 0.0):+.4f}').classes('text-gray-300')
                                    ui.label(f'Target perp delta: {target:+.4f}' if target is not None else 'Target perp delta: N/A').classes('text-gray-300')
                                    ui.label(f'Current perp delta: {current:+.4f}' if current is not None else 'Current perp delta: N/A').classes('text-gray-300')
                                    ui.label(f'Residual delta: {residual:+.4f}' if residual is not None else 'Residual delta: N/A').classes('text-gray-300')
                                    ui.label(f'Threshold: {threshold:.4f}' if threshold is not None else 'Threshold: N/A').classes('text-gray-400')
                                    ui.label(f'Open options: {metric.get("open_option_positions", 0)}').classes('text-gray-400')
                                    ui.label(f'Last rebalance: {last_side} {last_size:.4f}').classes('text-gray-400')
            else:
                with hedge_metrics_container:
                    ui.label('No hedge metrics available').classes('text-gray-400 text-center py-4')

            # Update activity log with recent events
            recent_events = bot_service.get_recent_events(limit=5)
            current_event_keys = []
            for event in recent_events[-5:]:  # Last 5 only
                event_key = (event.get('time'), event.get('message'), event.get('level'))
                current_event_keys.append(event_key)
                if event_key not in displayed_events:
                    activity_log.push(f"[{event['time']}] {event['message']}")
                    displayed_events.add(event_key)
            displayed_events.intersection_update(current_event_keys)

            # Update button states based on bot status
            if hedge_available:
                hedge_toggle_btn.props(remove='disable')
                if hedge_enabled:
                    hedge_toggle_btn.text = '🛡 Delta Hedge ON'
                    hedge_toggle_btn.classes(remove='bg-yellow-600 hover:bg-yellow-700 bg-gray-600', add='bg-green-600 hover:bg-green-700')
                else:
                    hedge_toggle_btn.text = '⏸ Delta Hedge OFF'
                    hedge_toggle_btn.classes(remove='bg-green-600 hover:bg-green-700 bg-gray-600', add='bg-yellow-600 hover:bg-yellow-700')
            else:
                hedge_toggle_btn.text = 'Delta Hedge Unavailable'
                hedge_toggle_btn.classes(remove='bg-green-600 hover:bg-green-700 bg-yellow-600 hover:bg-yellow-700', add='bg-gray-600')
                hedge_toggle_btn.props('disable')

            if state.is_running:
                status_indicator.text = '🟢 Running'
                _set_status_indicator_tone('text-green-500')
                start_btn.props('disable')
                stop_btn.props(remove='disable')
            else:
                status_indicator.text = '⚫ Stopped'
                _set_status_indicator_tone('text-gray-400')
                start_btn.props(remove='disable')
                stop_btn.props('disable')

            if state.error:
                status_indicator.text = '🔴 Error'
                _set_status_indicator_tone('text-red-500')
                activity_log.push(f'Error: {state.error}')

            # Update refresh timestamp
            if last_refresh_time:
                refresh_seconds_ago = int(time.time() - last_refresh_time)
                if refresh_seconds_ago < 60:
                    last_refresh_label.text = f'Last refreshed: {refresh_seconds_ago} seconds ago'
                else:
                    minutes = refresh_seconds_ago // 60
                    last_refresh_label.text = f'Last refreshed: {minutes} minutes ago'
                refresh_timer_label.text = '(auto-updating)'
            else:
                last_refresh_label.text = 'Last refreshed: Never'
                refresh_timer_label.text = '(click Refresh Data to fetch data)'

        except Exception as e:
            activity_log.push(f'Dashboard update error: {str(e)}')

    # ===== AUTO-REFRESH TIMER =====
    # Update dashboard every 3 seconds
    ui.timer(3.0, update_dashboard)

    # Initial update (call immediately, but don't await in sync context)
    # The timer will handle subsequent updates
