"""
Market Data Page - Live market data and technical indicators
"""

import plotly.graph_objects as go
from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager


def create_market(bot_service: BotService, state_manager: StateManager):
    """Create market data page with live prices and technical indicators"""

    ui.label('Market Data').classes('text-3xl font-bold mb-4 text-white')

    # ===== ASSET SELECTOR =====
    with ui.row().classes('w-full items-center gap-4 mb-6'):
        ui.label('Select Asset:').classes('text-lg font-semibold text-white')

        # Get assets from bot config
        state = state_manager.get_state()
        configured_assets = bot_service.get_assets() if bot_service.is_running() else ['BTC', 'ETH', 'SOL']
        available_assets = configured_assets if configured_assets else ['BTC', 'ETH', 'SOL']

        asset_select = ui.select(
            label='Asset',
            options=available_assets,
            value=available_assets[0] if available_assets else 'BTC'
        ).classes('w-48')

        interval_select = ui.select(
            label='Interval',
            options=['1m', '5m', '15m', '1h', '4h', '1d'],
            value='5m'
        ).classes('w-32')

    # ===== PRICE CARDS =====
    with ui.grid(columns=4).classes('w-full gap-4 mb-6'):
        # Current Price Card
        with ui.card().classes('metric-card'):
            current_price_label = ui.label('$0.00').classes('text-4xl font-bold text-white')
            ui.label('Current Price').classes('text-sm text-gray-200 mt-2')

        # 24h Change Card
        with ui.card().classes('metric-card'):
            change_24h_label = ui.label('+0.00%').classes('text-4xl font-bold text-green-400')
            ui.label('24h Change').classes('text-sm text-gray-200 mt-2')

        # 24h Volume Card
        with ui.card().classes('metric-card'):
            volume_24h_label = ui.label('$0.00M').classes('text-4xl font-bold text-white')
            ui.label('24h Volume').classes('text-sm text-gray-200 mt-2')

        # Open Interest Card
        with ui.card().classes('metric-card'):
            open_interest_label = ui.label('$0.00M').classes('text-4xl font-bold text-white')
            ui.label('Open Interest').classes('text-sm text-gray-200 mt-2')

    # ===== PRICE CHART =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Price Chart').classes('text-xl font-bold text-white mb-2')

        # Candlestick chart
        price_chart = ui.plotly(go.Figure(
            data=[go.Candlestick(
                x=[],
                open=[],
                high=[],
                low=[],
                close=[],
                name='Price'
            )],
            layout=go.Layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=50, r=20, t=20, b=40),
                xaxis=dict(title='Time', showgrid=True, gridcolor='#374151'),
                yaxis=dict(title='Price ($)', showgrid=True, gridcolor='#374151'),
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='#e5e7eb'),
                showlegend=True
            )
        )).classes('w-full')

    # ===== TECHNICAL INDICATORS =====
    with ui.row().classes('w-full gap-4 mb-6'):
        # Left column - Trend Indicators
        with ui.card().classes('flex-1 p-4'):
            ui.label('Trend Indicators').classes('text-xl font-bold text-white mb-4')

            with ui.column().classes('gap-3 w-full'):
                # SMA 99
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('SMA 99 (5m)').classes('text-gray-300')
                    ema20_label = ui.label('$0.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('SMA 99 (HTF)').classes('text-gray-300')
                    ema50_label = ui.label('$0.00').classes('text-white font-semibold')

                ui.separator()

                # Keltner
                ui.label('Keltner 130 x 4').classes('text-lg font-bold text-white mt-2')
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Middle').classes('text-gray-300')
                    macd_line_label = ui.label('0.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Upper').classes('text-gray-300')
                    macd_signal_label = ui.label('0.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Lower').classes('text-gray-300')
                    macd_hist_label = ui.label('0.00').classes('text-green-400 font-semibold')

        # Right column - Momentum Indicators
        with ui.card().classes('flex-1 p-4'):
            ui.label('Momentum Indicators').classes('text-xl font-bold text-white mb-4')

            with ui.column().classes('gap-3 w-full'):
                # Anchored VWAP
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('AVWAP 2026').classes('text-gray-300')
                    rsi_label = ui.label('50.00').classes('text-white font-semibold')

                # Position Bar (price vs AVWAP)
                rsi_progress = ui.linear_progress(value=0.5, show_value=False).classes('w-full')

                ui.separator()

                # Long-term AVWAP
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('AVWAP Anchor').classes('text-gray-300')
                    atr_label = ui.label('$0.00').classes('text-white font-semibold')

                ui.separator()

                # Opening Range
                ui.label('Opening Range').classes('text-lg font-bold text-white mt-2')
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('High').classes('text-gray-300')
                    stoch_k_label = ui.label('50.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Low').classes('text-gray-300')
                    stoch_d_label = ui.label('50.00').classes('text-white font-semibold')

    # ===== INDICATOR CHART =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('RSI & MACD').classes('text-xl font-bold text-white mb-2')

        # Create subplot for RSI and MACD
        indicator_chart = ui.plotly(go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode='lines', name='RSI', line=dict(color='#f59e0b', width=2)),
                go.Scatter(x=[], y=[], mode='lines', name='MACD', line=dict(color='#3b82f6', width=2), yaxis='y2'),
            ],
            layout=go.Layout(
                template='plotly_dark',
                height=300,
                margin=dict(l=50, r=50, t=20, b=40),
                xaxis=dict(title='Time', showgrid=True, gridcolor='#374151'),
                yaxis=dict(title='RSI', showgrid=True, gridcolor='#374151', range=[0, 100]),
                yaxis2=dict(title='MACD', overlaying='y', side='right', showgrid=False),
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='#e5e7eb'),
                showlegend=True
            )
        )).classes('w-full')

    # ===== MARKET SENTIMENT =====
    with ui.card().classes('w-full p-4'):
        ui.label('Market Sentiment').classes('text-xl font-bold text-white mb-4')

        with ui.row().classes('w-full gap-6 items-center'):
            # Sentiment gauge
            with ui.column().classes('flex-1'):
                sentiment_label = ui.label('NEUTRAL').classes('text-3xl font-bold text-gray-400')
                sentiment_desc = ui.label('Waiting for clear signals').classes('text-sm text-gray-400 mt-2')

            # Signal indicators
            with ui.column().classes('flex-1'):
                with ui.row().classes('items-center gap-2 mb-2'):
                    trend_icon = ui.label('○').classes('text-2xl text-gray-400')
                    ui.label('Trend Signal').classes('text-gray-300')

                with ui.row().classes('items-center gap-2 mb-2'):
                    momentum_icon = ui.label('○').classes('text-2xl text-gray-400')
                    ui.label('Momentum Signal').classes('text-gray-300')

                with ui.row().classes('items-center gap-2'):
                    volume_icon = ui.label('○').classes('text-2xl text-gray-400')
                    ui.label('Volume Signal').classes('text-gray-300')

    # ===== HEDGE STATUS =====
    with ui.card().classes('w-full p-4 mt-6'):
        ui.label('Delta Hedge Status').classes('text-xl font-bold text-white mb-4')
        with ui.row().classes('w-full gap-6 items-center'):
            with ui.column().classes('flex-1 gap-2'):
                hedge_status_label = ui.label('IDLE').classes('text-3xl font-bold text-gray-400')
                hedge_status_desc = ui.label('No hedge data yet').classes('text-sm text-gray-400')
            with ui.column().classes('flex-1 gap-2'):
                hedge_target_label = ui.label('Target: --').classes('text-gray-300')
                hedge_current_label = ui.label('Current: --').classes('text-gray-300')
                hedge_residual_label = ui.label('Residual: --').classes('text-gray-300')
                hedge_last_label = ui.label('Last rebalance: --').classes('text-gray-400 text-sm')

    # ===== AUTO-REFRESH LOGIC =====
    async def update_market_data():
        """Update market data and indicators from real bot data"""
        state = state_manager.get_state()
        selected_asset = asset_select.value
        hedge_metric = next(
            (m for m in (getattr(state, 'hedge_metrics', []) or []) if m.get('underlying') == selected_asset),
            None,
        )
        degraded_reason = None
        hedge_status = getattr(state, 'hedge_status', {}) or {}
        degraded_map = hedge_status.get('degraded_underlyings', {}) or {}
        if selected_asset in degraded_map:
            degraded_reason = degraded_map[selected_asset]

        # Get market data for selected asset from bot state
        market_data = None
        if state.market_data:
            # market_data can be either dict with asset keys or list of dicts
            if isinstance(state.market_data, dict):
                market_data = state.market_data.get(selected_asset)
            elif isinstance(state.market_data, list):
                market_data = next((m for m in state.market_data if m.get('asset') == selected_asset), None)

        if not market_data:
            # No data available yet
            current_price_label.set_text('Loading...')
            change_24h_label.set_text('--')
            volume_24h_label.set_text('--')
            open_interest_label.set_text('--')
            hedge_status_label.set_text('NO DATA')
            hedge_status_desc.set_text('Waiting for market and hedge data from bot...')
            return

        # Update price cards with real data
        current_price = market_data.get('price') or market_data.get('current_price', 0)
        current_price_label.set_text(f'${current_price:,.2f}')
        
        # 24h change (mock for now - need to calculate from price history)
        change_24h_label.set_text('+0.00%')
        change_24h_label.classes('text-4xl font-bold text-gray-400')
        
        # Volume and OI
        open_interest = market_data.get('open_interest', 0)
        if open_interest:
            open_interest_label.set_text(f'${open_interest/1e6:.1f}M')
        else:
            open_interest_label.set_text('--')
        
        volume_24h_label.set_text('--')  # Not available in current data
        
        # Update indicators - use 5m data from market_data
        intraday = market_data.get('intraday', {})
        long_term = market_data.get('long_term', {})
        
        sma99_5m = intraday.get('sma99')
        sma99_lt = long_term.get('sma99')
        avwap = intraday.get('avwap') or long_term.get('avwap')
        opening_range = intraday.get('opening_range', {})
        keltner = intraday.get('keltner', {})
        
        if sma99_5m:
            ema20_label.set_text(f'${sma99_5m:,.2f}')
        else:
            ema20_label.set_text('--')
        
        if sma99_lt:
            ema50_label.set_text(f'${sma99_lt:,.2f}')
        else:
            ema50_label.set_text('--')
        
        middle = keltner.get('middle')
        upper = keltner.get('upper')
        lower = keltner.get('lower')
        macd_line_label.set_text(f'${middle:,.2f}' if middle else '--')
        macd_signal_label.set_text(f'${upper:,.2f}' if upper else '--')
        macd_hist_label.set_text(f'${lower:,.2f}' if lower else '--')

        if avwap:
            rsi_label.set_text(f'${avwap:,.2f}')
            if current_price and current_price > avwap:
                rsi_progress.set_value(0.75)
            elif current_price and current_price < avwap:
                rsi_progress.set_value(0.25)
            else:
                rsi_progress.set_value(0.5)
        else:
            rsi_label.set_text('--')
            rsi_progress.set_value(0.5)

        if avwap:
            atr_label.set_text(f'${avwap:,.2f}')
        else:
            atr_label.set_text('--')

        or_high = opening_range.get('high')
        or_low = opening_range.get('low')
        stoch_k_label.set_text(f'${or_high:,.2f}' if or_high else '--')
        stoch_d_label.set_text(f'${or_low:,.2f}' if or_low else '--')
        
        # Update sentiment based on indicators
        opening_range_position = opening_range.get('position')
        if sma99_5m and avwap and current_price:
            if current_price > sma99_5m and current_price > avwap and opening_range_position == 'above':
                sentiment_label.set_text('BULLISH')
                sentiment_label.classes('text-3xl font-bold text-green-400')
                sentiment_desc.set_text('Price above SMA99 and AVWAP, holding above opening range')
                trend_icon.set_text('●')
                trend_icon.classes('text-2xl text-green-400')
            elif current_price < sma99_5m and current_price < avwap and opening_range_position == 'below':
                sentiment_label.set_text('BEARISH')
                sentiment_label.classes('text-3xl font-bold text-red-400')
                sentiment_desc.set_text('Price below SMA99 and AVWAP, holding below opening range')
                trend_icon.set_text('●')
                trend_icon.classes('text-2xl text-red-400')
            else:
                sentiment_label.set_text('NEUTRAL')
                sentiment_label.classes('text-3xl font-bold text-gray-400')
                sentiment_desc.set_text('Mixed signals, waiting for clear direction')
                trend_icon.set_text('○')
                trend_icon.classes('text-2xl text-gray-400')
        else:
            sentiment_label.set_text('NO DATA')
            sentiment_label.classes('text-3xl font-bold text-gray-500')
            sentiment_desc.set_text('Waiting for market data from bot...')
        
        momentum_icon.set_text('○')
        momentum_icon.classes('text-2xl text-gray-400')
        volume_icon.set_text('○')
        volume_icon.classes('text-2xl text-gray-400')

        if hedge_metric:
            hedge_status_label.set_text(str(hedge_metric.get('status', 'unknown')).upper())
            if hedge_metric.get('degraded'):
                hedge_status_label.classes('text-3xl font-bold text-red-400')
                hedge_status_desc.set_text(degraded_reason or hedge_metric.get('degraded_reason') or 'Live greeks unavailable')
            else:
                hedge_status_label.classes('text-3xl font-bold text-green-400')
                hedge_status_desc.set_text(
                    f"{hedge_metric.get('open_option_positions', 0)} option leg(s), {len(hedge_metric.get('subscribed_instruments', []))} ticker subscriptions"
                )
            target = hedge_metric.get('target_perp_delta')
            current = hedge_metric.get('current_perp_delta')
            residual = hedge_metric.get('residual_delta')
            hedge_target_label.set_text(f'Target: {target:+.4f}' if target is not None else 'Target: --')
            hedge_current_label.set_text(f'Current: {current:+.4f}' if current is not None else 'Current: --')
            hedge_residual_label.set_text(f'Residual: {residual:+.4f}' if residual is not None else 'Residual: --')
            last_side = hedge_metric.get('last_rebalance_side') or 'none'
            last_size = hedge_metric.get('last_rebalance_size') or 0.0
            hedge_last_label.set_text(f'Last rebalance: {last_side} {last_size:.4f}')
        else:
            if degraded_reason:
                hedge_status_label.set_text('DEGRADED')
                hedge_status_label.classes('text-3xl font-bold text-red-400')
                hedge_status_desc.set_text(degraded_reason)
            else:
                hedge_status_label.set_text('IDLE')
                hedge_status_label.classes('text-3xl font-bold text-gray-400')
                hedge_status_desc.set_text('No active hedge for this asset')
            hedge_target_label.set_text('Target: --')
            hedge_current_label.set_text('Current: --')
            hedge_residual_label.set_text('Residual: --')
            hedge_last_label.set_text('Last rebalance: --')

    # Auto-refresh every 5 seconds
    ui.timer(5.0, update_market_data)

    # Refresh on asset/interval change
    asset_select.on('update:model-value', lambda: update_market_data())
    interval_select.on('update:model-value', lambda: update_market_data())
