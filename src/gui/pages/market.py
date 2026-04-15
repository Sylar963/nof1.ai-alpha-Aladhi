"""
Market Data Page - Live market data and technical indicators
"""

import plotly.graph_objects as go
from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager
from src.gui.services.ui_utils import is_ui_alive


def create_market(bot_service: BotService, state_manager: StateManager):
    """Create market data page with live prices and technical indicators"""

    def _ui_ok():
        return is_ui_alive(asset_select)

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

    def _set_change_label(value: float | None):
        change_24h_label.classes(remove='text-green-400 text-red-400 text-gray-400')
        if value is None:
            change_24h_label.set_text('--')
            change_24h_label.classes(add='text-gray-400')
        elif value > 0:
            change_24h_label.set_text(f'+{value:.2f}%')
            change_24h_label.classes(add='text-green-400')
        elif value < 0:
            change_24h_label.set_text(f'{value:.2f}%')
            change_24h_label.classes(add='text-red-400')
        else:
            change_24h_label.set_text('0.00%')
            change_24h_label.classes(add='text-gray-400')

    def _set_status_tone(label, tone: str):
        label.classes(remove='text-green-400 text-red-400 text-gray-400 text-gray-500 text-yellow-400')
        label.classes(add=tone)

    ui.label('Market Data').classes('text-3xl font-bold mb-4 text-white')

    # ===== ASSET SELECTOR =====
    with ui.row().classes('w-full items-center gap-4 mb-6'):
        ui.label('Select Asset:').classes('text-lg font-semibold text-white')

        # Get assets from bot config
        configured_assets = bot_service.get_assets()
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
            ui.label('Source: Hyperliquid mids').classes('text-xs text-gray-200/80 mt-1')

        # 24h Change Card
        with ui.card().classes('metric-card'):
            change_24h_label = ui.label('+0.00%').classes('text-4xl font-bold text-green-400')
            ui.label('24h Change').classes('text-sm text-gray-200 mt-2')
            ui.label('Source: Hyperliquid prevDayPx').classes('text-xs text-gray-200/80 mt-1')

        # 24h Volume Card
        with ui.card().classes('metric-card'):
            volume_24h_label = ui.label('$0.00M').classes('text-4xl font-bold text-white')
            ui.label('24h Volume').classes('text-sm text-gray-200 mt-2')
            ui.label('Source: Hyperliquid dayNtlVlm').classes('text-xs text-gray-200/80 mt-1')

        # Open Interest Card
        with ui.card().classes('metric-card'):
            open_interest_label = ui.label('0').classes('text-4xl font-bold text-white')
            ui.label('Open Interest').classes('text-sm text-gray-200 mt-2')
            ui.label('Source: Hyperliquid openInterest').classes('text-xs text-gray-200/80 mt-1')

    # ===== PRICE CHART =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Price Chart').classes('text-xl font-bold text-white mb-2')
        ui.label('Indicators use TAAPI. Price candles fall back to Hyperliquid mids when OHLC is unavailable.').classes('text-xs text-gray-400 mb-3')

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
                    keltner_middle_label = ui.label('0.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Upper').classes('text-gray-300')
                    keltner_upper_label = ui.label('0.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Lower').classes('text-gray-300')
                    keltner_lower_label = ui.label('0.00').classes('text-green-400 font-semibold')

        # Right column - Momentum Indicators
        with ui.card().classes('flex-1 p-4'):
            ui.label('Momentum Indicators').classes('text-xl font-bold text-white mb-4')

            with ui.column().classes('gap-3 w-full'):
                # Anchored VWAP
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('AVWAP 2026').classes('text-gray-300')
                    avwap_label = ui.label('50.00').classes('text-white font-semibold')

                # Position Bar (price vs AVWAP)
                avwap_progress = ui.linear_progress(value=0.5, show_value=False).classes('w-full')

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
                    opening_range_high_label = ui.label('50.00').classes('text-white font-semibold')

                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Low').classes('text-gray-300')
                    opening_range_low_label = ui.label('50.00').classes('text-white font-semibold')

    # ===== INDICATOR CHART =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Open Range & Keltner').classes('text-xl font-bold text-white mb-2')

        indicator_chart = ui.plotly(go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode='lines', name='Keltner Upper', line=dict(color='#ef4444', width=2)),
                go.Scatter(x=[], y=[], mode='lines', name='Keltner Middle', line=dict(color='#3b82f6', width=2)),
                go.Scatter(x=[], y=[], mode='lines', name='Keltner Lower', line=dict(color='#10b981', width=2)),
                go.Scatter(x=[], y=[], mode='lines', name='Open Range High', line=dict(color='#f59e0b', width=2, dash='dash')),
                go.Scatter(x=[], y=[], mode='lines', name='Open Range Low', line=dict(color='#fbbf24', width=2, dash='dot')),
            ],
            layout=go.Layout(
                template='plotly_dark',
                height=300,
                margin=dict(l=50, r=20, t=20, b=40),
                xaxis=dict(title='Recent Points', showgrid=True, gridcolor='#374151'),
                yaxis=dict(title='Price ($)', showgrid=True, gridcolor='#374151'),
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

    refresh_in_flight = False

    def _lookup_market_data(state):
        market_data = None
        if state.market_data:
            if isinstance(state.market_data, dict):
                market_data = state.market_data.get(asset_select.value)
            elif isinstance(state.market_data, list):
                market_data = next((m for m in state.market_data if m.get('asset') == asset_select.value), None)
        return market_data

    def _has_indicator_payload(market_data) -> bool:
        if not market_data:
            return False
        intraday = market_data.get('intraday') or {}
        long_term = market_data.get('long_term') or {}
        for frame in (intraday, long_term):
            series = frame.get('series') or {}
            if frame.get('avwap') is not None:
                return True
            if frame.get('opening_range'):
                return True
            if any(series.get(key) for key in ('sma99', 'keltner_middle', 'keltner_upper', 'keltner_lower', 'timestamps')):
                return True
            candles = series.get('price_candles') or {}
            if any(candles.get(key) for key in ('time', 'open', 'high', 'low', 'close')):
                return True
        return False

    def _clear_price_chart():
        candle_trace = price_chart.figure.data[0]
        candle_trace.x = []
        candle_trace.open = []
        candle_trace.high = []
        candle_trace.low = []
        candle_trace.close = []
        price_chart.update()

    def _set_price_chart(price_candles: dict, market_snapshot: dict):
        """Render recent OHLC candles when available."""
        time_values = list(price_candles.get('time') or [])
        open_values = list(price_candles.get('open') or [])
        high_values = list(price_candles.get('high') or [])
        low_values = list(price_candles.get('low') or [])
        close_values = list(price_candles.get('close') or [])

        if not time_values or not open_values or not high_values or not low_values or not close_values:
            fallback_times = list(market_snapshot.get('recent_timestamps') or [])
            fallback_prices = list(market_snapshot.get('recent_mid_prices') or [])
            if fallback_times and fallback_prices:
                point_count = min(len(fallback_times), len(fallback_prices))
                aligned_prices = fallback_prices[-point_count:]
                time_values = fallback_times[-point_count:]
                open_values = aligned_prices
                high_values = aligned_prices
                low_values = aligned_prices
                close_values = aligned_prices

        point_count = min(len(time_values), len(open_values), len(high_values), len(low_values), len(close_values))

        if point_count == 0:
            _clear_price_chart()
            return

        price_chart.figure.data[0].x = time_values[-point_count:]
        price_chart.figure.data[0].open = open_values[-point_count:]
        price_chart.figure.data[0].high = high_values[-point_count:]
        price_chart.figure.data[0].low = low_values[-point_count:]
        price_chart.figure.data[0].close = close_values[-point_count:]
        price_chart.update()

    def _set_indicator_chart(series: dict, opening_range: dict, market_snapshot: dict):
        """Render Keltner bands with the current opening range."""
        keltner_upper = list(series.get('keltner_upper') or [])
        keltner_middle = list(series.get('keltner_middle') or [])
        keltner_lower = list(series.get('keltner_lower') or [])
        timestamps = list(series.get('timestamps') or market_snapshot.get('recent_timestamps') or [])
        point_count = max(
            len(keltner_upper),
            len(keltner_middle),
            len(keltner_lower),
            0,
        )

        if point_count == 0:
            for trace in indicator_chart.figure.data:
                trace.x = []
                trace.y = []
            indicator_chart.update()
            return

        if len(timestamps) >= point_count:
            x_values = timestamps[-point_count:]
        else:
            x_values = list(range(1, point_count + 1))

        def _align(values: list) -> list:
            if len(values) >= point_count:
                return values[-point_count:]
            return [None] * (point_count - len(values)) + values

        or_high = opening_range.get('high')
        or_low = opening_range.get('low')
        or_high_series = [or_high] * point_count if or_high is not None else [None] * point_count
        or_low_series = [or_low] * point_count if or_low is not None else [None] * point_count

        indicator_chart.figure.data[0].x = x_values
        indicator_chart.figure.data[0].y = _align(keltner_upper)
        indicator_chart.figure.data[1].x = x_values
        indicator_chart.figure.data[1].y = _align(keltner_middle)
        indicator_chart.figure.data[2].x = x_values
        indicator_chart.figure.data[2].y = _align(keltner_lower)
        indicator_chart.figure.data[3].x = x_values
        indicator_chart.figure.data[3].y = or_high_series
        indicator_chart.figure.data[4].x = x_values
        indicator_chart.figure.data[4].y = or_low_series
        indicator_chart.update()

    async def _bootstrap_market_snapshot():
        """Fetch indicators on demand when the bot is not supplying them."""
        if not _ui_ok():
            return
        nonlocal refresh_in_flight
        if refresh_in_flight or bot_service.is_running():
            return

        state = state_manager.get_state()
        if _has_indicator_payload(_lookup_market_data(state)):
            return

        refresh_in_flight = True
        try:
            await bot_service.refresh_market_data(include_indicators=True)
        finally:
            refresh_in_flight = False

    # ===== AUTO-REFRESH LOGIC =====
    async def update_market_data():
        """Update market data and indicators from real bot data"""
        if not _ui_ok():
            return
        state = state_manager.get_state()
        selected_asset = asset_select.value
        selected_interval = str(interval_select.value or '5m')
        valid_intervals = {'1m', '5m', '15m', '1h', '4h', '1d'}
        if selected_interval not in valid_intervals:
            selected_interval = '5m'
        hedge_metric = next(
            (m for m in (getattr(state, 'hedge_metrics', []) or []) if m.get('underlying') == selected_asset),
            None,
        )
        degraded_reason = None
        hedge_status = getattr(state, 'hedge_status', {}) or {}
        hedge_enabled = bool(hedge_status.get('enabled', True))
        hedge_available = bool(hedge_status.get('available', True))
        degraded_map = hedge_status.get('degraded_underlyings', {}) or {}
        state_error = hedge_status.get('state_error')
        if selected_asset in degraded_map:
            degraded_reason = degraded_map[selected_asset]

        # Get market data for selected asset from bot state
        market_data = _lookup_market_data(state)

        if (not market_data or not _has_indicator_payload(market_data)) and not bot_service.is_running():
            await _bootstrap_market_snapshot()
            state = state_manager.get_state()
            market_data = _lookup_market_data(state)

        if not market_data:
            # No data available yet
            current_price_label.set_text('Loading...')
            change_24h_label.set_text('--')
            volume_24h_label.set_text('--')
            open_interest_label.set_text('--')
            hedge_status_label.set_text('NO DATA')
            hedge_status_desc.set_text('Waiting for market and hedge data from bot...')
            hedge_target_label.set_text('Target: --')
            hedge_current_label.set_text('Current: --')
            hedge_residual_label.set_text('Residual: --')
            hedge_last_label.set_text('Last rebalance: --')
            _clear_price_chart()
            _set_indicator_chart({}, {}, {})
            return

        # Update price cards with real data
        current_price = market_data.get('price') or market_data.get('current_price', 0)
        current_price_label.set_text(f'${current_price:,.2f}')
        
        prev_day_price = market_data.get('prev_day_price')
        if prev_day_price and prev_day_price > 0:
            change_24h_pct = ((current_price - prev_day_price) / prev_day_price) * 100
        else:
            change_24h_pct = None
        _set_change_label(change_24h_pct)

        # Volume and OI
        volume_24h = market_data.get('volume_24h')
        volume_24h_label.set_text(_format_compact_number(volume_24h, prefix='$'))

        open_interest = market_data.get('open_interest', 0)
        if open_interest:
            open_interest_label.set_text(_format_compact_number(open_interest))
        else:
            open_interest_label.set_text('--')
        
        # Update indicators from the selected interval, with the alternate
        # cadence as a fallback when a metric only exists on one payload.
        intraday = market_data.get('intraday') or {}
        long_term = market_data.get('long_term') or {}
        long_term_interval = str(long_term.get('interval') or '4h')

        if selected_interval in {'1m', '5m', '15m'}:
            selected_frame = intraday
            alternate_frame = long_term
        elif selected_interval == long_term_interval or selected_interval in {'1h', '4h', '1d'}:
            selected_frame = long_term or intraday
            alternate_frame = intraday
        else:
            selected_frame = intraday
            alternate_frame = long_term

        sma99_selected = selected_frame.get('sma99')
        sma99_alt = alternate_frame.get('sma99')
        avwap = selected_frame.get('avwap') or alternate_frame.get('avwap')
        opening_range = selected_frame.get('opening_range') or intraday.get('opening_range', {})
        keltner = selected_frame.get('keltner') or alternate_frame.get('keltner', {})
        
        if sma99_selected:
            ema20_label.set_text(f'${sma99_selected:,.2f}')
        else:
            ema20_label.set_text('--')
        
        if sma99_alt:
            ema50_label.set_text(f'${sma99_alt:,.2f}')
        else:
            ema50_label.set_text('--')
        
        selected_series = selected_frame.get('series') or alternate_frame.get('series') or {}
        _set_price_chart(selected_series.get('price_candles') or {}, market_data)
        middle = keltner.get('middle')
        upper = keltner.get('upper')
        lower = keltner.get('lower')
        keltner_middle_label.set_text(f'${middle:,.2f}' if middle is not None else '--')
        keltner_upper_label.set_text(f'${upper:,.2f}' if upper is not None else '--')
        keltner_lower_label.set_text(f'${lower:,.2f}' if lower is not None else '--')

        if avwap:
            avwap_label.set_text(f'${avwap:,.2f}')
            if current_price and current_price > avwap:
                avwap_progress.set_value(0.75)
            elif current_price and current_price < avwap:
                avwap_progress.set_value(0.25)
            else:
                avwap_progress.set_value(0.5)
        else:
            avwap_label.set_text('--')
            avwap_progress.set_value(0.5)

        if avwap:
            atr_label.set_text(f'${avwap:,.2f}')
        else:
            atr_label.set_text('--')

        or_high = opening_range.get('high')
        or_low = opening_range.get('low')
        opening_range_high_label.set_text(f'${or_high:,.2f}' if or_high is not None else '--')
        opening_range_low_label.set_text(f'${or_low:,.2f}' if or_low is not None else '--')
        _set_indicator_chart(selected_series, opening_range, market_data)
        
        # Update sentiment based on indicators
        opening_range_position = opening_range.get('position')
        if sma99_selected and avwap and current_price:
            if current_price > sma99_selected and current_price > avwap and opening_range_position == 'above':
                sentiment_label.set_text('BULLISH')
                _set_status_tone(sentiment_label, 'text-green-400')
                sentiment_desc.set_text('Price above SMA99 and AVWAP, holding above opening range')
                trend_icon.set_text('●')
                _set_status_tone(trend_icon, 'text-green-400')
            elif current_price < sma99_selected and current_price < avwap and opening_range_position == 'below':
                sentiment_label.set_text('BEARISH')
                _set_status_tone(sentiment_label, 'text-red-400')
                sentiment_desc.set_text('Price below SMA99 and AVWAP, holding below opening range')
                trend_icon.set_text('●')
                _set_status_tone(trend_icon, 'text-red-400')
            else:
                sentiment_label.set_text('NEUTRAL')
                _set_status_tone(sentiment_label, 'text-gray-400')
                sentiment_desc.set_text('Mixed signals, waiting for clear direction')
                trend_icon.set_text('○')
                _set_status_tone(trend_icon, 'text-gray-400')
        else:
            sentiment_label.set_text('NO DATA')
            _set_status_tone(sentiment_label, 'text-gray-500')
            sentiment_desc.set_text('Waiting for market data from bot...')

        momentum_icon.set_text('○')
        _set_status_tone(momentum_icon, 'text-gray-400')
        volume_icon.set_text('○')
        _set_status_tone(volume_icon, 'text-gray-400')

        if hedge_metric:
            hedge_status_label.set_text(str(hedge_metric.get('status', 'unknown')).upper())
            if hedge_metric.get('degraded'):
                _set_status_tone(hedge_status_label, 'text-red-400')
                hedge_status_desc.set_text(degraded_reason or hedge_metric.get('degraded_reason') or 'Live greeks unavailable')
            else:
                _set_status_tone(hedge_status_label, 'text-green-400')
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
            if not hedge_available:
                hedge_status_label.set_text('UNAVAILABLE')
                _set_status_tone(hedge_status_label, 'text-yellow-400')
                hedge_status_desc.set_text('Delta hedge is unavailable in the current bot configuration')
            elif not hedge_enabled:
                hedge_status_label.set_text('DISABLED')
                _set_status_tone(hedge_status_label, 'text-yellow-400')
                hedge_status_desc.set_text('Delta hedge is paused from the dashboard control')
            elif state_error:
                hedge_status_label.set_text('UNAVAILABLE')
                _set_status_tone(hedge_status_label, 'text-yellow-400')
                hedge_status_desc.set_text(str(state_error))
            elif degraded_reason:
                hedge_status_label.set_text('DEGRADED')
                _set_status_tone(hedge_status_label, 'text-red-400')
                hedge_status_desc.set_text(degraded_reason)
            else:
                hedge_status_label.set_text('IDLE')
                _set_status_tone(hedge_status_label, 'text-gray-400')
                hedge_status_desc.set_text('No active hedge for this asset')
            hedge_target_label.set_text('Target: --')
            hedge_current_label.set_text('Current: --')
            hedge_residual_label.set_text('Residual: --')
            hedge_last_label.set_text('Last rebalance: --')

    # Auto-refresh every 5 seconds
    ui.timer(5.0, update_market_data)
    ui.timer(0.1, _bootstrap_market_snapshot, once=True)

    # Refresh on asset/interval change
    async def _handle_market_selection_change(_=None):
        if not _ui_ok():
            return
        await update_market_data()

    asset_select.on('update:model-value', _handle_market_selection_change)
    interval_select.on('update:model-value', _handle_market_selection_change)
