"""
Settings Page - Configuration management with tabs
"""

import asyncio
import json
import os
from pathlib import Path
from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager
from src.gui.services.ui_utils import is_ui_alive
from src.backend.config_loader import CONFIG


def create_settings(bot_service: BotService, state_manager: StateManager):
    """Create settings page with 4 tabs for configuration"""

    def _ui_ok():
        return is_ui_alive(assets_input)

    def _assets_to_text(raw_assets) -> str:
        if isinstance(raw_assets, list):
            return ', '.join(str(asset).strip() for asset in raw_assets if str(asset).strip())
        return str(raw_assets or '')

    ui.label('Settings').classes('text-3xl font-bold mb-4 text-white')

    # Configuration file path
    config_file = Path('data/config.json')

    # Load configuration from file or use defaults
    def load_config():
        """Load configuration from file"""
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                ui.notify(f'Failed to load config: {e}', type='warning')

        # Return defaults from environment
        return {
            'strategy': {
                'assets': _assets_to_text(bot_service._parse_assets(CONFIG.get('assets')) or ['BTC', 'ETH']),
                'interval': CONFIG.get('interval') or '5m',
                'llm_model': CONFIG.get('llm_model') or 'x-ai/grok-4',
                'reasoning_enabled': CONFIG.get('reasoning_enabled', False),
                'reasoning_effort': CONFIG.get('reasoning_effort') or 'high'
            },
            'api_keys': {
                'taapi_api_key': CONFIG.get('taapi_api_key') or '',
                'hyperliquid_private_key': CONFIG.get('hyperliquid_private_key') or '',
                'hyperliquid_network': CONFIG.get('hyperliquid_network') or 'mainnet',
                'openrouter_api_key': CONFIG.get('openrouter_api_key') or '',
                'thalex_network': CONFIG.get('thalex_network') or 'test',
                'thalex_key_id': CONFIG.get('thalex_key_id') or '',
                'thalex_private_key_path': CONFIG.get('thalex_private_key_path') or '',
                'thalex_account': CONFIG.get('thalex_account') or '',
            },
            'risk_management': {
                'max_position_size': 1000,
                'max_leverage': 3,
                'stop_loss_pct': 5,
                'take_profit_pct': 10,
                'max_open_positions': 5
            },
            'notifications': {
                'desktop_enabled': True,
                'telegram_enabled': False,
                'telegram_token': '',
                'telegram_chat_id': ''
            }
        }

    def save_config(config_data):
        """Save configuration to file"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            ui.notify(f'Failed to save config: {e}', type='negative')
            return False

    # Load initial configuration
    config_data = load_config()

    def _sync_runtime_api_config() -> None:
        """Keep in-memory config and env in sync with the API settings form."""
        runtime_values = {
            'taapi_api_key': taapi_input.value or '',
            'hyperliquid_private_key': hyperliquid_input.value or '',
            'hyperliquid_network': hyperliquid_network.value or 'mainnet',
            'openrouter_api_key': openrouter_input.value or '',
            'thalex_network': thalex_network.value or 'test',
            'thalex_key_id': thalex_key_id_input.value or '',
            'thalex_private_key_path': thalex_pem_input.value or '',
            'thalex_account': thalex_account_input.value or '',
        }
        env_map = {
            'taapi_api_key': 'TAAPI_API_KEY',
            'hyperliquid_private_key': 'HYPERLIQUID_PRIVATE_KEY',
            'hyperliquid_network': 'HYPERLIQUID_NETWORK',
            'openrouter_api_key': 'OPENROUTER_API_KEY',
            'thalex_network': 'THALEX_NETWORK',
            'thalex_key_id': 'THALEX_KEY_ID',
            'thalex_private_key_path': 'THALEX_PRIVATE_KEY_PATH',
            'thalex_account': 'THALEX_ACCOUNT',
        }

        for key, value in runtime_values.items():
            CONFIG[key] = value
            env_name = env_map[key]
            if value:
                os.environ[env_name] = value
            else:
                os.environ.pop(env_name, None)

    # ===== TABBED INTERFACE =====
    with ui.card().classes('w-full p-4'):
        with ui.tabs().classes('w-full') as tabs:
            tab_strategy = ui.tab('Strategy', icon='analytics')
            tab_api = ui.tab('API Keys', icon='key')
            tab_risk = ui.tab('Risk Management', icon='shield')
            tab_notifications = ui.tab('Notifications', icon='notifications')

        with ui.tab_panels(tabs, value=tab_strategy).classes('w-full'):
            # ===== TAB 1: STRATEGY CONFIGURATION =====
            with ui.tab_panel(tab_strategy):
                ui.label('Strategy Configuration').classes('text-2xl font-bold mb-4 text-white')

                with ui.column().classes('gap-4 w-full max-w-2xl'):
                    # Assets input
                    ui.label('Trading Assets').classes('text-lg font-semibold text-white')
                    assets_input = ui.textarea(
                        label='Assets (comma-separated)',
                        placeholder='BTC, ETH, SOL',
                        value=config_data['strategy']['assets']
                    ).classes('w-full')
                    ui.label('Example: BTC, ETH, SOL or BTC ETH SOL').classes('text-xs text-gray-400')

                    ui.separator()

                    # Interval selection
                    ui.label('Trading Interval').classes('text-lg font-semibold text-white')
                    interval_select = ui.select(
                        label='Interval',
                        options=['1m', '5m', '15m', '1h', '4h'],
                        value=config_data['strategy']['interval']
                    ).classes('w-full')
                    ui.label('Timeframe for trading decisions').classes('text-xs text-gray-400')

                    ui.separator()

                    # LLM Model selection
                    ui.label('LLM Model').classes('text-lg font-semibold text-white')
                    _llm_options = [
                        'x-ai/grok-4',
                        'openai/gpt-4',
                        'anthropic/claude-3.5-sonnet',
                        'deepseek/deepseek-chat-v3.1',
                    ]
                    _current_llm = config_data['strategy']['llm_model']
                    if _current_llm not in _llm_options:
                        _llm_options.append(_current_llm)
                    llm_model_select = ui.select(
                        label='Model',
                        options=_llm_options,
                        value=_current_llm,
                    ).classes('w-full')
                    ui.label('LLM model for trading decisions').classes('text-xs text-gray-400')

                    ui.separator()

                    # Reasoning configuration
                    with ui.row().classes('items-center gap-4'):
                        reasoning_enabled = ui.checkbox(
                            'Enable Reasoning Tokens',
                            value=config_data['strategy']['reasoning_enabled']
                        )
                        ui.label('Use extended reasoning for better decisions').classes('text-sm text-gray-400')

                    reasoning_effort = ui.select(
                        label='Reasoning Effort',
                        options=['low', 'medium', 'high'],
                        value=config_data['strategy']['reasoning_effort']
                    ).classes('w-full')

                    ui.separator()

                    # Save and Load buttons
                    async def save_strategy_config():
                        if not _ui_ok():
                            return
                        try:
                            # Update config data
                            config_data['strategy']['assets'] = assets_input.value
                            config_data['strategy']['interval'] = interval_select.value
                            config_data['strategy']['llm_model'] = llm_model_select.value
                            config_data['strategy']['reasoning_enabled'] = reasoning_enabled.value
                            config_data['strategy']['reasoning_effort'] = reasoning_effort.value

                            # Save to file
                            if save_config(config_data):
                                # Update bot service config
                                assets_list = [a.strip() for a in config_data['strategy']['assets'].replace(',', ' ').split() if a.strip()]
                                await bot_service.update_config({
                                    'assets': assets_list,
                                    'interval': config_data['strategy']['interval'],
                                    'llm_model': config_data['strategy']['llm_model'],
                                    'reasoning_enabled': config_data['strategy']['reasoning_enabled'],
                                    'reasoning_effort': config_data['strategy']['reasoning_effort'],
                                })
                                if _ui_ok():
                                    ui.notify('Strategy configuration saved successfully!', type='positive')
                            else:
                                if _ui_ok():
                                    ui.notify('Failed to save configuration', type='negative')
                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error saving config: {str(e)}', type='negative')

                    async def load_strategy_config():
                        if not _ui_ok():
                            return
                        try:
                            loaded_config = load_config()
                            assets_input.value = loaded_config['strategy']['assets']
                            interval_select.value = loaded_config['strategy']['interval']
                            _loaded_llm = loaded_config['strategy']['llm_model']
                            if _loaded_llm not in llm_model_select.options:
                                llm_model_select.options.append(_loaded_llm)
                                llm_model_select.update()
                            llm_model_select.value = _loaded_llm
                            reasoning_enabled.value = loaded_config['strategy']['reasoning_enabled']
                            reasoning_effort.value = loaded_config['strategy']['reasoning_effort']
                            ui.notify('Configuration loaded successfully!', type='positive')
                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error loading config: {str(e)}', type='negative')

                    with ui.row().classes('gap-2'):
                        ui.button('Save Configuration', on_click=save_strategy_config, icon='save').props('color=primary')
                        ui.button('Load Configuration', on_click=load_strategy_config, icon='refresh').props('color=secondary')

            # ===== TAB 2: API KEYS =====
            with ui.tab_panel(tab_api):
                ui.label('API Keys Configuration').classes('text-2xl font-bold mb-4 text-white')

                with ui.column().classes('gap-4 w-full max-w-2xl'):
                    # Connection status indicators
                    ui.label('Connection Status').classes('text-lg font-semibold text-white')
                    with ui.row().classes('gap-4 items-center'):
                        taapi_status = ui.label('TAAPI: Not configured').classes('text-sm text-gray-400')
                        hyperliquid_status = ui.label('Hyperliquid: Not configured').classes('text-sm text-gray-400')
                        openrouter_status = ui.label('OpenRouter: Not configured').classes('text-sm text-gray-400')
                        thalex_status = ui.label('Thalex: Not configured').classes('text-sm text-gray-400')
                    thalex_error = ui.label('').classes('text-xs text-red-300')

                    ui.separator()

                    # TAAPI Key
                    ui.label('TAAPI.io API Key').classes('text-lg font-semibold text-white')
                    taapi_input = ui.input(
                        label='TAAPI API Key',
                        placeholder='eyJhbGciOiJI...',
                        value=config_data['api_keys']['taapi_api_key'],
                        password=True,
                        password_toggle_button=True
                    ).classes('w-full')
                    ui.label('Get your API key from https://taapi.io').classes('text-xs text-gray-400')

                    ui.separator()

                    # Hyperliquid Key
                    ui.label('Hyperliquid Private Key').classes('text-lg font-semibold text-white')
                    hyperliquid_input = ui.input(
                        label='Private Key',
                        placeholder='0x...',
                        value=config_data['api_keys']['hyperliquid_private_key'],
                        password=True,
                        password_toggle_button=True
                    ).classes('w-full')
                    ui.label('Your wallet private key for trading').classes('text-xs text-gray-400')

                    # Network selection
                    hyperliquid_network = ui.select(
                        label='Network',
                        options=['mainnet', 'testnet'],
                        value=config_data['api_keys']['hyperliquid_network']
                    ).classes('w-full')

                    ui.separator()

                    # OpenRouter Key
                    ui.label('OpenRouter API Key').classes('text-lg font-semibold text-white')
                    openrouter_input = ui.input(
                        label='OpenRouter API Key',
                        placeholder='sk-or-v1-...',
                        value=config_data['api_keys']['openrouter_api_key'],
                        password=True,
                        password_toggle_button=True
                    ).classes('w-full')
                    ui.label('Get your API key from https://openrouter.ai').classes('text-xs text-gray-400')

                    ui.separator()

                    # Thalex (options venue)
                    ui.label('Thalex (Options Venue)').classes('text-lg font-semibold text-white')
                    ui.label('JWT RS512 with an RSA private key generated in the Thalex UI. Start on TESTNET.').classes('text-xs text-gray-400')

                    thalex_network = ui.select(
                        label='Thalex Network',
                        options=['test', 'prod'],
                        value=config_data['api_keys']['thalex_network'],
                    ).classes('w-full')

                    thalex_key_id_input = ui.input(
                        label='Thalex Key ID (kid)',
                        placeholder='K...',
                        value=config_data['api_keys']['thalex_key_id'],
                    ).classes('w-full')

                    thalex_pem_input = ui.input(
                        label='Thalex Private Key PEM Path',
                        placeholder='./secrets/thalex_private.pem',
                        value=config_data['api_keys']['thalex_private_key_path'],
                    ).classes('w-full')

                    thalex_account_input = ui.input(
                        label='Thalex Account (optional, for sub-accounts)',
                        placeholder='',
                        value=config_data['api_keys']['thalex_account'],
                    ).classes('w-full')

                    ui.separator()

                    # Test connections button
                    async def test_api_connections():
                        """Test all API connections"""
                        if not _ui_ok():
                            return
                        try:
                            ui.notify('Testing API connections...', type='info')

                            _sync_runtime_api_config()

                            # Test connections via bot service
                            results = await bot_service.test_api_connections()

                            if not _ui_ok():
                                return

                            # Update status indicators
                            _set_api_status(taapi_status, 'TAAPI', bool(results.get('TAAPI', False)))
                            _set_api_status(hyperliquid_status, 'Hyperliquid', bool(results.get('Hyperliquid', False)))
                            _set_api_status(openrouter_status, 'OpenRouter', bool(results.get('OpenRouter', False)))
                            _set_api_status(thalex_status, 'Thalex', bool(results.get('Thalex', False)))
                            errors = results.get('errors')
                            if isinstance(errors, dict) and errors.get('Thalex'):
                                thalex_error.text = f"Thalex error: {errors.get('Thalex')}"
                            else:
                                thalex_error.text = ''

                            # Show summary notification
                            api_names = ('TAAPI', 'Hyperliquid', 'OpenRouter', 'Thalex')
                            connected_count = sum(1 for name in api_names if results.get(name))
                            total_count = len(api_names)

                            if connected_count == total_count:
                                ui.notify(f'All APIs connected successfully! ({connected_count}/{total_count})', type='positive')
                            elif connected_count > 0:
                                ui.notify(f'Partially connected: {connected_count}/{total_count} APIs', type='warning')
                            else:
                                ui.notify('All connections failed. Check your API keys.', type='negative')

                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error testing connections: {str(e)}', type='negative')

                    async def save_api_keys():
                        """Save API keys to configuration"""
                        if not _ui_ok():
                            return
                        try:
                            # Update config data
                            config_data['api_keys']['taapi_api_key'] = taapi_input.value
                            config_data['api_keys']['hyperliquid_private_key'] = hyperliquid_input.value
                            config_data['api_keys']['hyperliquid_network'] = hyperliquid_network.value
                            config_data['api_keys']['openrouter_api_key'] = openrouter_input.value
                            config_data['api_keys']['thalex_network'] = thalex_network.value
                            config_data['api_keys']['thalex_key_id'] = thalex_key_id_input.value
                            config_data['api_keys']['thalex_private_key_path'] = thalex_pem_input.value
                            config_data['api_keys']['thalex_account'] = thalex_account_input.value

                            # Save to file
                            if save_config(config_data):
                                _sync_runtime_api_config()
                                _refresh_api_status_hints()

                                ui.notify('API keys saved successfully!', type='positive')
                                ui.notify('Note: Restart the bot for changes to take effect', type='info')
                            else:
                                ui.notify('Failed to save API keys', type='negative')
                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error saving API keys: {str(e)}', type='negative')

                    with ui.row().classes('gap-2'):
                        ui.button('Save API Keys', on_click=save_api_keys, icon='save').props('color=primary')
                        ui.button('Test Connections', on_click=test_api_connections, icon='network_check').props('color=secondary')

                    def _set_status_tone(label, tone: str):
                        label.classes(remove='text-gray-400 text-green-400 text-red-400 text-yellow-400')
                        label.classes(add=tone)

                    def _set_api_hint(label, service_name: str, configured: bool, pending: bool = False):
                        if not configured:
                            label.text = f'{service_name}: Not configured'
                            _set_status_tone(label, 'text-gray-400')
                        elif pending:
                            label.text = f'{service_name}: Configured, not tested'
                            _set_status_tone(label, 'text-yellow-400')

                    def _set_api_status(label, service_name: str, connected: bool):
                        label.text = f"{service_name}: {'Connected' if connected else 'Failed'}"
                        _set_status_tone(label, 'text-green-400' if connected else 'text-red-400')

                    def _refresh_api_status_hints():
                        _set_api_hint(taapi_status, 'TAAPI', bool(taapi_input.value), pending=True)
                        _set_api_hint(hyperliquid_status, 'Hyperliquid', bool(hyperliquid_input.value), pending=True)
                        _set_api_hint(openrouter_status, 'OpenRouter', bool(openrouter_input.value), pending=True)
                        thalex_configured = bool(thalex_key_id_input.value and thalex_pem_input.value)
                        _set_api_hint(thalex_status, 'Thalex', thalex_configured, pending=True)

                    _refresh_api_status_hints()

                    ui.separator()

                    # Warning
                    with ui.card().classes('bg-orange-900 p-4'):
                        ui.label('⚠️ Security Warning').classes('text-lg font-bold text-white mb-2')
                        ui.label('Never share your private keys. Keys are stored in data/config.json - keep this file secure!').classes('text-sm text-gray-200')

            # ===== TAB 3: RISK MANAGEMENT =====
            with ui.tab_panel(tab_risk):
                ui.label('Risk Management').classes('text-2xl font-bold mb-4 text-white')

                with ui.column().classes('gap-6 w-full max-w-2xl'):
                    # Max Position Size
                    ui.label('Maximum Position Size').classes('text-lg font-semibold text-white')
                    with ui.row().classes('items-center w-full gap-4'):
                        max_position_slider = ui.slider(
                            min=100,
                            max=10000,
                            value=config_data['risk_management']['max_position_size'],
                            step=100
                        ).classes('flex-grow')
                        max_position_label = ui.label(f"${config_data['risk_management']['max_position_size']:,.0f}").classes('text-white font-bold min-w-[100px]')
                        max_position_slider.on('update:model-value', lambda e: max_position_label.set_text(f'${e.args:,.0f}'))
                    ui.label('Maximum USD allocation per position').classes('text-xs text-gray-400')

                    ui.separator()

                    # Max Leverage
                    ui.label('Maximum Leverage').classes('text-lg font-semibold text-white')
                    with ui.row().classes('items-center w-full gap-4'):
                        max_leverage_slider = ui.slider(
                            min=1,
                            max=20,
                            value=config_data['risk_management']['max_leverage'],
                            step=0.5
                        ).classes('flex-grow')
                        max_leverage_label = ui.label(f"{config_data['risk_management']['max_leverage']:.1f}x").classes('text-white font-bold min-w-[100px]')
                        max_leverage_slider.on('update:model-value', lambda e: max_leverage_label.set_text(f'{e.args:.1f}x'))
                    ui.label('Maximum leverage for perpetual futures (1x-20x)').classes('text-xs text-gray-400')

                    ui.separator()

                    # Stop Loss %
                    ui.label('Default Stop Loss').classes('text-lg font-semibold text-white')
                    with ui.row().classes('items-center w-full gap-4'):
                        stop_loss_slider = ui.slider(
                            min=1,
                            max=20,
                            value=config_data['risk_management']['stop_loss_pct'],
                            step=0.5
                        ).classes('flex-grow')
                        stop_loss_label = ui.label(f"{config_data['risk_management']['stop_loss_pct']:.1f}%").classes('text-white font-bold min-w-[100px]')
                        stop_loss_slider.on('update:model-value', lambda e: stop_loss_label.set_text(f'{e.args:.1f}%'))
                    ui.label('Default stop loss percentage from entry').classes('text-xs text-gray-400')

                    ui.separator()

                    # Take Profit %
                    ui.label('Default Take Profit').classes('text-lg font-semibold text-white')
                    with ui.row().classes('items-center w-full gap-4'):
                        take_profit_slider = ui.slider(
                            min=2,
                            max=50,
                            value=config_data['risk_management']['take_profit_pct'],
                            step=1
                        ).classes('flex-grow')
                        take_profit_label = ui.label(f"{config_data['risk_management']['take_profit_pct']:.0f}%").classes('text-white font-bold min-w-[100px]')
                        take_profit_slider.on('update:model-value', lambda e: take_profit_label.set_text(f'{e.args:.0f}%'))
                    ui.label('Default take profit percentage from entry').classes('text-xs text-gray-400')

                    ui.separator()

                    # Max Open Positions
                    ui.label('Maximum Open Positions').classes('text-lg font-semibold text-white')
                    max_positions_input = ui.number(
                        label='Max Positions',
                        value=config_data['risk_management']['max_open_positions'],
                        min=1,
                        max=20
                    ).classes('w-full')
                    ui.label('Maximum number of concurrent open positions').classes('text-xs text-gray-400')

                    ui.separator()

                    # Save button
                    async def save_risk_config():
                        if not _ui_ok():
                            return
                        try:
                            # Update config data
                            config_data['risk_management']['max_position_size'] = int(max_position_slider.value)
                            config_data['risk_management']['max_leverage'] = float(max_leverage_slider.value)
                            config_data['risk_management']['stop_loss_pct'] = float(stop_loss_slider.value)
                            config_data['risk_management']['take_profit_pct'] = float(take_profit_slider.value)
                            config_data['risk_management']['max_open_positions'] = int(max_positions_input.value)

                            # Save to file
                            if save_config(config_data):
                                ui.notify('Risk management settings saved successfully!', type='positive')
                            else:
                                ui.notify('Failed to save risk settings', type='negative')
                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error saving risk config: {str(e)}', type='negative')

                    ui.button('Save Risk Settings', on_click=save_risk_config, icon='save').props('color=primary')

                    ui.separator()

                    # Warning about leverage
                    with ui.card().classes('bg-red-900 p-4'):
                        ui.label('⚠️ Risk Warning').classes('text-lg font-bold text-white mb-2')
                        ui.label('High leverage increases both potential profits and losses. Use with caution!').classes('text-sm text-gray-200')

            # ===== TAB 4: NOTIFICATIONS =====
            with ui.tab_panel(tab_notifications):
                ui.label('Notifications').classes('text-2xl font-bold mb-4 text-white')

                with ui.column().classes('gap-4 w-full max-w-2xl'):
                    # Desktop Notifications
                    ui.label('Desktop Notifications').classes('text-lg font-semibold text-white')
                    desktop_enabled_checkbox = ui.checkbox(
                        'Enable Desktop Notifications',
                        value=config_data['notifications']['desktop_enabled']
                    )
                    ui.label('Show system notifications for important events').classes('text-xs text-gray-400')

                    ui.separator()

                    # Telegram Bot
                    ui.label('Telegram Notifications').classes('text-lg font-semibold text-white')
                    telegram_enabled_checkbox = ui.checkbox(
                        'Enable Telegram Notifications',
                        value=config_data['notifications']['telegram_enabled']
                    )

                    # Telegram token and chat ID (shown when enabled)
                    telegram_token_input = ui.input(
                        label='Telegram Bot Token',
                        placeholder='123456:ABC-DEF...',
                        value=config_data['notifications']['telegram_token'],
                        password=True,
                        password_toggle_button=True
                    ).classes('w-full')
                    telegram_token_input.visible = telegram_enabled_checkbox.value

                    telegram_chat_input = ui.input(
                        label='Telegram Chat ID',
                        placeholder='123456789',
                        value=config_data['notifications']['telegram_chat_id']
                    ).classes('w-full')
                    telegram_chat_input.visible = telegram_enabled_checkbox.value

                    # Toggle visibility when checkbox changes
                    def toggle_telegram_inputs(e):
                        telegram_token_input.visible = e.value
                        telegram_chat_input.visible = e.value

                    telegram_enabled_checkbox.on('update:model-value', toggle_telegram_inputs)

                    ui.label('Get your bot token from @BotFather on Telegram').classes('text-xs text-gray-400')

                    ui.separator()

                    # Test Notification button
                    async def test_notification():
                        if not _ui_ok():
                            return
                        try:
                            if desktop_enabled_checkbox.value:
                                ui.notify('Test notification sent! This is how trade alerts will look.', type='info')
                            else:
                                ui.notify('Desktop notifications are disabled', type='warning')

                            if telegram_enabled_checkbox.value and telegram_token_input.value:
                                ui.notify('Telegram delivery test is not implemented in the GUI yet.', type='warning')

                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error sending test notification: {str(e)}', type='negative')

                    # Save button
                    async def save_notification_config():
                        if not _ui_ok():
                            return
                        try:
                            # Update config data
                            config_data['notifications']['desktop_enabled'] = desktop_enabled_checkbox.value
                            config_data['notifications']['telegram_enabled'] = telegram_enabled_checkbox.value
                            config_data['notifications']['telegram_token'] = telegram_token_input.value
                            config_data['notifications']['telegram_chat_id'] = telegram_chat_input.value

                            # Save to file
                            if save_config(config_data):
                                ui.notify('Notification settings saved successfully!', type='positive')
                            else:
                                ui.notify('Failed to save notification settings', type='negative')
                        except Exception as e:
                            if _ui_ok():
                                ui.notify(f'Error saving notification config: {str(e)}', type='negative')

                    with ui.row().classes('gap-2'):
                        ui.button('Save Notification Settings', on_click=save_notification_config, icon='save').props('color=primary')
                        ui.button('Test Notification', on_click=test_notification, icon='notifications_active').props('color=secondary')

                    ui.separator()

                    # Info box
                    with ui.card().classes('bg-blue-900 p-4'):
                        ui.label('ℹ️ Notification Features').classes('text-lg font-bold text-white mb-2')
                        ui.label('Configure notifications for trade executions, errors, and daily summaries.').classes('text-sm text-gray-200')

    # Load current configuration on page load
    async def load_initial_config():
        """Load current bot configuration on page initialization"""
        if not _ui_ok():
            return
        try:
            current_config = await bot_service.get_current_config() if hasattr(bot_service, 'get_current_config') else None
            if not _ui_ok():
                return
            if current_config:
                # Update UI with current config
                assets_input.value = _assets_to_text(current_config.get('assets')) or assets_input.value
                interval_select.value = current_config.get('interval', interval_select.value)
                _cfg_llm = current_config.get('llm_model', llm_model_select.value)
                if _cfg_llm not in llm_model_select.options:
                    llm_model_select.options.append(_cfg_llm)
                    llm_model_select.update()
                llm_model_select.value = _cfg_llm
                reasoning_enabled.value = bool(current_config.get('reasoning_enabled', reasoning_enabled.value))
                reasoning_effort.value = current_config.get('reasoning_effort', reasoning_effort.value)

                taapi_input.value = current_config.get('taapi_key', taapi_input.value)
                hyperliquid_input.value = current_config.get('hyperliquid_private_key', hyperliquid_input.value)
                hyperliquid_network.value = current_config.get('hyperliquid_network', hyperliquid_network.value)
                openrouter_input.value = current_config.get('openrouter_key', openrouter_input.value)
                thalex_network.value = current_config.get('thalex_network', thalex_network.value)
                thalex_key_id_input.value = current_config.get('thalex_key_id', thalex_key_id_input.value)
                thalex_pem_input.value = current_config.get('thalex_private_key_path', thalex_pem_input.value)
                thalex_account_input.value = current_config.get('thalex_account', thalex_account_input.value)

                _refresh_api_status_hints()
        except Exception as e:
            pass  # Fail silently on initial load

    # Schedule initial config load
    asyncio.create_task(load_initial_config())
