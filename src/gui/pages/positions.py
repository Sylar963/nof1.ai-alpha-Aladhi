"""
Positions Page - Active trading positions with real-time PnL tracking
"""

from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager
from src.gui.services.ui_utils import is_ui_alive


def create_positions(bot_service: BotService, state_manager: StateManager):
    """Create positions page with live table and action buttons"""

    def _ui_ok():
        return is_ui_alive(table)

    def _format_price(value) -> str:
        return f'${float(value):,.2f}' if value is not None else 'N/A'

    def _format_signed_currency(value) -> str:
        return f'${float(value):+,.2f}' if value is not None else 'N/A'

    def _format_position_snapshot(row: dict) -> str:
        return (
            f"Side: {row.get('side', 'N/A')}\n"
            f"Size: {float(row.get('quantity') or 0):.4f} {row.get('symbol', '')}\n"
            f"Entry Price: {_format_price(row.get('entry_price'))}\n"
            f"Current Price: {_format_price(row.get('current_price'))}\n"
            f"Unrealized PnL: {_format_signed_currency(row.get('unrealized_pnl'))}\n"
            f"PnL %: {float(row.get('pnl_pct') or 0):+.2f}%\n"
            f"Leverage: {row.get('leverage', 'N/A')}x\n"
            f"Liquidation Price: {_format_price(row.get('liquidation_price'))}"
        )
    
    ui.label('Active Positions').classes('text-3xl font-bold mb-4 text-white')
    ui.label('Source: latest bot state merged from Hyperliquid and Thalex portfolio data').classes('text-xs text-gray-400 mb-3')
    
    # Summary cards
    with ui.row().classes('w-full gap-4 mb-6'):
        # Total Positions
        with ui.card().classes('flex-1 p-4 bg-gradient-to-br from-blue-600 to-blue-800'):
            positions_count = ui.label('0').classes('text-3xl font-bold text-white')
            ui.label('Total Positions').classes('text-sm text-gray-200 mt-1')
        
        # Total Unrealized PnL
        with ui.card().classes('flex-1 p-4 bg-gradient-to-br from-purple-600 to-purple-800'):
            total_pnl = ui.label('$0.00').classes('text-3xl font-bold text-white')
            ui.label('Unrealized PnL').classes('text-sm text-gray-200 mt-1')
        
        # Total Exposure
        with ui.card().classes('flex-1 p-4 bg-gradient-to-br from-indigo-600 to-indigo-800'):
            total_exposure = ui.label('$0.00').classes('text-3xl font-bold text-white')
            ui.label('Total Exposure').classes('text-sm text-gray-200 mt-1')
    
    # Positions table
    with ui.card().classes('w-full p-4'):
        ui.label('Position Details').classes('text-xl font-bold text-white mb-4')
        
        # Table columns definition
        columns = [
            {'name': 'symbol', 'label': 'Position', 'field': 'symbol', 'align': 'left', 'sortable': True},
            {'name': 'venue', 'label': 'Venue', 'field': 'venue', 'align': 'center', 'sortable': True},
            {'name': 'opened_by', 'label': 'Opened By', 'field': 'opened_by', 'align': 'center', 'sortable': True},
            {'name': 'side', 'label': 'Side', 'field': 'side', 'align': 'center', 'sortable': True},
            {'name': 'quantity', 'label': 'Size', 'field': 'quantity', 'align': 'right', 'sortable': True},
            {'name': 'entry_price', 'label': 'Entry Price', 'field': 'entry_price', 'align': 'right', 'sortable': True},
            {'name': 'current_price', 'label': 'Current Price', 'field': 'current_price', 'align': 'right', 'sortable': True},
            {'name': 'unrealized_pnl', 'label': 'Unrealized PnL', 'field': 'unrealized_pnl', 'align': 'right', 'sortable': True},
            {'name': 'pnl_pct', 'label': 'PnL %', 'field': 'pnl_pct', 'align': 'right', 'sortable': True},
            {'name': 'leverage', 'label': 'Leverage', 'field': 'leverage', 'align': 'center', 'sortable': True},
            {'name': 'liquidation_price', 'label': 'Liq. Price', 'field': 'liquidation_price', 'align': 'right', 'sortable': True},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'},
        ]
        
        # Create table
        table = ui.table(
            columns=columns,
            rows=[],
            row_key='row_id',
            pagination={'rowsPerPage': 10, 'sortBy': 'unrealized_pnl', 'descending': True}
        ).classes('w-full')
        
        # Custom cell rendering for colored PnL
        table.add_slot('body-cell-unrealized_pnl', '''
            <q-td :props="props">
                <span :class="props.row.unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500'" class="font-bold">
                    {{ props.row.unrealized_pnl >= 0 ? '+' : '' }}${{ props.row.unrealized_pnl.toFixed(2) }}
                </span>
            </q-td>
        ''')
        
        # Custom cell rendering for PnL %
        table.add_slot('body-cell-pnl_pct', '''
            <q-td :props="props">
                <span :class="props.row.pnl_pct >= 0 ? 'text-green-500' : 'text-red-500'" class="font-bold">
                    {{ props.row.pnl_pct >= 0 ? '+' : '' }}{{ props.row.pnl_pct.toFixed(2) }}%
                </span>
            </q-td>
        ''')
        
        # Custom cell rendering for Side
        table.add_slot('body-cell-side', '''
            <q-td :props="props">
                <q-badge :color="props.row.side === 'LONG' ? 'green' : props.row.side === 'SHORT' ? 'red' : 'grey'" :label="props.row.side" />
            </q-td>
        ''')

        table.add_slot('body-cell-venue', '''
            <q-td :props="props">
                <q-badge :color="props.row.venue === 'THALEX' ? 'purple' : 'blue'" :label="props.row.venue" />
            </q-td>
        ''')

        table.add_slot('body-cell-opened_by', '''
            <q-td :props="props">
                <q-badge :color="props.row.opened_by === 'AI' ? 'positive' : 'grey-7'" :label="props.row.opened_by" />
            </q-td>
        ''')

        table.add_slot('body-cell-quantity', '''
            <q-td :props="props">
                <span>{{ props.row.quantity !== null ? props.row.quantity.toFixed(4) : '-' }}</span>
            </q-td>
        ''')

        table.add_slot('body-cell-entry_price', '''
            <q-td :props="props">
                <span v-if="props.row.entry_price !== null">${{ props.row.entry_price.toFixed(2) }}</span>
                <span v-else class="text-gray-500">-</span>
            </q-td>
        ''')

        table.add_slot('body-cell-current_price', '''
            <q-td :props="props">
                <span v-if="props.row.current_price !== null">${{ props.row.current_price.toFixed(2) }}</span>
                <span v-else class="text-gray-500">-</span>
            </q-td>
        ''')

        table.add_slot('body-cell-liquidation_price', '''
            <q-td :props="props">
                <span v-if="props.row.liquidation_price !== null">${{ props.row.liquidation_price.toFixed(2) }}</span>
                <span v-else class="text-gray-500">-</span>
            </q-td>
        ''')

        table.add_slot('body-cell-leverage', '''
            <q-td :props="props">
                <span>{{ props.row.leverage !== null ? Number(props.row.leverage).toFixed(1) + 'x' : '-' }}</span>
            </q-td>
        ''')
        
        # Custom cell rendering for Actions
        table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn flat dense icon="show_chart" color="blue" size="sm" @click="$parent.$emit('chart', props.row)">
                    <q-tooltip>View Chart</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.closable" flat dense icon="close" color="red" size="sm" @click="$parent.$emit('close', props.row)">
                    <q-tooltip>Close Position</q-tooltip>
                </q-btn>
            </q-td>
        ''')
        
        # Chart dialog
        chart_dialog = ui.dialog()
        with chart_dialog, ui.card().classes('w-96'):
            dialog_title = ui.label('').classes('text-xl font-bold mb-4')
            dialog_content = ui.label('').classes('text-gray-400 whitespace-pre-wrap')
            ui.button('Close', on_click=chart_dialog.close).classes('mt-4')
        
        # Close confirmation dialog
        close_dialog = ui.dialog()
        with close_dialog, ui.card().classes('w-96'):
            close_title = ui.label('').classes('text-xl font-bold mb-4')
            close_message = ui.label('').classes('text-gray-300 mb-4')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=close_dialog.close).classes('bg-gray-600')
                close_confirm_btn = ui.button('Close Position', on_click=lambda: None).classes('bg-red-600')
        
        # Event handlers
        current_position = {'target': None, 'label': None}
        
        def show_chart(e):
            """Show current position snapshot dialog."""
            position = e.args
            dialog_title.text = f"{position['symbol']} Snapshot"
            dialog_content.text = _format_position_snapshot(position)
            chart_dialog.open()
        
        async def show_close_dialog(e):
            """Show close confirmation dialog"""
            position = e.args
            current_position['target'] = position.get('asset') or position.get('symbol')
            current_position['label'] = position.get('symbol')
            close_title.text = f"Close {position['symbol']} Position?"
            close_message.text = f"Are you sure you want to close your {position['side']} position in {position['symbol']}?\n\nCurrent PnL: ${position['unrealized_pnl']:.2f} ({position['pnl_pct']:+.2f}%)"
            close_dialog.open()

        async def confirm_close():
            """Confirm position closing"""
            if not _ui_ok():
                return
            target = current_position['target']
            label = current_position['label'] or target
            if target:
                try:
                    close_dialog.close()
                    ui.notify(f'Closing {label} position...', type='info')
                    
                    success = await bot_service.close_position(target)
                    
                    if success:
                        ui.notify(f'Successfully closed {label} position!', type='positive')
                    else:
                        ui.notify(f'Failed to close {label} position', type='negative')
                except Exception as e:
                    if _ui_ok():
                        ui.notify(f'Error closing position: {str(e)}', type='negative')
        
        # Wire up event handlers
        table.on('chart', show_chart)
        table.on('close', show_close_dialog)
        close_confirm_btn.on('click', confirm_close)
    
    # Empty state message
    empty_message = ui.label('No active positions').classes('text-center text-gray-500 text-lg mt-8')
    empty_message.visible = True
    
    # Update function
    async def update_positions():
        """Update positions table with latest data"""
        if not _ui_ok():
            return
        try:
            state = state_manager.get_state()
            positions = state.positions or []
            
            # Show/hide empty message
            empty_message.visible = len(positions) == 0
            table.visible = len(positions) > 0
            
            if positions:
                # Calculate summary metrics
                total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
                total_notional = sum(
                    abs((p.get('quantity', 0) or 0) * ((p.get('current_price') if p.get('current_price') not in (None, 0) else p.get('entry_price')) or 0))
                    for p in positions
                )
                
                # Update summary cards
                positions_count.text = str(len(positions))
                total_pnl.text = f"${total_unrealized_pnl:+,.2f}"
                if total_unrealized_pnl >= 0:
                    total_pnl.classes(remove='text-red-500', add='text-green-500')
                else:
                    total_pnl.classes(remove='text-green-500', add='text-red-500')
                
                total_exposure.text = f"${total_notional:,.2f}"
                
                # Format positions for table
                rows = []
                for pos in positions:
                    quantity = pos.get('quantity', 0)
                    entry_price = pos.get('entry_price') if pos.get('entry_price') not in (None, 0) else None
                    current_price = pos.get('current_price') if pos.get('current_price') not in (None, 0) else None
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                    
                    # Calculate PnL percentage
                    pnl_pct = 0.0
                    if entry_price and current_price and entry_price > 0:
                        if quantity > 0:  # LONG
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        elif quantity < 0:  # SHORT
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    rows.append({
                        'row_id': pos.get('row_id') or f"{pos.get('venue', 'hyperliquid')}:{pos.get('instrument_name') or pos.get('symbol')}",
                        'symbol': pos.get('symbol', ''),
                        'venue': str(pos.get('venue', 'hyperliquid')).upper(),
                        'opened_by': pos.get('opened_by', 'External'),
                        'side': 'LONG' if quantity > 0 else 'SHORT' if quantity < 0 else 'FLAT',
                        'quantity': abs(quantity),
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct,
                        'leverage': pos.get('leverage'),
                        'liquidation_price': pos.get('liquidation_price') if pos.get('liquidation_price') not in (None, 0) else None,
                        'asset': pos.get('asset') or pos.get('symbol', ''),
                        'closable': bool(pos.get('closable', True)),
                    })
                
                table.rows = rows
                table.update()
            else:
                # Clear summary when no positions
                positions_count.text = '0'
                total_pnl.text = '$0.00'
                total_exposure.text = '$0.00'
                total_pnl.classes(remove='text-green-500 text-red-500', add='text-white')
        
        except Exception as e:
            if _ui_ok():
                ui.notify(f'Error updating positions: {str(e)}', type='warning')
    
    # Auto-refresh every 2 seconds
    ui.timer(2.0, update_positions)
    
    # Initial update
    # Note: Can't await in sync context, timer will handle it
