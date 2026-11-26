"""Open the standalone canvas replay from the main desktop app."""

from nicegui import ui


def create_replay():
    with ui.row().classes('w-full items-center justify-between'):
        ui.label('Arena Replay').classes('text-xl font-bold')
        ui.link('Open full screen ↗', 'http://127.0.0.1:8081', new_tab=True)
    ui.label('Start the replay service with: venv/bin/python scripts/replay_paradex.py').classes('text-xs text-gray-500')
    ui.element('iframe').props('src="http://127.0.0.1:8081" title="Arena Replay"').classes('w-full border-0').style('height: calc(100vh - 180px)')
