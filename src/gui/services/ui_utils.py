"""
UI Utilities - Helpers for managing NiceGUI client state and context
"""

from typing import Any

from nicegui import ui

_UNSET = object()
_NOTIFY_ARG_MAP = {
    'close_button': 'closeBtn',
    'multi_line': 'multiLine',
}


def is_ui_alive(element: ui.element) -> bool:
    """
    Check if a NiceGUI element is still attached to a valid client.
    Returns False if the element or its parent has been deleted (common in SPAs).
    """
    try:
        if element is None or getattr(element, 'is_deleted', False):
            return False
        # Accessing .client / .parent_slot will raise once NiceGUI has torn
        # down the page tree for this element.
        _ = element.client
        _ = element.parent_slot
        return True
    except (RuntimeError, AttributeError):
        return False


def safe_notify(client: Any, message: object, **kwargs: Any) -> bool:
    """Send a NiceGUI notification through a captured client.

    Async callbacks can outlive the slot/context they were created in after the
    user navigates away. ``ui.notify(...)`` resolves through that ambient slot
    context and crashes in that situation. Sending through the captured client
    avoids the dead-slot lookup while still no-oping safely once the client is
    gone.
    """
    if client is None:
        return False
    try:
        if getattr(client, '_deleted', False) or not getattr(client, 'has_socket_connection', False):
            return False
        options = {
            _NOTIFY_ARG_MAP.get(key, key): value
            for key, value in kwargs.items()
            if value is not None
        }
        options['message'] = str(message)
        client.outbox.enqueue_message('notify', options, client.id)
        return True
    except (RuntimeError, AttributeError):
        return False


class RenderGate:
    """Skip identical-payload re-renders on timer-driven updates.

    Pages poll state every 1-5s but the bot only emits new market data on
    its 5-minute trading cycle. Without a gate, the heavier renderers
    (Plotly figures, big tables) re-serialize the same payload to the
    browser dozens of times per real update. Build a cheap tuple of the
    fields the page actually displays and pass it to ``changed()``.
    ``True`` means render; ``False`` means skip.
    """

    __slots__ = ("_last",)

    def __init__(self) -> None:
        self._last = _UNSET

    def changed(self, signature) -> bool:
        if signature == self._last:
            return False
        self._last = signature
        return True

    def reset(self) -> None:
        self._last = _UNSET
