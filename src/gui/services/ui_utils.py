"""
UI Utilities - Helpers for managing NiceGUI client state and context
"""

from nicegui import ui

_UNSET = object()


def is_ui_alive(element: ui.element) -> bool:
    """
    Check if a NiceGUI element is still attached to a valid client.
    Returns False if the element or its parent has been deleted (common in SPAs).
    """
    try:
        if element is None:
            return False
        # Accessing .client will raise RuntimeError if the element is deleted
        _ = element.client
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
