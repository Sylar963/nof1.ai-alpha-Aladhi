"""
UI Utilities - Helpers for managing NiceGUI client state and context
"""

from nicegui import ui

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
