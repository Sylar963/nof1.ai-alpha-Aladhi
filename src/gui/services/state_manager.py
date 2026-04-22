"""
State Manager - Global reactive state management for UI

A single shared StateManager is intentional: there is one bot, so every
connected client observes the same state snapshot. Per-client isolation
would be wrong here (clients would each see a different, stale bot).

What IS guarded: writers. The bot engine callback runs on the UI event
loop, but the application also performs reads/writes from ``to_thread``
callbacks (e.g. trade executions) — serialising writes with a lock
prevents torn reads if those ever dispatch an update from a worker
thread. Observer callbacks run outside the lock so slow observers can't
block state progression.
"""

import threading
from typing import Optional
from src.backend.bot_engine import BotState


class StateManager:
    """Manages global application state for UI components"""

    def __init__(self):
        self._state: BotState = BotState()
        self._observers = []
        self._lock = threading.Lock()

    def update(self, new_state: BotState):
        """
        Update state with new data from bot engine.
        Called by bot_service when bot state changes.
        """
        with self._lock:
            self._state = new_state
            observers = list(self._observers)
        for observer in observers:
            try:
                observer(new_state)
            except Exception:
                pass

    def get_state(self) -> BotState:
        """Get current application state"""
        # Snapshot assignment is atomic under the GIL; the lock only
        # matters for observer bookkeeping consistency.
        return self._state

    def subscribe(self, callback):
        """Subscribe to state changes (future enhancement)"""
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from state changes"""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)
