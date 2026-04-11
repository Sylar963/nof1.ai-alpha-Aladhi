"""Two-cadence scheduler for the options pipeline.

Runs two independent background tasks alongside the main perps loop:

- **vol surface refresh** (default 15 minutes / 900s) — pulls fresh Thalex
  + Deribit data, rebuilds the vol surface, persists today's anchor.
- **options decision** (default 3 hours / 10800s) — calls the options
  agent, parses decisions, hands them to the executor.

Both loops share the same machinery: a configurable interval, a
fire-and-keep-running error policy (a callback exception is logged but the
loop keeps ticking), and clean cancel-on-stop semantics.

The scheduler doesn't *do* anything itself — it's a pure scheduling shell.
The bot engine injects the actual callbacks (typically wrappers around
``options_intel.builder.build_options_context`` and
``OptionsAgent.decide``). That keeps this module trivially testable with
fake callbacks and no network IO.

Setting either interval to ``0`` disables that loop entirely. Useful when
the operator wants the surface refresh running but isn't ready to let the
LLM emit decisions yet.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


logger = logging.getLogger(__name__)


_DEFAULT_VOL_SURFACE_INTERVAL_SECONDS: float = 900.0   # 15 minutes
_DEFAULT_OPTIONS_DECISION_INTERVAL_SECONDS: float = 10800.0  # 3 hours


@dataclass
class OptionsSchedulerConfig:
    """Cadence config for the two background tasks.

    Setting either interval to ``0`` disables that loop entirely.
    """

    vol_surface_interval_seconds: float = _DEFAULT_VOL_SURFACE_INTERVAL_SECONDS
    options_decision_interval_seconds: float = _DEFAULT_OPTIONS_DECISION_INTERVAL_SECONDS


AsyncCallable = Callable[[], Awaitable[None]]


class OptionsScheduler:
    """Pure scheduling shell for the options pipeline."""

    def __init__(
        self,
        config: OptionsSchedulerConfig,
        refresh_vol_surface: AsyncCallable,
        run_options_decision: AsyncCallable,
    ) -> None:
        self.config = config
        self._refresh_vol_surface = refresh_vol_surface
        self._run_options_decision = run_options_decision
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Spawn the background tasks."""
        if self._running:
            return
        self._running = True

        if self.config.vol_surface_interval_seconds > 0:
            self._tasks.append(
                asyncio.create_task(
                    self._loop(
                        "vol_surface",
                        self.config.vol_surface_interval_seconds,
                        self._refresh_vol_surface,
                    )
                )
            )
        if self.config.options_decision_interval_seconds > 0:
            self._tasks.append(
                asyncio.create_task(
                    self._loop(
                        "options_decision",
                        self.config.options_decision_interval_seconds,
                        self._run_options_decision,
                    )
                )
            )

    async def stop(self) -> None:
        """Cancel all background tasks and wait for them to wind down."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                pass
        self._tasks.clear()

    async def _loop(self, name: str, interval: float, callback: AsyncCallable) -> None:
        """Generic interval loop with swallow-and-log error semantics."""
        try:
            while self._running:
                try:
                    await callback()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("OptionsScheduler[%s] callback failed: %s", name, exc)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
