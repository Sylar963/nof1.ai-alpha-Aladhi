"""Two-cadence scheduler for the options pipeline.

Runs two independent background tasks alongside the main perps loop:

- **vol surface refresh** (default 15 minutes / 900s) — pulls fresh Thalex
  + Deribit data, rebuilds the vol surface, persists today's anchor.
- **options decision** (default 3 hours / 10800s) — calls the options
  agent, parses decisions, hands them to the executor.

On startup, the scheduler fetches the vol surface **first**, then fires the
initial options decision immediately after — so the very first decision
always has fresh context.  After that bootstrap, both loops continue on
their own independent cadences.

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
        self._bootstrap_done = asyncio.Event()

    async def start(self) -> None:
        """Spawn the background tasks.

        A dedicated bootstrap task runs first: it fetches the vol surface
        once, then fires the first options decision, guaranteeing the
        decision always has context.  After that, the two independent
        cadence loops take over.
        """
        if self._running:
            return
        self._running = True
        self._bootstrap_done.clear()

        self._tasks.append(asyncio.create_task(self._bootstrap()))

        if self.config.vol_surface_interval_seconds > 0:
            self._tasks.append(
                asyncio.create_task(
                    self._loop(
                        "vol_surface",
                        self.config.vol_surface_interval_seconds,
                        self._refresh_vol_surface,
                        wait_for_bootstrap=False,
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
                        wait_for_bootstrap=True,
                    )
                )
            )

    # Vol surface: HTTP fetches + math, normally ~30s. Generous 5-min cap.
    _VOL_SURFACE_TIMEOUT: float = 300.0
    # Options decision: LLM call with verbose reasoning. glm-5.1 with reasoning
    # enabled has been observed taking up to 10 minutes per cycle. Cap at 15
    # min so a hung request doesn't block the scheduler forever.
    _DECISION_TIMEOUT: float = 900.0

    async def _bootstrap(self) -> None:
        """Fetch vol surface then fire the first options decision sequentially.

        The initial decision is gated on a successful vol surface refresh —
        running the LLM against a missing / stale surface would either feed
        it the previous cycle's data (misleading) or no surface at all
        (forces a noop). Either way the decision is worse than just waiting
        for the next cadence tick, so skip it on bootstrap surface failure.
        """
        vol_ok = False
        try:
            if self.config.vol_surface_interval_seconds > 0:
                logger.info("OptionsScheduler: bootstrap — fetching initial vol surface")
                try:
                    await asyncio.wait_for(
                        self._refresh_vol_surface(), timeout=self._VOL_SURFACE_TIMEOUT,
                    )
                    vol_ok = True
                except asyncio.TimeoutError:
                    logger.error(
                        "OptionsScheduler: bootstrap vol surface timed out after %.0fs",
                        self._VOL_SURFACE_TIMEOUT,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("OptionsScheduler: bootstrap vol surface failed: %s", exc)
            else:
                # No surface cadence configured — nothing to gate on. The
                # decision path is responsible for reading whatever surface
                # state exists.
                vol_ok = True

            if self.config.options_decision_interval_seconds > 0:
                if not vol_ok:
                    logger.warning(
                        "OptionsScheduler: skipping initial options decision — "
                        "bootstrap vol surface refresh did not succeed"
                    )
                else:
                    logger.info("OptionsScheduler: bootstrap — running initial options decision")
                    try:
                        await asyncio.wait_for(
                            self._run_options_decision(), timeout=self._DECISION_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "OptionsScheduler: bootstrap options decision timed out after %.0fs",
                            self._DECISION_TIMEOUT,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error("OptionsScheduler: bootstrap options decision failed: %s", exc)
        finally:
            self._bootstrap_done.set()

    async def stop(self) -> None:
        """Cancel all background tasks and wait for them to wind down."""
        self._running = False
        self._bootstrap_done.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                pass
        self._tasks.clear()

    _MAX_CONSECUTIVE_ERRORS = 5

    # Per-loop callback timeouts. Vol surface is HTTP+math (fast), options
    # decision is an LLM call with verbose reasoning (slow — see _bootstrap
    # docstring above). Looked up from the loop name.
    _CALLBACK_TIMEOUTS: dict[str, float] = {
        "vol_surface": _VOL_SURFACE_TIMEOUT,
        "options_decision": _DECISION_TIMEOUT,
    }

    async def _loop(
        self,
        name: str,
        interval: float,
        callback: AsyncCallable,
        wait_for_bootstrap: bool = False,
    ) -> None:
        """Generic interval loop with swallow-and-log error semantics.

        When *wait_for_bootstrap* is True the loop skips its first immediate
        execution and sleeps first — the bootstrap task already ran it.
        When False the loop sleeps first anyway since bootstrap handles the
        initial invocation.

        After ``_MAX_CONSECUTIVE_ERRORS`` failures in a row the loop stops
        itself to avoid silent infinite retries.
        """
        consecutive_errors = 0
        callback_timeout = self._CALLBACK_TIMEOUTS.get(name, self._DECISION_TIMEOUT)
        try:
            if wait_for_bootstrap:
                await self._bootstrap_done.wait()

            # Bootstrap already ran both callbacks once, so sleep first.
            await asyncio.sleep(interval)

            while self._running:
                try:
                    await asyncio.wait_for(callback(), timeout=callback_timeout)
                    consecutive_errors = 0
                except asyncio.TimeoutError:
                    consecutive_errors += 1
                    logger.error(
                        "OptionsScheduler[%s] callback timed out after %.0fs (%d/%d)",
                        name, callback_timeout, consecutive_errors, self._MAX_CONSECUTIVE_ERRORS,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    consecutive_errors += 1
                    logger.error(
                        "OptionsScheduler[%s] callback failed (%d/%d): %s",
                        name, consecutive_errors, self._MAX_CONSECUTIVE_ERRORS, exc,
                    )
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "OptionsScheduler[%s] stopping after %d consecutive failures",
                        name, consecutive_errors,
                    )
                    break
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
