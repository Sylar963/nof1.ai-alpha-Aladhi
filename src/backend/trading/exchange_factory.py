"""Venue dispatcher for exchange adapters.

The factory is the only place that knows the full set of supported venues. The
bot engine and decision pipeline reference adapters via :meth:`create` and never
import a venue-specific class directly. New venues are added by registering them
here (and importing this module so the registration runs).
"""

from __future__ import annotations

from typing import Type

from src.backend.trading.exchange_adapter import ExchangeAdapter


class ExchangeFactory:
    """Registry + constructor for ExchangeAdapter implementations."""

    _registry: dict[str, Type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, venue: str, adapter_cls: Type[ExchangeAdapter]) -> None:
        """Register an adapter class under a venue name (case-insensitive)."""
        cls._registry[venue.lower()] = adapter_cls

    @classmethod
    def create(cls, venue: str, **kwargs) -> ExchangeAdapter:
        """Instantiate the adapter registered for ``venue``.

        Raises:
            ValueError: If no adapter is registered for the given venue name.
        """
        key = venue.lower()
        adapter_cls = cls._registry.get(key)
        if adapter_cls is None:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise ValueError(f"Unknown venue: {venue!r}. Available: {available}")
        return adapter_cls(**kwargs)

    @classmethod
    def available_venues(cls) -> list[str]:
        """Return the list of registered venue names — useful for diagnostics and UI."""
        return sorted(cls._registry.keys())


def _register_default_adapters() -> None:
    """Register the venues shipped with the project.

    Imports happen inside the function to avoid circular imports at module load time
    and to keep the factory importable even if a particular adapter's optional
    dependencies are missing.
    """
    try:
        from src.backend.trading.hyperliquid_api import HyperliquidAPI

        ExchangeFactory.register("hyperliquid", HyperliquidAPI)
    except ImportError:
        pass

    try:
        from src.backend.trading.thalex_api import ThalexAPI

        ExchangeFactory.register("thalex", ThalexAPI)
    except ImportError:
        pass


_register_default_adapters()
