"""Metadata renderer registry for configuration-driven injection.

This module provides a registry pattern for MetadataRenderer implementations,
allowing strategies to define custom position metadata rendering without
coupling the runner to specific strategy types.

Usage:
    renderer = RendererRegistry.get("cross_sectional")
    position_data = renderer.render_position(symbol, position_info, metadata, ts_ns)
"""

from typing import Callable

from nautilus_quants.backtest.protocols import BaseMetadataRenderer, MetadataRenderer


# Type alias for renderer factory
RendererFactory = Callable[[], MetadataRenderer]


class RendererRegistry:
    """Registry for MetadataRenderer implementations.

    Provides configuration-driven renderer lookup by name.

    Example:
        # Get renderer by name from config
        renderer = RendererRegistry.get("cross_sectional")

        # List available renderers
        available = RendererRegistry.list_available()
        # Returns: ["base", "cross_sectional"]
    """

    _renderers: dict[str, RendererFactory] = {}

    @classmethod
    def register(cls, name: str, factory: RendererFactory) -> None:
        """Register a renderer factory by name.

        Args:
            name: Unique renderer name (e.g., "cross_sectional")
            factory: Callable that returns a MetadataRenderer instance
        """
        cls._renderers[name] = factory

    @classmethod
    def get(cls, name: str | None) -> MetadataRenderer:
        """Get renderer instance by name.

        Args:
            name: Renderer name, or None for default

        Returns:
            MetadataRenderer instance

        Raises:
            ValueError: If name not registered
        """
        if name is None or name == "base":
            return BaseMetadataRenderer()

        if name not in cls._renderers:
            available = ", ".join(sorted(cls._renderers.keys()))
            raise ValueError(
                f"Unknown metadata_renderer: '{name}'. "
                f"Available: {available or 'none (use base)'}"
            )

        return cls._renderers[name]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered renderer names.

        Returns:
            List of renderer names, always includes "base"
        """
        return ["base"] + sorted(cls._renderers.keys())


