"""Tests for RendererRegistry."""

import pytest

from nautilus_quants.backtest.protocols import BaseMetadataRenderer
from nautilus_quants.backtest.registry import RendererRegistry
from nautilus_quants.strategies.cross_sectional.metadata import (  # noqa: F401 - triggers self-registration
    CrossSectionalMetadataRenderer,
)


class TestRendererRegistry:
    """Tests for RendererRegistry."""

    def test_get_none_returns_base(self):
        """Test getting None returns BaseMetadataRenderer."""
        renderer = RendererRegistry.get(None)
        assert isinstance(renderer, BaseMetadataRenderer)

    def test_get_base_returns_base(self):
        """Test getting 'base' returns BaseMetadataRenderer."""
        renderer = RendererRegistry.get("base")
        assert isinstance(renderer, BaseMetadataRenderer)

    def test_get_cross_sectional(self):
        """Test getting cross_sectional renderer."""
        renderer = RendererRegistry.get("cross_sectional")
        assert isinstance(renderer, CrossSectionalMetadataRenderer)

    def test_get_unknown_raises_value_error(self):
        """Test unknown renderer name raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            RendererRegistry.get("nonexistent_renderer")

        assert "Unknown metadata_renderer: 'nonexistent_renderer'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_list_available_includes_base(self):
        """Test list_available always includes 'base'."""
        available = RendererRegistry.list_available()
        assert "base" in available

    def test_list_available_includes_cross_sectional(self):
        """Test list_available includes registered renderers."""
        available = RendererRegistry.list_available()
        assert "cross_sectional" in available

    def test_register_custom_renderer(self):
        """Test registering a custom renderer."""

        class CustomRenderer:
            def get_column_config(self):
                return []

            def render_position(self, symbol, position_info, metadata, timestamp_ns):
                return {"custom": True}

        # Register
        RendererRegistry.register("custom_test", CustomRenderer)

        try:
            # Verify it's available
            assert "custom_test" in RendererRegistry.list_available()

            # Verify we can get it
            renderer = RendererRegistry.get("custom_test")
            assert isinstance(renderer, CustomRenderer)
        finally:
            # Cleanup: remove from registry
            del RendererRegistry._renderers["custom_test"]
