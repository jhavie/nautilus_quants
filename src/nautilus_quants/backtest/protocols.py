# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Protocol definitions for backtest report generation.

This module defines abstract interfaces for strategy-specific metadata
handling, enabling loose coupling between strategies and ReportGenerator.

Constitution Compliance:
    - Principle V (Separation of Concerns): Decouples strategy from report logic
    - Principle IV (Type Safety): Uses Protocol for static typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    pass

# Cache key for strategy → report metadata transfer via engine.cache
POSITION_METADATA_CACHE_KEY = "position_metadata"

# Cache key for EquitySnapshotActor → ReportGenerator MTM equity transfer
EQUITY_SNAPSHOTS_CACHE_KEY = "equity_snapshots"


class ColumnConfig(TypedDict):
    """Configuration for a table column in position timeline.

    Attributes:
        key: Data field name in position dict
        label: Display text for table header
        format: Formatting type (text, number, rank_pair, composite_pair, side, datetime)
        sort_key: Field name for sorting (optional, uses key if not specified)
    """

    key: str
    label: str
    format: str
    sort_key: str | None


@runtime_checkable
class PositionMetadataProvider(Protocol):
    """Protocol for strategies to provide position metadata.

    Strategies implement this to store and serialize their position metadata
    in a format-agnostic way. The metadata is later consumed by ReportGenerator.
    """

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all position metadata.

        Returns:
            Dict mapping position key (e.g., instrument_id) to metadata dict.
            The metadata structure is strategy-specific.
        """
        ...

    def serialize(self) -> bytes:
        """Serialize all metadata to bytes for cache storage.

        Returns:
            Pickled or otherwise serialized metadata bytes.
        """
        ...


@runtime_checkable
class MetadataRenderer(Protocol):
    """Protocol for rendering strategy-specific metadata to HTML.

    Each strategy provides its own renderer that knows how to:
    1. Extract relevant fields from position metadata
    2. Format fields for display
    3. Define table column configuration
    """

    def get_column_config(self) -> list[ColumnConfig]:
        """Get column configuration for position table.

        Returns:
            List of ColumnConfig defining table columns.
            Order determines display order.
        """
        ...

    def render_position(
        self,
        symbol: str,
        position_info: dict[str, Any],
        metadata: dict[str, Any] | None,
        timestamp_ns: int,
    ) -> dict[str, Any]:
        """Render a single position's data for the timeline table.

        Args:
            symbol: Position symbol (e.g., "BTCUSDT")
            position_info: Basic position info (value, side, ts_opened, etc.)
            metadata: Strategy-specific metadata (rank_history, entry_rank, etc.)
            timestamp_ns: Current timeline timestamp in nanoseconds

        Returns:
            Dict with keys matching ColumnConfig.key values, ready for display.
        """
        ...


class BaseMetadataRenderer:
    """Default renderer for strategies without specific metadata.

    Provides basic position info (symbol, side, value, ts_opened) without
    any strategy-specific fields like rank or composite values.
    """

    def get_column_config(self) -> list[ColumnConfig]:
        """Get basic column configuration."""
        return [
            {"key": "symbol", "label": "Symbol", "format": "text", "sort_key": None},
            {"key": "side", "label": "Side", "format": "side", "sort_key": None},
            {"key": "ts_opened", "label": "Opened", "format": "datetime", "sort_key": "ts_opened"},
            {"key": "value", "label": "Value (USDT)", "format": "number", "sort_key": "value"},
        ]

    def render_position(
        self,
        symbol: str,
        position_info: dict[str, Any],
        metadata: dict[str, Any] | None,
        timestamp_ns: int,
    ) -> dict[str, Any]:
        """Render basic position data."""
        return {
            "symbol": symbol,
            "side": position_info.get("side", ""),
            "ts_opened": position_info.get("ts_opened"),
            "value": position_info.get("value", 0.0),
        }
