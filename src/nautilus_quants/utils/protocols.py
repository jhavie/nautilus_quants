# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Protocol definitions for strategy metadata rendering.

Defines abstract interfaces for strategy-specific metadata handling,
enabling loose coupling between strategies and ReportGenerator.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


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
    """Protocol for strategies to provide position metadata."""

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        ...

    def serialize(self) -> bytes:
        ...


@runtime_checkable
class MetadataRenderer(Protocol):
    """Protocol for rendering strategy-specific metadata to HTML."""

    def get_column_config(self) -> list[ColumnConfig]:
        ...

    def render_position(
        self,
        symbol: str,
        position_info: dict[str, Any],
        metadata: dict[str, Any] | None,
        timestamp_ns: int,
    ) -> dict[str, Any]:
        ...


class BaseMetadataRenderer:
    """Default renderer for strategies without specific metadata."""

    def get_column_config(self) -> list[ColumnConfig]:
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
        return {
            "symbol": symbol,
            "side": position_info.get("side", ""),
            "ts_opened": position_info.get("ts_opened"),
            "value": position_info.get("value", 0.0),
        }
