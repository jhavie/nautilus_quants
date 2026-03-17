# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Metadata provider and renderer for CrossSectionalFactorStrategy.

This module implements the Protocol interfaces for handling position metadata
specific to the cross-sectional factor strategy, including rank tracking
and composite value history.

Constitution Compliance:
    - Principle V (Separation of Concerns): Separates metadata logic from strategy
    - Principle IV (Type Safety): Implements Protocol interfaces
"""

from __future__ import annotations

import pickle
from typing import Any

from nautilus_quants.utils.protocols import ColumnConfig
from nautilus_quants.utils.registry import RendererRegistry


class CrossSectionalMetadataProvider:
    """Manages position metadata for CrossSectionalFactorStrategy.

    Tracks entry rank/composite, rank history, and close reasons for each position.
    Implements PositionMetadataProvider protocol.
    """

    def __init__(self) -> None:
        """Initialize the metadata provider."""
        # Active positions: inst_id -> metadata
        self._metadata: dict[str, dict[str, Any]] = {}
        # Closed positions: unique_key -> metadata
        self._closed_metadata: dict[str, dict[str, Any]] = {}

    def record_open(
        self,
        instrument_id: str,
        side: str,
        rank: int | None,
        composite: float | None,
        buffer_ratio: float,
        ts_event: int,
    ) -> None:
        """Record metadata when opening a position.

        Args:
            instrument_id: Instrument identifier string
            side: Position side ("LONG" or "SHORT")
            rank: Entry rank in cross-sectional ranking
            composite: Entry composite factor value
            buffer_ratio: Buffer ratio from strategy config
            ts_event: Timestamp in nanoseconds
        """
        self._metadata[instrument_id] = {
            "side": side,
            "entry_rank": rank,
            "entry_composite": composite,
            "buffer_ratio": buffer_ratio,
            "ts_opened": ts_event,
            "rank_history": [],
        }

    def update_rank_history(
        self,
        instrument_ids: list[str],
        rank_lookup: dict[str, int],
        composite_lookup: dict[str, float],
        ts_event: int,
    ) -> None:
        """Update rank history for all specified positions.

        Args:
            instrument_ids: List of instrument IDs to update
            rank_lookup: Current rank for each instrument
            composite_lookup: Current composite value for each instrument
            ts_event: Timestamp in nanoseconds
        """
        for inst_id in instrument_ids:
            if inst_id in self._metadata:
                self._metadata[inst_id]["rank_history"].append({
                    "ts_event": ts_event,
                    "rank": rank_lookup.get(inst_id, -1),
                    "composite": composite_lookup.get(inst_id),
                })

    def record_close(
        self,
        instrument_id: str,
        reason: str,
        hour_count: int,
    ) -> None:
        """Record metadata when closing a position.

        Args:
            instrument_id: Instrument identifier string
            reason: Close reason string
            hour_count: Current hour count (for unique key generation)
        """
        if instrument_id in self._metadata:
            metadata = self._metadata.pop(instrument_id)
            metadata["close_reason"] = reason
            # Use unique key for closed positions (may have multiple per instrument)
            close_key = f"{instrument_id}:{hour_count}"
            self._closed_metadata[close_key] = metadata

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all position metadata (open and closed).

        Returns:
            Merged dict of all metadata.
        """
        return {
            **self._metadata,
            **self._closed_metadata,
        }

    def serialize(self) -> bytes:
        """Serialize all metadata to bytes.

        Returns:
            Pickled metadata bytes.
        """
        return pickle.dumps(self.get_all_metadata())


@RendererRegistry.register_as("cross_sectional")
class CrossSectionalMetadataRenderer:
    """Renders CrossSectional strategy metadata for HTML reports.

    Implements MetadataRenderer protocol to format rank and composite
    value displays for the position timeline table.
    """

    def get_column_config(self) -> list[ColumnConfig]:
        """Get column configuration for cross-sectional strategy.

        Returns columns for rank, composite, symbol, side, opened time, and value.
        """
        return [
            {
                "key": "rank_display",
                "label": "Rank (Cur/Entry)",
                "format": "text",
                "sort_key": "current_rank",
            },
            {
                "key": "composite_display",
                "label": "Composite (Cur/Entry)",
                "format": "text",
                "sort_key": "current_composite",
            },
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
        """Render position data with rank and composite values.

        Args:
            symbol: Position symbol
            position_info: Basic position info
            metadata: Strategy-specific metadata with rank_history
            timestamp_ns: Current timeline timestamp

        Returns:
            Dict with rendered values including rank_display and composite_display.
        """
        # Extract entry values from metadata
        entry_rank = None
        entry_composite = None
        current_rank = None
        current_composite = None

        if metadata:
            entry_rank = metadata.get("entry_rank")
            entry_composite = metadata.get("entry_composite")
            # Default current to entry
            current_rank = entry_rank
            current_composite = entry_composite

            # Find most recent rank_history entry <= timestamp
            for history_entry in metadata.get("rank_history", []):
                if history_entry.get("ts_event", 0) <= timestamp_ns:
                    current_rank = history_entry.get("rank", current_rank)
                    current_composite = history_entry.get("composite", current_composite)
                else:
                    break

        # Format rank display: current(entry)
        entry_rank_str = str(entry_rank) if entry_rank is not None else "-"
        current_rank_str = str(current_rank) if current_rank is not None else "-"
        if current_rank_str != "-" or entry_rank_str != "-":
            rank_display = f"{current_rank_str}({entry_rank_str})"
        else:
            rank_display = "-"

        # Format composite display: current(entry)
        entry_comp_str = f"{entry_composite:.2f}" if entry_composite is not None else "-"
        current_comp_str = f"{current_composite:.2f}" if current_composite is not None else "-"
        if current_comp_str != "-" or entry_comp_str != "-":
            composite_display = f"{current_comp_str}({entry_comp_str})"
        else:
            composite_display = "-"

        return {
            "symbol": symbol,
            "side": position_info.get("side", ""),
            "ts_opened": position_info.get("ts_opened"),
            "value": position_info.get("value", 0.0),
            "rank_display": rank_display,
            "composite_display": composite_display,
            "current_rank": current_rank,
            "current_composite": current_composite,
            "entry_rank": entry_rank,
            "entry_composite": entry_composite,
        }
