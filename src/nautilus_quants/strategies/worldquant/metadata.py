# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Metadata provider and renderer for WorldQuantAlphaStrategy.

This module implements the Protocol interfaces for handling position metadata
specific to the WorldQuant alpha strategy, including alpha101 value tracking
and final weight history.
"""

from __future__ import annotations

import pickle
from typing import Any

from nautilus_quants.utils.protocols import ColumnConfig
from nautilus_quants.utils.registry import RendererRegistry


class WorldQuantMetadataProvider:
    """Manages position metadata for WorldQuantAlphaStrategy.

    Tracks entry alpha101/weight, alpha101 history, and close reasons for each position.
    Implements PositionMetadataProvider protocol.
    """

    def __init__(self) -> None:
        """Initialize the metadata provider."""
        self._metadata: dict[str, dict[str, Any]] = {}
        self._closed_metadata: dict[str, dict[str, Any]] = {}

    def record_open(
        self,
        instrument_id: str,
        side: str,
        alpha101: float | None,
        weight: float | None,
        ts_event: int,
        neutralized: float | None = None,
        scaled: float | None = None,
        decayed: float | None = None,
    ) -> None:
        """Record metadata when opening a position.

        Args:
            instrument_id: Instrument identifier string
            side: Position side ("LONG" or "SHORT")
            alpha101: Entry alpha101 rank value (0~1)
            weight: Entry final weight (positive=long, negative=short)
            ts_event: Timestamp in nanoseconds
            neutralized: Entry value after market neutralization (step 3)
            scaled: Entry value after scaling (step 4)
            decayed: Entry value after decay+rescale (step 5)
        """
        self._metadata[instrument_id] = {
            "side": side,
            "entry_alpha101": alpha101,
            "entry_neutralized": neutralized,
            "entry_scaled": scaled,
            "entry_decayed": decayed,
            "entry_weight": weight,
            "ts_opened": ts_event,
            "alpha101_history": [],
        }

    def update_alpha101_history(
        self,
        instrument_ids: list[str],
        alpha101_lookup: dict[str, float],
        weight_lookup: dict[str, float],
        ts_event: int,
        neutralized_lookup: dict[str, float] | None = None,
        scaled_lookup: dict[str, float] | None = None,
        decayed_lookup: dict[str, float] | None = None,
    ) -> None:
        """Update alpha101 history for all specified positions.

        Args:
            instrument_ids: List of instrument IDs to update
            alpha101_lookup: Current alpha101 value for each instrument
            weight_lookup: Current final weight for each instrument
            ts_event: Timestamp in nanoseconds
            neutralized_lookup: Current value after neutralization (step 3)
            scaled_lookup: Current value after scaling (step 4)
            decayed_lookup: Current value after decay+rescale (step 5)
        """
        for inst_id in instrument_ids:
            if inst_id in self._metadata:
                self._metadata[inst_id]["alpha101_history"].append({
                    "ts_event": ts_event,
                    "alpha101": alpha101_lookup.get(inst_id),
                    "neutralized": neutralized_lookup.get(inst_id) if neutralized_lookup else None,
                    "scaled": scaled_lookup.get(inst_id) if scaled_lookup else None,
                    "decayed": decayed_lookup.get(inst_id) if decayed_lookup else None,
                    "weight": weight_lookup.get(inst_id),
                })

    def record_close(
        self,
        instrument_id: str,
        reason: str,
        signal_count: int,
    ) -> None:
        """Record metadata when closing a position.

        Args:
            instrument_id: Instrument identifier string
            reason: Close reason string
            signal_count: Current signal count (for unique key generation)
        """
        if instrument_id in self._metadata:
            metadata = self._metadata.pop(instrument_id)
            metadata["close_reason"] = reason
            close_key = f"{instrument_id}:{signal_count}"
            self._closed_metadata[close_key] = metadata

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all position metadata (open and closed)."""
        return {
            **self._metadata,
            **self._closed_metadata,
        }

    def serialize(self) -> bytes:
        """Serialize all metadata to bytes."""
        return pickle.dumps(self.get_all_metadata())


@RendererRegistry.register_as("worldquant")
class WorldQuantMetadataRenderer:
    """Renders WorldQuant strategy metadata for HTML reports.

    Implements MetadataRenderer protocol to format alpha101 and weight
    displays for the position timeline table.
    """

    def get_column_config(self) -> list[ColumnConfig]:
        """Get column configuration for WorldQuant strategy."""
        return [
            {"key": "alpha101_display",    "label": "Alpha101 (Cur/Entry)",  "format": "text", "sort_key": "current_alpha101"},
            {"key": "neutralized_display", "label": "Neutral (Cur/Entry)",   "format": "text", "sort_key": "current_neutralized"},
            {"key": "scaled_display",      "label": "Scaled (Cur/Entry)",    "format": "text", "sort_key": "current_scaled"},
            {"key": "decayed_display",     "label": "Decay (Cur/Entry)",     "format": "text", "sort_key": "current_decayed"},
            {"key": "weight_display",      "label": "Weight (Cur/Entry)",    "format": "text", "sort_key": "current_weight"},
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
        """Render position data with alpha101, pipeline intermediate, and weight values."""
        entry_alpha101 = None
        entry_neutralized = None
        entry_scaled = None
        entry_decayed = None
        entry_weight = None
        current_alpha101 = None
        current_neutralized = None
        current_scaled = None
        current_decayed = None
        current_weight = None

        if metadata:
            entry_alpha101 = metadata.get("entry_alpha101")
            entry_neutralized = metadata.get("entry_neutralized")
            entry_scaled = metadata.get("entry_scaled")
            entry_decayed = metadata.get("entry_decayed")
            entry_weight = metadata.get("entry_weight")
            current_alpha101 = entry_alpha101
            current_neutralized = entry_neutralized
            current_scaled = entry_scaled
            current_decayed = entry_decayed
            current_weight = entry_weight

            for history_entry in metadata.get("alpha101_history", []):
                if history_entry.get("ts_event", 0) <= timestamp_ns:
                    current_alpha101 = history_entry.get("alpha101", current_alpha101)
                    current_neutralized = history_entry.get("neutralized", current_neutralized)
                    current_scaled = history_entry.get("scaled", current_scaled)
                    current_decayed = history_entry.get("decayed", current_decayed)
                    current_weight = history_entry.get("weight", current_weight)
                else:
                    break

        def _fmt4(cur: float | None, entry: float | None) -> str:
            cur_s = f"{cur:.4f}" if cur is not None else "-"
            entry_s = f"{entry:.4f}" if entry is not None else "-"
            if cur_s != "-" or entry_s != "-":
                return f"{cur_s}({entry_s})"
            return "-"

        def _fmt3(cur: float | None, entry: float | None) -> str:
            cur_s = f"{cur:.3f}" if cur is not None else "-"
            entry_s = f"{entry:.3f}" if entry is not None else "-"
            if cur_s != "-" or entry_s != "-":
                return f"{cur_s}({entry_s})"
            return "-"

        return {
            "symbol": symbol,
            "side": position_info.get("side", ""),
            "ts_opened": position_info.get("ts_opened"),
            "value": position_info.get("value", 0.0),
            "alpha101_display":    _fmt3(current_alpha101, entry_alpha101),
            "neutralized_display": _fmt4(current_neutralized, entry_neutralized),
            "scaled_display":      _fmt4(current_scaled, entry_scaled),
            "decayed_display":     _fmt4(current_decayed, entry_decayed),
            "weight_display":      _fmt4(current_weight, entry_weight),
            "current_alpha101":    current_alpha101,
            "current_neutralized": current_neutralized,
            "current_scaled":      current_scaled,
            "current_decayed":     current_decayed,
            "current_weight":      current_weight,
            "entry_alpha101":      entry_alpha101,
            "entry_neutralized":   entry_neutralized,
            "entry_scaled":        entry_scaled,
            "entry_decayed":       entry_decayed,
            "entry_weight":        entry_weight,
        }
