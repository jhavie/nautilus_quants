# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Data Synchronizer for Multi-Instrument Factor Computation.

Handles synchronization of bar data across multiple instruments
for cross-sectional factor calculations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId


@dataclass
class InstrumentData:
    """Historical data for a single instrument."""
    instrument_id: str
    open_history: list[float] = field(default_factory=list)
    high_history: list[float] = field(default_factory=list)
    low_history: list[float] = field(default_factory=list)
    close_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    max_history: int = 500  # Maximum history to keep
    extra_fields: list[str] = field(default_factory=list)
    extra_history: dict[str, list[float]] = field(default_factory=dict)

    def set_extra_fields(self, fields: list[str]) -> None:
        """Set extra bar fields for tracking (called once on first bar)."""
        self.extra_fields = fields
        for f in fields:
            if f not in self.extra_history:
                self.extra_history[f] = []

    def update(self, bar: Bar) -> None:
        """Update with new bar data."""
        self.open_history.append(float(bar.open))
        self.high_history.append(float(bar.high))
        self.low_history.append(float(bar.low))
        self.close_history.append(float(bar.close))
        self.volume_history.append(float(bar.volume))
        self.timestamps.append(bar.ts_event)
        for f in self.extra_fields:
            self.extra_history[f].append(float(getattr(bar, f, 0)))

        # Trim to max history
        if len(self.close_history) > self.max_history:
            self.open_history = self.open_history[-self.max_history:]
            self.high_history = self.high_history[-self.max_history:]
            self.low_history = self.low_history[-self.max_history:]
            self.close_history = self.close_history[-self.max_history:]
            self.volume_history = self.volume_history[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]
            for f in self.extra_fields:
                self.extra_history[f] = self.extra_history[f][-self.max_history:]

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Get history as numpy arrays."""
        result = {
            "open": np.array(self.open_history),
            "high": np.array(self.high_history),
            "low": np.array(self.low_history),
            "close": np.array(self.close_history),
            "volume": np.array(self.volume_history),
        }
        for f in self.extra_fields:
            result[f] = np.array(self.extra_history.get(f, []))
        return result
    
    @property
    def current_close(self) -> float:
        """Get current (latest) close price."""
        return self.close_history[-1] if self.close_history else float('nan')
    
    @property
    def current_volume(self) -> float:
        """Get current (latest) volume."""
        return self.volume_history[-1] if self.volume_history else float('nan')
    
    @property
    def last_timestamp(self) -> int:
        """Get last update timestamp."""
        return self.timestamps[-1] if self.timestamps else 0


class DataSynchronizer:
    """
    Synchronizes bar data across multiple instruments.
    
    Tracks bar arrivals and determines when all instruments have
    data for a given timestamp, enabling cross-sectional calculations.
    
    Attributes:
        instruments: Set of instrument IDs being tracked
        max_history: Maximum history length to maintain
        sync_tolerance_ns: Tolerance for timestamp alignment (nanoseconds)
    """
    
    def __init__(
        self,
        instruments: list[str] | None = None,
        max_history: int = 500,
        sync_tolerance_ns: int = 0,
    ) -> None:
        """
        Initialize the synchronizer.
        
        Args:
            instruments: List of instrument IDs to track (can be added later)
            max_history: Maximum history length per instrument
            sync_tolerance_ns: Tolerance for timestamp matching
        """
        self.instruments: set[str] = set(instruments or [])
        self.max_history = max_history
        self.sync_tolerance_ns = sync_tolerance_ns
        
        self._data: dict[str, InstrumentData] = {}
        self._pending_bars: dict[int, dict[str, Bar]] = defaultdict(dict)
        self._last_sync_ts: int = 0
        self._extra_fields: list[str] = []
    
    def set_extra_fields(self, fields: list[str]) -> None:
        """Set extra bar fields to track across all instruments."""
        self._extra_fields = fields
        for data in self._data.values():
            data.set_extra_fields(fields)

    def add_instrument(self, instrument_id: str) -> None:
        """Add an instrument to track."""
        self.instruments.add(instrument_id)
        if instrument_id not in self._data:
            inst_data = InstrumentData(
                instrument_id=instrument_id,
                max_history=self.max_history,
            )
            if self._extra_fields:
                inst_data.set_extra_fields(self._extra_fields)
            self._data[instrument_id] = inst_data
    
    def remove_instrument(self, instrument_id: str) -> None:
        """Remove an instrument from tracking."""
        self.instruments.discard(instrument_id)
        self._data.pop(instrument_id, None)
    
    def on_bar(self, bar: Bar) -> bool:
        """
        Process incoming bar data.
        
        Args:
            bar: The bar to process
            
        Returns:
            True if all instruments are now synchronized for this timestamp
        """
        instrument_id = str(bar.bar_type.instrument_id)
        
        # Add instrument if not tracked or data not initialized
        if instrument_id not in self.instruments or instrument_id not in self._data:
            self.add_instrument(instrument_id)
        
        # Update instrument data
        self._data[instrument_id].update(bar)
        
        # Track pending bars for synchronization
        ts = bar.ts_event
        self._pending_bars[ts][instrument_id] = bar
        
        # Check if all instruments have data for this timestamp
        return self._check_sync(ts)
    
    def _check_sync(self, timestamp: int) -> bool:
        """Check if all instruments are synchronized at timestamp."""
        if not self.instruments:
            return False
        
        pending = self._pending_bars.get(timestamp, {})
        
        # All instruments must have data
        if set(pending.keys()) >= self.instruments:
            self._last_sync_ts = timestamp
            # Clean up old pending bars
            self._cleanup_pending(timestamp)
            return True
        
        return False
    
    def _cleanup_pending(self, current_ts: int) -> None:
        """Remove old pending bar entries."""
        old_timestamps = [
            ts for ts in self._pending_bars 
            if ts < current_ts - self.sync_tolerance_ns
        ]
        for ts in old_timestamps:
            del self._pending_bars[ts]
    
    def get_instrument_data(self, instrument_id: str) -> InstrumentData | None:
        """Get data for a specific instrument."""
        return self._data.get(instrument_id)
    
    def get_all_current_values(self, field: str = "close") -> dict[str, float]:
        """
        Get current values for all instruments.
        
        Args:
            field: Field to get ('close', 'volume', etc.)
            
        Returns:
            Dict of {instrument_id: value}
        """
        result: dict[str, float] = {}
        
        for instrument_id, data in self._data.items():
            if field == "close":
                result[instrument_id] = data.current_close
            elif field == "volume":
                result[instrument_id] = data.current_volume
            else:
                arrays = data.get_arrays()
                if field in arrays and len(arrays[field]) > 0:
                    result[instrument_id] = float(arrays[field][-1])
                else:
                    result[instrument_id] = float('nan')
        
        return result
    
    def get_all_history(self, field: str = "close") -> dict[str, np.ndarray]:
        """
        Get historical data for all instruments.
        
        Args:
            field: Field to get
            
        Returns:
            Dict of {instrument_id: history_array}
        """
        result: dict[str, np.ndarray] = {}
        
        for instrument_id, data in self._data.items():
            arrays = data.get_arrays()
            if field in arrays:
                result[instrument_id] = arrays[field]
        
        return result
    
    @property
    def is_ready(self) -> bool:
        """Check if synchronizer has data for all instruments."""
        if not self.instruments:
            return False
        return all(
            instrument_id in self._data and len(self._data[instrument_id].close_history) > 0
            for instrument_id in self.instruments
        )
    
    @property
    def min_history_length(self) -> int:
        """Get minimum history length across all instruments."""
        if not self._data:
            return 0
        return min(
            len(data.close_history) 
            for data in self._data.values()
        )
    
    def reset(self) -> None:
        """Reset all data."""
        extra_fields = self._extra_fields
        self._data.clear()
        self._pending_bars.clear()
        self._last_sync_ts = 0
        self._extra_fields = extra_fields

        # Re-initialize instrument data
        for instrument_id in self.instruments:
            inst_data = InstrumentData(
                instrument_id=instrument_id,
                max_history=self.max_history,
            )
            if self._extra_fields:
                inst_data.set_extra_fields(self._extra_fields)
            self._data[instrument_id] = inst_data
