# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for DataSynchronizer."""

import pytest

from nautilus_quants.factors.engine.data_synchronizer import (
    DataSynchronizer,
    InstrumentData,
)


class MockBar:
    """Mock bar for testing."""
    
    def __init__(
        self,
        instrument_id: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        ts_event: int,
    ):
        self.bar_type = type('BarType', (), {'instrument_id': instrument_id})()
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.ts_event = ts_event


class TestInstrumentData:
    """Tests for InstrumentData."""

    def test_update(self):
        """Test updating instrument data."""
        data = InstrumentData(instrument_id="TEST")
        bar = MockBar("TEST", 100, 105, 95, 102, 1000, 1234567890)
        
        data.update(bar)
        
        assert len(data.close_history) == 1
        assert data.current_close == 102.0
        assert data.current_volume == 1000.0

    def test_max_history(self):
        """Test history trimming."""
        data = InstrumentData(instrument_id="TEST", max_history=5)
        
        for i in range(10):
            bar = MockBar("TEST", 100+i, 105+i, 95+i, 102+i, 1000+i, i)
            data.update(bar)
        
        assert len(data.close_history) == 5
        assert data.close_history[0] == 107.0  # 102 + 5

    def test_get_arrays(self):
        """Test getting numpy arrays."""
        data = InstrumentData(instrument_id="TEST")
        
        for i in range(3):
            bar = MockBar("TEST", 100+i, 105+i, 95+i, 102+i, 1000+i, i)
            data.update(bar)
        
        arrays = data.get_arrays()
        
        assert "close" in arrays
        assert "volume" in arrays
        assert len(arrays["close"]) == 3


class TestDataSynchronizer:
    """Tests for DataSynchronizer."""

    def test_add_instrument(self):
        """Test adding instruments."""
        sync = DataSynchronizer()
        sync.add_instrument("BTCUSDT")
        sync.add_instrument("ETHUSDT")
        
        assert "BTCUSDT" in sync.instruments
        assert "ETHUSDT" in sync.instruments

    def test_on_bar(self):
        """Test processing bars."""
        sync = DataSynchronizer(instruments=["BTCUSDT"])
        bar = MockBar("BTCUSDT", 50000, 51000, 49000, 50500, 100, 1234567890)
        
        sync.on_bar(bar)
        
        data = sync.get_instrument_data("BTCUSDT")
        assert data is not None
        assert data.current_close == 50500.0

    def test_get_all_current_values(self):
        """Test getting current values for all instruments."""
        sync = DataSynchronizer()
        
        bar1 = MockBar("BTCUSDT", 50000, 51000, 49000, 50500, 100, 1)
        bar2 = MockBar("ETHUSDT", 3000, 3100, 2900, 3050, 200, 1)
        
        sync.on_bar(bar1)
        sync.on_bar(bar2)
        
        values = sync.get_all_current_values("close")
        
        assert values["BTCUSDT"] == 50500.0
        assert values["ETHUSDT"] == 3050.0

    def test_is_ready(self):
        """Test readiness check."""
        sync = DataSynchronizer(instruments=["BTCUSDT", "ETHUSDT"])
        
        assert not sync.is_ready
        
        bar1 = MockBar("BTCUSDT", 50000, 51000, 49000, 50500, 100, 1)
        sync.on_bar(bar1)
        
        assert not sync.is_ready  # Still waiting for ETHUSDT
        
        bar2 = MockBar("ETHUSDT", 3000, 3100, 2900, 3050, 200, 1)
        sync.on_bar(bar2)
        
        assert sync.is_ready

    def test_min_history_length(self):
        """Test minimum history length."""
        sync = DataSynchronizer()
        
        # Add 5 bars for BTCUSDT
        for i in range(5):
            bar = MockBar("BTCUSDT", 50000+i, 51000, 49000, 50500+i, 100, i)
            sync.on_bar(bar)
        
        # Add 3 bars for ETHUSDT
        for i in range(3):
            bar = MockBar("ETHUSDT", 3000+i, 3100, 2900, 3050+i, 200, i)
            sync.on_bar(bar)
        
        assert sync.min_history_length == 3

    def test_reset(self):
        """Test reset."""
        sync = DataSynchronizer(instruments=["BTCUSDT"])
        bar = MockBar("BTCUSDT", 50000, 51000, 49000, 50500, 100, 1)
        sync.on_bar(bar)
        
        assert sync.is_ready
        
        sync.reset()
        
        # Should still have instrument registered but no data
        assert "BTCUSDT" in sync.instruments
        data = sync.get_instrument_data("BTCUSDT")
        assert data is not None
        assert len(data.close_history) == 0
