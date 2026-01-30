# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Integration tests for Alpha101 factors."""

import numpy as np
import pytest

from nautilus_quants.factors.builtin.alpha101 import (
    ALPHA101_FACTORS,
    list_alpha101_factors,
    register_alpha101_factors,
)
from nautilus_quants.factors.engine.factor_engine import FactorEngine


class MockBar:
    """Mock bar for testing."""
    
    def __init__(self, instrument_id: str, close: float, volume: float, ts_event: int):
        self.bar_type = type('BarType', (), {'instrument_id': instrument_id})()
        self.open = close * 0.99
        self.high = close * 1.01
        self.low = close * 0.98
        self.close = close
        self.volume = volume
        self.ts_event = ts_event


class TestAlpha101:
    """Tests for Alpha101 factors."""

    def test_list_factors(self):
        """Test listing available factors."""
        factors = list_alpha101_factors()
        
        assert len(factors) >= 10
        assert "alpha001" in factors
        assert "alpha010" in factors

    def test_register_all_factors(self):
        """Test registering all Alpha101 factors."""
        engine = FactorEngine()
        register_alpha101_factors(engine)
        
        assert len(engine.factor_names) >= 10

    def test_register_specific_factors(self):
        """Test registering specific factors."""
        engine = FactorEngine()
        register_alpha101_factors(engine, ["alpha001", "alpha002"])
        
        assert "alpha001" in engine.factor_names
        assert "alpha002" in engine.factor_names
        assert "alpha003" not in engine.factor_names

    def test_factor_computation(self):
        """Test that Alpha101 factors can be computed."""
        engine = FactorEngine()
        register_alpha101_factors(engine, ["alpha004", "alpha005"])
        
        # Generate test data
        for i in range(100):
            bar = MockBar("BTCUSDT", 50000 + np.random.randn() * 100, 1000 + i, i)
            result = engine.on_bar(bar)
        
        assert result is not None
        assert "alpha004" in result.factors
        assert "alpha005" in result.factors

    def test_invalid_factor_name(self):
        """Test error handling for invalid factor name."""
        engine = FactorEngine()
        
        with pytest.raises(ValueError, match="Unknown Alpha101 factor"):
            register_alpha101_factors(engine, ["invalid_factor"])
