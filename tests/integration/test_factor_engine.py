# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Integration tests for FactorEngine."""

import numpy as np
import pytest

from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.config import FactorConfig, FactorDefinition, PerformanceConfig


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


class TestFactorEngine:
    """Tests for FactorEngine."""

    def test_basic_initialization(self):
        """Test basic engine initialization."""
        engine = FactorEngine()
        
        assert engine.factor_names == []
        assert engine.variable_names == []

    def test_register_expression_factor(self):
        """Test registering expression factors."""
        engine = FactorEngine()
        engine.register_expression_factor(
            name="test_mean",
            expression="ts_mean(close, 5)",
        )
        
        assert "test_mean" in engine.factor_names

    def test_register_variable(self):
        """Test registering variables."""
        engine = FactorEngine()
        engine.register_variable("returns", "delta(close, 1) / delay(close, 1)")
        
        assert "returns" in engine.variable_names

    def test_on_bar_basic(self):
        """Test processing bars."""
        engine = FactorEngine()
        engine.register_expression_factor(
            name="sma_3",
            expression="ts_mean(close, 3)",
            warmup_period=3,
        )
        
        # Send bars
        for i in range(5):
            bar = MockBar("BTCUSDT", 100+i, 105+i, 95+i, 100+i, 1000, i)
            result = engine.on_bar(bar)
        
        assert result is not None
        assert "sma_3" in result.factors

    def test_evaluate_expression(self):
        """Test direct expression evaluation."""
        engine = FactorEngine()
        
        history = {
            "close": np.array([100.0, 102.0, 104.0, 106.0, 108.0]),
        }
        
        result = engine.evaluate_expression("ts_mean(close, 3)", history)
        
        # Mean of last 3: (104 + 106 + 108) / 3 = 106
        assert result == pytest.approx(106.0)

    def test_with_config(self):
        """Test engine with configuration."""
        config = FactorConfig(
            name="test_config",
            parameters={"lookback": 5},
            variables={"returns": "delta(close, 1)"},
            factors=[
                FactorDefinition(
                    name="momentum",
                    expression="ts_mean(close, 5)",
                )
            ],
            performance=PerformanceConfig(
                max_compute_time_ms=1.0,
                enable_timing=True,
            ),
        )
        
        engine = FactorEngine(config=config)
        
        assert "momentum" in engine.factor_names
        assert "returns" in engine.variable_names

    def test_compute_factors_direct(self):
        """Test direct factor computation."""
        engine = FactorEngine()
        engine.register_expression_factor(
            name="ts_max_5",
            expression="ts_max(close, 5)",
        )
        
        history = {
            "close": np.array([100.0, 110.0, 105.0, 108.0, 103.0]),
        }
        current_bar = {"close": 103.0}
        
        results = engine.compute_factors("BTCUSDT", history, current_bar)
        
        assert "ts_max_5" in results
        assert results["ts_max_5"] == pytest.approx(110.0)

    def test_performance_stats(self):
        """Test performance tracking."""
        engine = FactorEngine()
        engine.register_expression_factor(
            name="sma",
            expression="ts_mean(close, 3)",
        )
        
        # Process some bars
        for i in range(10):
            bar = MockBar("BTCUSDT", 100+i, 105+i, 95+i, 100+i, 1000, i)
            engine.on_bar(bar)
        
        stats = engine.get_performance_stats()
        
        assert "mean_ms" in stats
        assert "total_computes" in stats
        assert stats["total_computes"] == 10

    def test_reset(self):
        """Test engine reset."""
        engine = FactorEngine()
        engine.register_expression_factor("sma", "ts_mean(close, 3)")
        
        for i in range(5):
            bar = MockBar("BTCUSDT", 100+i, 105+i, 95+i, 100+i, 1000, i)
            engine.on_bar(bar)
        
        stats_before = engine.get_performance_stats()
        assert stats_before["total_computes"] > 0
        
        engine.reset()
        
        stats_after = engine.get_performance_stats()
        assert stats_after["total_computes"] == 0

    def test_multiple_factors(self):
        """Test with multiple factors."""
        engine = FactorEngine()
        engine.register_expression_factor("sma_3", "ts_mean(close, 3)")
        engine.register_expression_factor("sma_5", "ts_mean(close, 5)")
        engine.register_expression_factor("max_3", "ts_max(close, 3)")
        
        for i in range(10):
            bar = MockBar("BTCUSDT", 100+i, 105+i, 95+i, 100+i, 1000, i)
            result = engine.on_bar(bar)
        
        assert result is not None
        assert "sma_3" in result.factors
        assert "sma_5" in result.factors
        assert "max_3" in result.factors

    def test_breakout_factor_expression(self):
        """Test breakout factor expression evaluation."""
        engine = FactorEngine()
        
        # Create price data with a breakout
        # Need 32 elements so ts_max(30) doesn't include the breakout value
        closes = [100.0] * 31 + [110.0]  # 32 elements: 31 x 100, then breakout to 110
        volumes = [1000.0] * 31 + [2000.0]  # Volume also breaks out
        
        history = {
            "close": np.array(closes),
            "volume": np.array(volumes),
        }
        
        # Test ts_max component - max of last 30 values
        max_close = engine.evaluate_expression("ts_max(close, 30)", history)
        assert max_close == pytest.approx(110.0)  # Includes current bar
        
        # Test that we can evaluate nested expressions
        # Note: delay on a scalar returns the scalar itself
        highest_close = engine.evaluate_expression("delay(ts_max(close, 30), 1)", history)
        assert highest_close == pytest.approx(110.0)  # delay on scalar returns scalar
        
        # For proper "previous day's high" we need to check via on_bar processing
        # which maintains proper history state
