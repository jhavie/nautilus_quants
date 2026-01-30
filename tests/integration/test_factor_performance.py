# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Integration tests for factor performance."""

import time

import numpy as np
import pytest

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


class TestFactorPerformance:
    """Performance tests for FactorEngine."""

    def test_single_factor_performance(self):
        """Test single factor computation time."""
        engine = FactorEngine()
        engine.register_expression_factor("sma_20", "ts_mean(close, 20)")
        
        # Warm up
        for i in range(50):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000, i)
            engine.on_bar(bar)
        
        # Measure
        start = time.perf_counter()
        iterations = 1000
        
        for i in range(iterations):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000, 50 + i)
            engine.on_bar(bar)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        
        # Should be well under 1ms per computation
        assert avg_ms < 1.0, f"Average computation time {avg_ms:.3f}ms exceeds 1ms"

    def test_multiple_factors_performance(self):
        """Test multiple factor computation time."""
        engine = FactorEngine()
        
        # Register 10 factors
        for i in range(10):
            engine.register_expression_factor(
                f"sma_{5*(i+1)}", 
                f"ts_mean(close, {5*(i+1)})"
            )
        
        # Warm up
        for i in range(100):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000, i)
            engine.on_bar(bar)
        
        # Measure
        start = time.perf_counter()
        iterations = 500
        
        for i in range(iterations):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000, 100 + i)
            engine.on_bar(bar)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        
        # 10 factors should still be under 1ms
        assert avg_ms < 1.0, f"Average computation time {avg_ms:.3f}ms exceeds 1ms"

    def test_complex_expression_performance(self):
        """Test complex expression computation time."""
        engine = FactorEngine()
        
        # Complex breakout-style expression
        engine.register_expression_factor(
            "breakout",
            "(close > delay(ts_max(close, 30), 1)) * (volume > delay(ts_max(volume, 30), 1))",
        )
        
        # Warm up
        for i in range(200):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000 + i, i)
            engine.on_bar(bar)
        
        # Measure
        start = time.perf_counter()
        iterations = 500
        
        for i in range(iterations):
            bar = MockBar("BTCUSDT", 50000 + i * 10, 1000 + i, 200 + i)
            engine.on_bar(bar)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        
        assert avg_ms < 1.0, f"Average computation time {avg_ms:.3f}ms exceeds 1ms"

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        engine = FactorEngine()
        engine.register_expression_factor("sma", "ts_mean(close, 10)")
        
        for i in range(100):
            bar = MockBar("BTCUSDT", 50000 + i, 1000, i)
            engine.on_bar(bar)
        
        stats = engine.get_performance_stats()
        
        assert stats["total_computes"] == 100
        assert stats["mean_ms"] >= 0
        assert stats["max_ms"] >= stats["mean_ms"]
        assert "p95_ms" in stats
