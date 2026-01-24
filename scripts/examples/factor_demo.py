#!/usr/bin/env python
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Framework Demo.

This script demonstrates how to use the factor framework to compute
Alpha101-style factors from market data.
"""

import numpy as np

from nautilus_quants.factors.engine import FactorEngine
from nautilus_quants.factors.builtin import register_alpha101_factors


class MockBar:
    """Mock bar for demonstration."""
    
    def __init__(self, instrument_id: str, close: float, volume: float, ts_event: int):
        self.bar_type = type('BarType', (), {'instrument_id': instrument_id})()
        self.open = close * 0.99
        self.high = close * 1.01
        self.low = close * 0.98
        self.close = close
        self.volume = volume
        self.ts_event = ts_event


def demo_basic_factor():
    """Demonstrate basic factor computation."""
    print("=" * 60)
    print("Demo 1: Basic Factor Computation")
    print("=" * 60)
    
    # Create engine
    engine = FactorEngine()
    
    # Register a simple moving average factor
    engine.register_expression_factor(
        name="sma_20",
        expression="ts_mean(close, 20)",
        description="20-period simple moving average",
    )
    
    # Register momentum factor
    engine.register_expression_factor(
        name="momentum_10",
        expression="delta(close, 10) / delay(close, 10)",
        description="10-period momentum",
    )
    
    print(f"Registered factors: {engine.factor_names}")
    
    # Generate synthetic price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Process bars
    for i, price in enumerate(prices):
        bar = MockBar("BTCUSDT", price, 1000 + i * 10, i * 3600_000_000_000)
        result = engine.on_bar(bar)
        
        if result and i >= 20:  # After warmup
            sma = result.factors.get("sma_20", {}).get("BTCUSDT", float('nan'))
            momentum = result.factors.get("momentum_10", {}).get("BTCUSDT", float('nan'))
            
            if i % 20 == 0:
                print(f"Bar {i:3d}: Price={price:.2f}, SMA20={sma:.2f}, Momentum={momentum:.4f}")
    
    print()


def demo_breakout_factor():
    """Demonstrate breakout factor computation."""
    print("=" * 60)
    print("Demo 2: Breakout Factor")
    print("=" * 60)
    
    engine = FactorEngine()
    
    # Register breakout factor components
    engine.register_variable("highest_close", "delay(ts_max(close, 30), 1)")
    engine.register_variable("highest_volume", "delay(ts_max(volume, 30), 1)")
    
    engine.register_expression_factor(
        name="breakout",
        expression="(close > highest_close) * (volume > highest_volume)",
        description="Price and volume breakout signal",
        warmup_period=32,
    )
    
    # Generate synthetic data with a breakout
    prices = [100.0] * 50  # Stable prices
    volumes = [1000.0] * 50  # Stable volume
    
    # Add breakout
    prices.extend([105.0, 110.0, 115.0])  # Price surge
    volumes.extend([2000.0, 2500.0, 3000.0])  # Volume surge
    
    print("Processing bars...")
    breakout_count = 0
    
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        bar = MockBar("ETHUSDT", price, volume, i * 3600_000_000_000)
        result = engine.on_bar(bar)
        
        if result:
            signal = result.factors.get("breakout", {}).get("ETHUSDT", 0)
            if signal == 1.0:
                breakout_count += 1
                print(f"🚀 BREAKOUT at bar {i}: Price={price:.2f}, Volume={volume:.0f}")
    
    print(f"\nTotal breakout signals: {breakout_count}")
    print()


def demo_alpha101():
    """Demonstrate Alpha101 factors."""
    print("=" * 60)
    print("Demo 3: Alpha101 Factors")
    print("=" * 60)
    
    engine = FactorEngine()
    
    # Register some Alpha101 factors
    register_alpha101_factors(engine, ["alpha004", "alpha005", "alpha006"])
    
    print(f"Registered Alpha101 factors: {engine.factor_names}")
    
    # Generate synthetic data
    np.random.seed(123)
    
    for i in range(100):
        price = 50000 + np.random.randn() * 500
        volume = 10000 + np.random.randn() * 1000
        bar = MockBar("BTCUSDT", price, abs(volume), i * 3600_000_000_000)
        result = engine.on_bar(bar)
        
        if result and i == 99:
            print("\nFinal factor values:")
            for factor_name in engine.factor_names:
                value = result.factors.get(factor_name, {}).get("BTCUSDT", float('nan'))
                print(f"  {factor_name}: {value:.4f}")
    
    # Performance stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance stats:")
    print(f"  Mean compute time: {stats['mean_ms']:.4f} ms")
    print(f"  Max compute time: {stats['max_ms']:.4f} ms")
    print(f"  Total computes: {stats['total_computes']}")
    print()


def demo_config_loading():
    """Demonstrate loading factors from config file."""
    print("=" * 60)
    print("Demo 4: Config-Driven Factors")
    print("=" * 60)
    
    try:
        from nautilus_quants.factors.config import load_factor_config
        
        config = load_factor_config("config/factors.yaml")
        engine = FactorEngine(config=config)
        
        print(f"Loaded config: {config.name} v{config.version}")
        print(f"Parameters: {list(config.parameters.keys())}")
        print(f"Variables: {list(config.variables.keys())}")
        print(f"Factors: {engine.factor_names}")
        
    except FileNotFoundError:
        print("Config file not found. Run from project root directory.")
    except Exception as e:
        print(f"Error loading config: {e}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("FACTOR FRAMEWORK DEMONSTRATION")
    print("=" * 60 + "\n")
    
    demo_basic_factor()
    demo_breakout_factor()
    demo_alpha101()
    demo_config_loading()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
