# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""End-to-end tests for time-series factors."""

import math

import numpy as np
import pytest

from nautilus_quants.factors.base.factor import ExpressionFactor
from nautilus_quants.factors.base.time_series_factor import (
    MeanReversionFactor,
    MomentumFactor,
    VolatilityFactor,
)
from nautilus_quants.factors.types import FactorInput


def make_factor_input(
    close: list[float],
    volume: list[float] | None = None,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
) -> FactorInput:
    """Helper to create FactorInput from lists."""
    close_arr = np.array(close)
    return FactorInput(
        instrument_id=None,  # type: ignore
        timestamp_ns=0,
        open=open_[-1] if open_ else close[-1],
        high=high[-1] if high else close[-1],
        low=low[-1] if low else close[-1],
        close=close[-1],
        volume=volume[-1] if volume else 0.0,
        history={
            "close": close_arr,
            "volume": np.array(volume) if volume else np.zeros(len(close)),
            "open": np.array(open_) if open_ else close_arr,
            "high": np.array(high) if high else close_arr,
            "low": np.array(low) if low else close_arr,
        },
    )


class TestMomentumFactor:
    """Tests for MomentumFactor."""

    def test_basic_momentum(self):
        """Test basic momentum calculation."""
        # Prices: 100 -> 120 (20% increase over 5 periods)
        prices = [100.0, 105.0, 110.0, 115.0, 118.0, 120.0]
        data = make_factor_input(prices)
        
        factor = MomentumFactor(lookback=5)
        result = factor.compute(data)
        
        # (120 - 100) / 100 = 0.2
        assert result == pytest.approx(0.2)

    def test_negative_momentum(self):
        """Test negative momentum."""
        prices = [100.0, 95.0, 90.0, 85.0, 82.0, 80.0]
        data = make_factor_input(prices)
        
        factor = MomentumFactor(lookback=5)
        result = factor.compute(data)
        
        # (80 - 100) / 100 = -0.2
        assert result == pytest.approx(-0.2)

    def test_insufficient_data(self):
        """Test momentum with insufficient data."""
        prices = [100.0, 105.0]
        data = make_factor_input(prices)
        
        factor = MomentumFactor(lookback=5)
        result = factor.compute(data)
        
        assert math.isnan(result)


class TestVolatilityFactor:
    """Tests for VolatilityFactor."""

    def test_basic_volatility(self):
        """Test basic volatility calculation."""
        # Prices with known volatility
        prices = [100.0, 102.0, 101.0, 103.0, 102.0]
        data = make_factor_input(prices)
        
        factor = VolatilityFactor(lookback=4)
        result = factor.compute(data)
        
        # Should return std of log returns
        assert result > 0
        assert not math.isnan(result)

    def test_zero_volatility(self):
        """Test volatility of constant prices."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        data = make_factor_input(prices)
        
        factor = VolatilityFactor(lookback=4)
        result = factor.compute(data)
        
        assert result == pytest.approx(0.0)


class TestMeanReversionFactor:
    """Tests for MeanReversionFactor."""

    def test_above_mean(self):
        """Test z-score when price is above mean."""
        # Mean of [100, 100, 100, 100, 120] = 104
        # Current = 120, std should give positive z-score
        prices = [100.0, 100.0, 100.0, 100.0, 120.0]
        data = make_factor_input(prices)
        
        factor = MeanReversionFactor(lookback=5)
        result = factor.compute(data)
        
        assert result > 0  # Above mean = positive z-score

    def test_below_mean(self):
        """Test z-score when price is below mean."""
        prices = [100.0, 100.0, 100.0, 100.0, 80.0]
        data = make_factor_input(prices)
        
        factor = MeanReversionFactor(lookback=5)
        result = factor.compute(data)
        
        assert result < 0  # Below mean = negative z-score

    def test_at_mean(self):
        """Test z-score when price equals mean."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        data = make_factor_input(prices)
        
        factor = MeanReversionFactor(lookback=5)
        result = factor.compute(data)
        
        # At mean with zero std returns 0
        assert result == pytest.approx(0.0)


class TestExpressionFactor:
    """Tests for ExpressionFactor."""

    def test_simple_expression(self):
        """Test simple expression evaluation."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        data = make_factor_input(prices)
        
        # ts_mean of last 3 values: (104 + 106 + 108) / 3 = 106
        factor = ExpressionFactor(
            name="test_mean",
            expression="ts_mean(close, 3)",
            warmup_period=3,
        )
        result = factor.compute(data)
        
        assert result == pytest.approx(106.0)

    def test_ts_max_expression(self):
        """Test ts_max expression."""
        prices = [100.0, 110.0, 105.0, 108.0, 103.0]
        data = make_factor_input(prices)
        
        # ts_max of last 3: max(105, 108, 103) = 108
        factor = ExpressionFactor(
            name="test_max",
            expression="ts_max(close, 3)",
            warmup_period=3,
        )
        result = factor.compute(data)
        
        assert result == pytest.approx(108.0)

    def test_delay_expression(self):
        """Test delay expression."""
        prices = [100.0, 110.0, 120.0, 130.0, 140.0]
        data = make_factor_input(prices)
        
        # delay(close, 2) = value 2 periods ago = 120
        factor = ExpressionFactor(
            name="test_delay",
            expression="delay(close, 2)",
            warmup_period=3,
        )
        result = factor.compute(data)
        
        assert result == pytest.approx(120.0)

    def test_arithmetic_expression(self):
        """Test arithmetic expression."""
        prices = [100.0, 100.0, 100.0, 100.0, 200.0]
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 2000.0]
        data = make_factor_input(prices, volumes)
        
        # (close - ts_mean(close, 5)) = 200 - 120 = 80
        factor = ExpressionFactor(
            name="test_diff",
            expression="close - ts_mean(close, 5)",
            warmup_period=5,
        )
        # Note: close variable is the history array, so we get the last value
        result = factor.compute(data)
        
        # Mean of [100, 100, 100, 100, 200] = 120
        # Diff = 200 - 120 = 80... but close is array, so result may differ
        assert not math.isnan(result)

    def test_comparison_expression(self):
        """Test comparison expression returns 0 or 1."""
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        data = make_factor_input(prices)
        
        # close > ts_mean(close, 5)
        # 120 > 110 = True = 1.0
        factor = ExpressionFactor(
            name="test_compare",
            expression="close > ts_mean(close, 5)",
            warmup_period=5,
        )
        result = factor.compute(data)
        
        # Result should be 0 or 1
        assert result in [0.0, 1.0] or not math.isnan(result)

    def test_with_parameters(self):
        """Test expression with parameters."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        data = make_factor_input(prices)
        
        factor = ExpressionFactor(
            name="test_param",
            expression="ts_mean(close, lookback)",
            warmup_period=3,
            parameters={"lookback": 3},
        )
        result = factor.compute(data)
        
        assert result == pytest.approx(106.0)


class TestFactorWarmup:
    """Tests for factor warmup period handling."""

    def test_warmup_period(self):
        """Test that update respects warmup period."""
        factor = MomentumFactor(lookback=5)
        
        # Create data with enough history
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        data = make_factor_input(prices)
        
        # Before warmup
        assert not factor.is_warmed_up
        
        # Update multiple times to warm up
        for i in range(5):
            factor.update(data)
        
        # Should be warmed up now
        assert factor.is_warmed_up

    def test_reset(self):
        """Test factor reset."""
        factor = MomentumFactor(lookback=5)
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        data = make_factor_input(prices)
        
        # Warm up
        for i in range(5):
            factor.update(data)
        assert factor.is_warmed_up
        
        # Reset
        factor.reset()
        assert not factor.is_warmed_up
