# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for time-series operators."""

import math

import numpy as np
import pytest

import pandas as pd

from nautilus_quants.factors.operators.time_series import (
    Delay,
    Delta,
    TsArgmax,
    TsArgmin,
    TsMax,
    TsMean,
    TsMin,
    TsRank,
    TsSkew,
    TsStd,
    TsSum,
    WqTsArgmax,
    WqTsArgmin,
    WqTsRank,
    delay,
    delta,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_skew,
    ts_std,
    ts_sum,
    wq_ts_argmax,
    wq_ts_argmin,
    wq_ts_rank,
)


class TestTsMean:
    """Tests for ts_mean operator."""

    def test_basic_mean(self):
        """Test basic rolling mean calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_mean(data, 3)
        assert result == pytest.approx(4.0)  # mean of [3, 4, 5]

    def test_full_window(self):
        """Test mean with window equal to data length."""
        data = np.array([2.0, 4.0, 6.0, 8.0])
        result = ts_mean(data, 4)
        assert result == pytest.approx(5.0)

    def test_insufficient_data(self):
        """Test with insufficient data returns NaN."""
        data = np.array([1.0, 2.0])
        result = ts_mean(data, 5)
        assert math.isnan(result)

    def test_single_value(self):
        """Test mean of single value."""
        data = np.array([42.0])
        result = ts_mean(data, 1)
        assert result == pytest.approx(42.0)


class TestTsSum:
    """Tests for ts_sum operator."""

    def test_basic_sum(self):
        """Test basic rolling sum."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_sum(data, 3)
        assert result == pytest.approx(12.0)  # 3 + 4 + 5

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = np.array([1.0])
        result = ts_sum(data, 3)
        assert math.isnan(result)


class TestTsStd:
    """Tests for ts_std operator."""

    def test_basic_std(self):
        """Test basic rolling standard deviation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_std(data, 5)
        expected = np.std([1, 2, 3, 4, 5], ddof=1)
        assert result == pytest.approx(expected)

    def test_constant_values(self):
        """Test std of constant values is zero."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = ts_std(data, 4)
        assert result == pytest.approx(0.0)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = np.array([1.0, 2.0])
        result = ts_std(data, 5)
        assert math.isnan(result)


class TestTsMinMax:
    """Tests for ts_min and ts_max operators."""

    def test_ts_min(self):
        """Test rolling minimum."""
        data = np.array([5.0, 3.0, 7.0, 2.0, 8.0])
        result = ts_min(data, 3)
        assert result == pytest.approx(2.0)  # min of [7, 2, 8]

    def test_ts_max(self):
        """Test rolling maximum."""
        data = np.array([5.0, 3.0, 7.0, 2.0, 8.0])
        result = ts_max(data, 3)
        assert result == pytest.approx(8.0)  # max of [7, 2, 8]

    def test_ts_max_full_window(self):
        """Test max with full window."""
        data = np.array([1.0, 9.0, 3.0, 5.0])
        result = ts_max(data, 4)
        assert result == pytest.approx(9.0)


class TestTsRank:
    """Tests for ts_rank operator."""

    def test_highest_rank(self):
        """Test rank when current value is highest."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_rank(data, 5)
        assert result == pytest.approx(1.0)  # highest = rank 1.0

    def test_lowest_rank(self):
        """Test rank when current value is lowest."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_rank(data, 5)
        # avg_rank = (0 less + (1 equal + 1)/2) / 5 = 1/5 = 0.2
        assert result == pytest.approx(0.2)

    def test_middle_rank(self):
        """Test rank when current value is in middle."""
        data = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        result = ts_rank(data, 5)
        # 3.0: 2 values less (1.0, 2.0), 1 equal
        # avg_rank = (2 + (1+1)/2) / 5 = 3/5 = 0.6
        assert result == pytest.approx(0.6)


class TestTsArgmaxArgmin:
    """Tests for ts_argmax and ts_argmin operators (WorldQuant semantics).
    
    WorldQuant convention:
    - 1 = oldest day in window
    - window = most recent day (today)
    """

    def test_argmax_most_recent(self):
        """Test argmax when max is most recent (today)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_argmax(data, 5)
        # WorldQuant: window=5 means today is position 5
        assert result == pytest.approx(5.0)  # max is at position 5 (most recent)

    def test_argmax_oldest(self):
        """Test argmax when max is oldest."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_argmax(data, 5)
        # WorldQuant: 1 = oldest day in window
        assert result == pytest.approx(1.0)  # max is at position 1 (oldest)

    def test_argmin_most_recent(self):
        """Test argmin when min is most recent (today)."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_argmin(data, 5)
        # WorldQuant: window=5 means today is position 5
        assert result == pytest.approx(5.0)  # min is at position 5 (most recent)

    def test_argmin_oldest(self):
        """Test argmin when min is oldest."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_argmin(data, 5)
        # WorldQuant: 1 = oldest day in window
        assert result == pytest.approx(1.0)  # min is at position 1 (oldest)
    
    def test_argmax_breakout_detection(self):
        """Test ts_argmax for breakout detection (WorldQuant style).
        
        ts_argmax(close, 31) == 31 means today's close is strictly greater
        than all values in the past 30 days.
        """
        # Scenario: breakout on the last bar
        data = np.array([100.0] * 30 + [105.0])  # 30 days at 100, today at 105
        result = ts_argmax(data, 31)
        assert result == pytest.approx(31.0)  # today is the max
        
        # Scenario: no breakout (max was earlier)
        data2 = np.array([100.0] * 15 + [110.0] + [100.0] * 14 + [105.0])
        result2 = ts_argmax(data2, 31)
        assert result2 == pytest.approx(16.0)  # max was at position 16


class TestDelta:
    """Tests for delta operator."""

    def test_basic_delta(self):
        """Test basic difference calculation."""
        data = np.array([10.0, 12.0, 15.0, 14.0, 18.0])
        result = delta(data, 2)
        # delta(2) = current - value_2_periods_ago = 18 - 15 = 3
        assert result == pytest.approx(3.0)

    def test_delta_one_period(self):
        """Test delta with 1 period."""
        data = np.array([5.0, 8.0])
        result = delta(data, 1)
        assert result == pytest.approx(3.0)  # 8 - 5

    def test_delta_insufficient_data(self):
        """Test delta with insufficient data."""
        data = np.array([1.0, 2.0])
        result = delta(data, 5)
        assert math.isnan(result)


class TestDelay:
    """Tests for delay operator."""

    def test_basic_delay(self):
        """Test basic lagged value."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = delay(data, 2)
        # delay(2) = value 2 periods ago = 30
        assert result == pytest.approx(30.0)

    def test_delay_one_period(self):
        """Test delay with 1 period."""
        data = np.array([100.0, 200.0])
        result = delay(data, 1)
        assert result == pytest.approx(100.0)

    def test_delay_insufficient_data(self):
        """Test delay with insufficient data."""
        data = np.array([1.0])
        result = delay(data, 2)
        assert math.isnan(result)


class TestOperatorClasses:
    """Test operator class instantiation and methods."""

    def test_operator_name(self):
        """Test operator names are set correctly."""
        assert TsMean.name == "ts_mean"
        assert TsMax.name == "ts_max"
        assert Delay.name == "delay"
        assert Delta.name == "delta"

    def test_operator_args(self):
        """Test operator argument validation."""
        op = TsMean()
        assert op.min_args == 2
        assert op.max_args == 2

    def test_operator_compute_direct(self):
        """Test calling compute directly on operator instance."""
        op = TsMean()
        data = np.array([1.0, 2.0, 3.0])
        result = op.compute(data, 3)
        assert result == pytest.approx(2.0)

    def test_wq_operator_names(self):
        """Test wq_ operator names are set correctly."""
        assert WqTsRank.name == "wq_ts_rank"
        assert WqTsArgmax.name == "wq_ts_argmax"
        assert WqTsArgmin.name == "wq_ts_argmin"


class TestWqTsRank:
    """Tests for wq_ts_rank operator (BRAIN semantics, [0, 1])."""

    def test_highest_value(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_rank(data, 5)
        assert result == pytest.approx(1.0)

    def test_lowest_value(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_rank(data, 5)
        assert result == pytest.approx(0.0)

    def test_median_value(self):
        data = np.array([200.0, 0.0, 100.0])
        result = wq_ts_rank(data, 3)
        assert result == pytest.approx(0.5)

    def test_window_one(self):
        data = np.array([42.0])
        result = wq_ts_rank(data, 1)
        assert result == pytest.approx(0.5)

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = wq_ts_rank(data, 5)
        assert math.isnan(result)


class TestWqTsArgmax:
    """Tests for wq_ts_argmax operator (BRAIN semantics, 0-indexed from today)."""

    def test_max_today(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_argmax(data, 5)
        assert result == pytest.approx(0.0)

    def test_max_oldest(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_argmax(data, 5)
        assert result == pytest.approx(4.0)

    def test_max_middle(self):
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        result = wq_ts_argmax(data, 6)
        assert result == pytest.approx(4.0)  # max=9, index 1, offset=6-1-1=4

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = wq_ts_argmax(data, 5)
        assert math.isnan(result)


class TestWqTsArgmin:
    """Tests for wq_ts_argmin operator (BRAIN semantics, 0-indexed from today)."""

    def test_min_today(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = wq_ts_argmin(data, 5)
        assert result == pytest.approx(0.0)

    def test_min_oldest(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wq_ts_argmin(data, 5)
        assert result == pytest.approx(4.0)

    def test_min_middle(self):
        data = np.array([4.0, 9.0, 5.0, 8.0, 2.0, 6.0])
        result = wq_ts_argmin(data, 6)
        assert result == pytest.approx(1.0)  # min=2, index 4, offset=6-1-4=1

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = wq_ts_argmin(data, 5)
        assert math.isnan(result)


class TestTsSkew:
    """Tests for ts_skew operator."""

    def test_positive_skew(self):
        """Right-skewed data should produce positive skewness."""
        # Exponential-like: many small values, few large values
        data = np.array([1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.3, 5.0, 1.0, 1.1])
        result = ts_skew(data, 10)
        assert result > 0, "Right-skewed data should have positive skewness"

    def test_symmetric_near_zero(self):
        """Symmetric data should have skewness close to 0."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_skew(data, 10)
        assert abs(result) < 0.5, f"Symmetric data skewness={result}, expected ~0"

    def test_negative_skew(self):
        """Left-skewed data should produce negative skewness."""
        data = np.array([5.0, 4.9, 4.8, 5.0, 4.9, 5.0, 4.7, 1.0, 5.0, 4.9])
        result = ts_skew(data, 10)
        assert result < 0, "Left-skewed data should have negative skewness"

    def test_warmup_nan(self):
        """Insufficient data should return NaN."""
        data = np.array([1.0, 2.0])
        result = ts_skew(data, 5)
        assert math.isnan(result)

    def test_window_less_than_3(self):
        """Window < 3 should return NaN (Fisher correction undefined)."""
        data = np.array([1.0, 2.0, 3.0])
        result = ts_skew(data, 2)
        assert math.isnan(result)

    def test_matches_scipy(self):
        """Panel skewness should match scipy reference."""
        from scipy.stats import skew as scipy_skew

        rng = np.random.RandomState(42)
        data = rng.randn(20)
        window = 10

        expected = float(scipy_skew(data[-window:], bias=False))
        result = ts_skew(data, window)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_panel(self):
        """Test panel (multi-instrument) computation."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            rng.randn(30, 3),
            columns=["A", "B", "C"],
        )
        op = TsSkew()
        result = op.compute_panel(df, 10)

        assert result.shape == df.shape
        # First 9 rows should be NaN (warmup)
        assert result.iloc[:9].isna().all().all()
        # Row 9+ should have values
        assert result.iloc[9:].notna().all().all()

    def test_constant_series_nan(self):
        """Constant values should produce NaN (zero std)."""
        data = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = ts_skew(data, 5)
        assert math.isnan(result)
