# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for time-series operators."""

import math

import numpy as np
import pytest

from nautilus_quants.factors.operators.time_series import (
    Delay,
    Delta,
    TsArgmax,
    TsArgmin,
    TsMax,
    TsMean,
    TsMin,
    TsRank,
    TsStd,
    TsSum,
    delay,
    delta,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
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
        assert result == pytest.approx(0.0)  # lowest = rank 0.0

    def test_middle_rank(self):
        """Test rank when current value is in middle."""
        data = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        result = ts_rank(data, 5)
        # 3.0 is greater than 1.0 and 2.0 (2 values)
        # rank = 2 / 4 = 0.5
        assert result == pytest.approx(0.5)


class TestTsArgmaxArgmin:
    """Tests for ts_argmax and ts_argmin operators."""

    def test_argmax_most_recent(self):
        """Test argmax when max is most recent."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_argmax(data, 5)
        assert result == pytest.approx(0.0)  # max is at position 0 (most recent)

    def test_argmax_oldest(self):
        """Test argmax when max is oldest."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_argmax(data, 5)
        assert result == pytest.approx(4.0)  # max is 4 positions ago

    def test_argmin_most_recent(self):
        """Test argmin when min is most recent."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_argmin(data, 5)
        assert result == pytest.approx(0.0)  # min is at position 0

    def test_argmin_oldest(self):
        """Test argmin when min is oldest."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_argmin(data, 5)
        assert result == pytest.approx(4.0)  # min is 4 positions ago


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
