# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for WorldQuant BRAIN-compatible cross-sectional operators."""

import math

import pytest

from nautilus_quants.factors.operators.cross_sectional import (
    demean,
    normalize,
    quantile,
    rank,
    scale,
    scale_down,
    winsorize,
    zscore,
)


class TestNormalize:
    """Tests for normalize operator (WorldQuant BRAIN style)."""

    def test_normalize_without_std(self):
        """WorldQuant example: x = [3,5,6,2], mean=4 → [-1,1,2,-2]"""
        values = {"a": 3.0, "b": 5.0, "c": 6.0, "d": 2.0}
        result = normalize(values, use_std=False, limit=0.0)

        assert result["a"] == pytest.approx(-1.0)
        assert result["b"] == pytest.approx(1.0)
        assert result["c"] == pytest.approx(2.0)
        assert result["d"] == pytest.approx(-2.0)

    def test_normalize_with_std(self):
        """WorldQuant example: std=1.82 → [-0.55, 0.55, 1.1, -1.1]"""
        values = {"a": 3.0, "b": 5.0, "c": 6.0, "d": 2.0}
        result = normalize(values, use_std=True, limit=0.0)

        # Population std of [-1, 1, 2, -2] is sqrt(10/4) ≈ 1.58
        # Expected: [-1/1.58, 1/1.58, 2/1.58, -2/1.58] ≈ [-0.63, 0.63, 1.26, -1.26]
        assert result["a"] < 0
        assert result["b"] > 0
        assert result["c"] > result["b"]  # Highest
        assert result["d"] < result["a"]  # Lowest

    def test_normalize_with_limit(self):
        """Test normalize with limit clipping."""
        values = {"a": 0.0, "b": 10.0, "c": 100.0}  # Mean ≈ 36.67
        result = normalize(values, use_std=False, limit=20.0)

        # c would be 100 - 36.67 = 63.33, but clipped to 20
        assert result["c"] == pytest.approx(20.0)
        # a would be 0 - 36.67 = -36.67, but clipped to -20
        assert result["a"] == pytest.approx(-20.0)

    def test_normalize_with_nan(self):
        """Test normalize handles NaN values."""
        values = {"a": 3.0, "b": float("nan"), "c": 5.0}
        result = normalize(values, use_std=False)

        assert not math.isnan(result["a"])
        assert math.isnan(result["b"])
        assert not math.isnan(result["c"])

    def test_normalize_empty(self):
        """Test normalize with empty input."""
        result = normalize({})
        assert result == {}

    def test_normalize_single_value(self):
        """Test normalize with single value returns NaN."""
        values = {"a": 5.0}
        result = normalize(values)
        assert math.isnan(result["a"])


class TestWinsorize:
    """Tests for winsorize operator (WorldQuant BRAIN style)."""

    def test_winsorize_basic(self):
        """WorldQuant example: x = [2,4,5,6,3,8,10], std=1 → [2.81,4,5,6,3,8,8.03]"""
        values = {"a": 2.0, "b": 4.0, "c": 5.0, "d": 6.0, "e": 3.0, "f": 8.0, "g": 10.0}
        result = winsorize(values, std_mult=1.0)

        # Mean ≈ 5.43, population std ≈ 2.61
        # lower = 5.43 - 2.61 ≈ 2.82, upper = 5.43 + 2.61 ≈ 8.04

        # Values within range should be unchanged
        assert result["b"] == pytest.approx(4.0)
        assert result["c"] == pytest.approx(5.0)
        assert result["d"] == pytest.approx(6.0)
        assert result["e"] == pytest.approx(3.0)
        assert result["f"] == pytest.approx(8.0)

        # Values at extremes should be clipped
        assert result["a"] > 2.0  # Clipped up to lower bound
        assert result["g"] < 10.0  # Clipped down to upper bound

    def test_winsorize_default_std(self):
        """Test winsorize with default std=4 (very wide bounds)."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        result = winsorize(values)  # Default std_mult=4.0

        # With 4 std, almost all values should be unchanged
        for k in values:
            assert result[k] == pytest.approx(values[k])

    def test_winsorize_with_nan(self):
        """Test winsorize handles NaN values."""
        values = {"a": 1.0, "b": float("nan"), "c": 10.0}
        result = winsorize(values, std_mult=1.0)

        assert not math.isnan(result["a"])
        assert math.isnan(result["b"])
        assert not math.isnan(result["c"])

    def test_winsorize_constant_values(self):
        """Test winsorize with constant values (std=0)."""
        values = {"a": 5.0, "b": 5.0, "c": 5.0}
        result = winsorize(values, std_mult=1.0)

        # When std is 0, values should be unchanged
        assert result["a"] == pytest.approx(5.0)
        assert result["b"] == pytest.approx(5.0)
        assert result["c"] == pytest.approx(5.0)


class TestScaleDown:
    """Tests for scale_down operator (WorldQuant BRAIN style)."""

    def test_scale_down_basic(self):
        """WorldQuant example: x = [15,7,0,20] → [0.75, 0.35, 0, 1]"""
        values = {"a": 15.0, "b": 7.0, "c": 0.0, "d": 20.0}
        result = scale_down(values, constant=0.0)

        assert result["a"] == pytest.approx(0.75)
        assert result["b"] == pytest.approx(0.35)
        assert result["c"] == pytest.approx(0.0)
        assert result["d"] == pytest.approx(1.0)

    def test_scale_down_with_constant(self):
        """WorldQuant example: x = [15,7,0,20], constant=1 → [-0.25,-0.65,-1,0]"""
        values = {"a": 15.0, "b": 7.0, "c": 0.0, "d": 20.0}
        result = scale_down(values, constant=1.0)

        assert result["a"] == pytest.approx(-0.25)
        assert result["b"] == pytest.approx(-0.65)
        assert result["c"] == pytest.approx(-1.0)
        assert result["d"] == pytest.approx(0.0)

    def test_scale_down_range_is_0_to_1(self):
        """Test that scale_down output is in [0, 1] when constant=0."""
        values = {"a": 100.0, "b": 50.0, "c": 200.0, "d": 0.0}
        result = scale_down(values, constant=0.0)

        for k in values:
            assert 0.0 <= result[k] <= 1.0

        # Min should map to 0, max to 1
        assert result["d"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)

    def test_scale_down_with_nan(self):
        """Test scale_down handles NaN values."""
        values = {"a": 0.0, "b": float("nan"), "c": 10.0}
        result = scale_down(values)

        assert not math.isnan(result["a"])
        assert math.isnan(result["b"])
        assert not math.isnan(result["c"])

    def test_scale_down_constant_values(self):
        """Test scale_down with constant values (range=0)."""
        values = {"a": 5.0, "b": 5.0, "c": 5.0}
        result = scale_down(values, constant=0.0)

        # When range is 0, all values should be 0.5
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(0.5)


class TestQuantile:
    """Tests for quantile operator (WorldQuant BRAIN style)."""

    def test_quantile_gaussian(self):
        """Test quantile with gaussian driver."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        result = quantile(values, driver="gaussian", sigma=1.0)

        # After gaussian quantile transform:
        # - Lowest value should be negative
        # - Highest value should be positive
        # - Values should be symmetric around 0
        assert result["a"] < 0
        assert result["e"] > 0
        assert result["a"] == pytest.approx(-result["e"], abs=0.1)

    def test_quantile_uniform(self):
        """Test quantile with uniform driver."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = quantile(values, driver="uniform", sigma=1.0)

        # Uniform just shifts by 0.5
        # All values should be in [-0.5, 0.5] range
        for k in values:
            assert -0.5 <= result[k] <= 0.5

    def test_quantile_sigma_scaling(self):
        """Test that sigma parameter scales the output."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0}
        result1 = quantile(values, driver="gaussian", sigma=1.0)
        result2 = quantile(values, driver="gaussian", sigma=2.0)

        # sigma=2 should give values 2x larger than sigma=1
        assert result2["a"] == pytest.approx(result1["a"] * 2.0)
        assert result2["c"] == pytest.approx(result1["c"] * 2.0)

    def test_quantile_with_nan(self):
        """Test quantile handles NaN values."""
        values = {"a": 1.0, "b": float("nan"), "c": 3.0}
        result = quantile(values, driver="gaussian")

        assert not math.isnan(result["a"])
        assert math.isnan(result["b"])
        assert not math.isnan(result["c"])


class TestAliases:
    """Tests for WorldQuant-compatible aliases."""

    def test_rank_alias(self):
        """Test rank alias works like cs_rank."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = rank(values)

        # popbo-aligned: rank(method='min', pct=True) → [1/n, 1]
        assert result["a"] == pytest.approx(1 / 3)
        assert result["b"] == pytest.approx(2 / 3)
        assert result["c"] == pytest.approx(1.0)

    def test_zscore_alias(self):
        """Test zscore alias works like cs_zscore."""
        values = {"a": 10.0, "b": 20.0, "c": 30.0}
        result = zscore(values)

        # Mean = 20, so b should be 0
        assert result["b"] == pytest.approx(0.0)
        assert result["a"] < 0
        assert result["c"] > 0

    def test_scale_alias(self):
        """Test scale alias works like cs_scale."""
        values = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = scale(values)

        # Sum of absolute values = 6
        assert result["a"] == pytest.approx(1 / 6)
        assert result["b"] == pytest.approx(2 / 6)
        assert result["c"] == pytest.approx(3 / 6)

    def test_demean_alias(self):
        """Test demean alias works like cs_demean."""
        values = {"a": 10.0, "b": 20.0, "c": 30.0}
        result = demean(values)

        # Mean = 20
        assert result["a"] == pytest.approx(-10.0)
        assert result["b"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(10.0)
