# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for cross-sectional operators."""

import math

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.cross_sectional import (
    CsVectorNeut,
    cs_demean,
    cs_max,
    cs_min,
    cs_rank,
    cs_scale,
    cs_zscore,
    vector_neut,
)


class TestCsRank:
    """Tests for cs_rank operator."""

    def test_basic_rank(self):
        """Test basic ranking."""
        values = {"A": 1.0, "B": 2.0, "C": 3.0}
        result = cs_rank(values)
        
        # popbo-aligned: rank(method='min', pct=True) → [1/n, 1]
        assert result["A"] == pytest.approx(1 / 3)  # Lowest
        assert result["B"] == pytest.approx(2 / 3)  # Middle
        assert result["C"] == pytest.approx(1.0)  # Highest

    def test_rank_with_ties(self):
        """Test ranking with equal values."""
        values = {"A": 1.0, "B": 1.0, "C": 2.0}
        result = cs_rank(values)
        
        # Both A and B are lower than C
        assert result["C"] == pytest.approx(1.0)

    def test_rank_with_nan(self):
        """Test ranking with NaN values."""
        values = {"A": 1.0, "B": float('nan'), "C": 3.0}
        result = cs_rank(values)
        
        # popbo-aligned: 2 valid values → rank [1/2, 1]
        assert result["A"] == pytest.approx(0.5)
        assert math.isnan(result["B"])
        assert result["C"] == pytest.approx(1.0)

    def test_empty_values(self):
        """Test with empty input."""
        result = cs_rank({})
        assert result == {}

    def test_single_value(self):
        """Test with single value."""
        values = {"A": 5.0}
        result = cs_rank(values)
        # popbo-aligned: single value → rank 1/1 = 1.0
        assert result["A"] == pytest.approx(1.0)


class TestCsZscore:
    """Tests for cs_zscore operator."""

    def test_basic_zscore(self):
        """Test basic z-score calculation."""
        values = {"A": 10.0, "B": 20.0, "C": 30.0}
        result = cs_zscore(values)
        
        # Mean = 20, values are symmetric
        assert result["A"] < 0  # Below mean
        assert result["B"] == pytest.approx(0.0)  # At mean
        assert result["C"] > 0  # Above mean

    def test_zscore_symmetry(self):
        """Test z-score symmetry."""
        values = {"A": 0.0, "B": 10.0, "C": 20.0}
        result = cs_zscore(values)
        
        # A and C should be equidistant from B (mean)
        assert result["A"] == pytest.approx(-result["C"])

    def test_zscore_with_nan(self):
        """Test z-score with NaN values."""
        values = {"A": 10.0, "B": float('nan'), "C": 30.0}
        result = cs_zscore(values)
        
        assert not math.isnan(result["A"])
        assert math.isnan(result["B"])
        assert not math.isnan(result["C"])

    def test_constant_values(self):
        """Test z-score with constant values."""
        values = {"A": 5.0, "B": 5.0, "C": 5.0}
        result = cs_zscore(values)
        
        # All should be 0 (at mean with zero std)
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)
        assert result["C"] == pytest.approx(0.0)


class TestCsScale:
    """Tests for cs_scale operator."""

    def test_basic_scale(self):
        """Test basic scaling."""
        values = {"A": 1.0, "B": 2.0, "C": 3.0}
        result = cs_scale(values)
        
        # Sum of absolute values = 6
        assert result["A"] == pytest.approx(1/6)
        assert result["B"] == pytest.approx(2/6)
        assert result["C"] == pytest.approx(3/6)

    def test_scale_with_negatives(self):
        """Test scaling with negative values."""
        values = {"A": -1.0, "B": 2.0}
        result = cs_scale(values)
        
        # Sum of absolute values = 3
        assert result["A"] == pytest.approx(-1/3)
        assert result["B"] == pytest.approx(2/3)

    def test_scale_sums_to_one(self):
        """Test that absolute values sum to 1."""
        values = {"A": 5.0, "B": -3.0, "C": 2.0}
        result = cs_scale(values)
        
        total = sum(abs(v) for v in result.values())
        assert total == pytest.approx(1.0)


class TestCsDemean:
    """Tests for cs_demean operator."""

    def test_basic_demean(self):
        """Test basic demeaning."""
        values = {"A": 10.0, "B": 20.0, "C": 30.0}
        result = cs_demean(values)
        
        # Mean = 20
        assert result["A"] == pytest.approx(-10.0)
        assert result["B"] == pytest.approx(0.0)
        assert result["C"] == pytest.approx(10.0)

    def test_demean_sums_to_zero(self):
        """Test that demeaned values sum to zero."""
        values = {"A": 5.0, "B": 15.0, "C": 25.0, "D": 35.0}
        result = cs_demean(values)
        
        total = sum(result.values())
        assert total == pytest.approx(0.0)


class TestCsMinMax:
    """Tests for cs_min and cs_max operators."""

    def test_cs_max(self):
        """Test cross-sectional max."""
        values = {"A": 1.0, "B": 5.0, "C": 3.0}
        result = cs_max(values)
        
        # All should get the max value
        assert result["A"] == pytest.approx(5.0)
        assert result["B"] == pytest.approx(5.0)
        assert result["C"] == pytest.approx(5.0)

    def test_cs_min(self):
        """Test cross-sectional min."""
        values = {"A": 1.0, "B": 5.0, "C": 3.0}
        result = cs_min(values)
        
        # All should get the min value
        assert result["A"] == pytest.approx(1.0)
        assert result["B"] == pytest.approx(1.0)
        assert result["C"] == pytest.approx(1.0)


class TestVectorNeut:
    """Tests for vector_neut (cross-sectional orthogonalization)."""

    def test_orthogonality(self):
        """Residual should be orthogonal to y: dot(residual, y_demeaned) ≈ 0."""
        x = {"A": 1.0, "B": 3.0, "C": 5.0, "D": 7.0, "E": 2.0}
        y = {"A": 2.0, "B": 4.0, "C": 6.0, "D": 8.0, "E": 3.0}

        result = vector_neut(x, y)

        # Compute dot(residual, y_demeaned)
        y_mean = np.mean(list(y.values()))
        dot_product = sum(result[k] * (y[k] - y_mean) for k in result if not math.isnan(result[k]))
        assert dot_product == pytest.approx(0.0, abs=1e-10)

    def test_residual_mean_zero(self):
        """Residual should have zero mean (demeaned)."""
        x = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}
        y = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

        result = vector_neut(x, y)

        valid_residuals = [v for v in result.values() if not math.isnan(v)]
        assert np.mean(valid_residuals) == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_correlated(self):
        """When x = a*y + b, residual should be ~0 for all instruments."""
        y = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        x = {k: 2 * v + 10 for k, v in y.items()}  # x = 2y + 10

        result = vector_neut(x, y)

        for k in result:
            assert result[k] == pytest.approx(0.0, abs=1e-10)

    def test_uncorrelated(self):
        """When x and y are uncorrelated, residual ≈ x - mean(x)."""
        x = {"A": 1.0, "B": -1.0, "C": 1.0, "D": -1.0}
        y = {"A": 1.0, "B": 1.0, "C": -1.0, "D": -1.0}

        result = vector_neut(x, y)

        x_mean = np.mean(list(x.values()))
        for k in result:
            expected = x[k] - x_mean
            assert result[k] == pytest.approx(expected, abs=1e-10)

    def test_nan_handling(self):
        """NaN instruments should be excluded from regression."""
        x = {"A": 1.0, "B": float("nan"), "C": 3.0, "D": 4.0, "E": 5.0}
        y = {"A": 2.0, "B": 3.0, "C": 4.0, "D": 5.0, "E": 6.0}

        result = vector_neut(x, y)
        assert math.isnan(result["B"]), "NaN input should produce NaN output"
        # Other values should be valid
        assert not math.isnan(result["A"])
        assert not math.isnan(result["C"])

    def test_insufficient_data(self):
        """Fewer than 3 valid instruments should return all NaN."""
        x = {"A": 1.0, "B": 2.0}
        y = {"A": 3.0, "B": 4.0}

        result = vector_neut(x, y)
        assert all(math.isnan(v) for v in result.values())

    def test_panel(self):
        """Test panel (multi-timestamp) computation."""
        rng = np.random.RandomState(42)
        n_t, n_inst = 10, 5
        cols = [f"INST_{i}" for i in range(n_inst)]
        df_x = pd.DataFrame(rng.randn(n_t, n_inst), columns=cols)
        df_y = pd.DataFrame(rng.randn(n_t, n_inst), columns=cols)

        op = CsVectorNeut()
        result = op.compute_vectorized(df_x, df_y)

        assert result.shape == df_x.shape

        # Each row's residual should be orthogonal to y
        for t in range(n_t):
            r = result.iloc[t].values
            y = df_y.iloc[t].values
            y_d = y - np.mean(y)
            dot_val = np.dot(r, y_d)
            assert dot_val == pytest.approx(0.0, abs=1e-10)

    def test_constant_y_demean_only(self):
        """When y is constant, residual = x - mean(x) (beta=0)."""
        x = {"A": 1.0, "B": 3.0, "C": 5.0, "D": 7.0}
        y = {"A": 2.0, "B": 2.0, "C": 2.0, "D": 2.0}

        result = vector_neut(x, y)

        x_mean = np.mean(list(x.values()))
        for k in result:
            assert result[k] == pytest.approx(x[k] - x_mean, abs=1e-10)
