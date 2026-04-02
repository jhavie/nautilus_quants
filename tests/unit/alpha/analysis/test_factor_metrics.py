# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for extended factor signal quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.analysis.config import MetricsConfig
from nautilus_quants.alpha.analysis.report import (
    compute_coverage,
    compute_ic_ar1,
    compute_ic_half_life,
    compute_ic_linearity,
    compute_monotonicity,
    compute_win_rate,
)


# ── Helpers ──


def _make_ic_df(
    values: dict[str, list[float]],
    start: str = "2024-01-01",
    freq: str = "h",
) -> pd.DataFrame:
    """Create IC DataFrame from dict of {period: values}."""
    n = len(next(iter(values.values())))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(values, index=idx)


def _make_factor_data(
    n_dates: int = 100,
    n_assets: int = 10,
    n_quantiles: int = 5,
    start: str = "2024-01-01",
    freq: str = "h",
    periods: list[str] | None = None,
) -> pd.DataFrame:
    """Create alphalens-style factor_data DataFrame."""
    periods = periods or ["1h", "4h", "24h"]
    dates = pd.date_range(start, periods=n_dates, freq=freq)
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    tuples = [(d, a) for d in dates for a in assets]
    idx = pd.MultiIndex.from_tuples(tuples, names=["date", "asset"])
    n = len(tuples)
    rng = np.random.RandomState(42)
    data: dict[str, np.ndarray] = {}
    for p in periods:
        data[p] = rng.randn(n) * 0.01
    data["factor"] = rng.randn(n)
    data["factor_quantile"] = np.tile(
        np.repeat(range(1, n_quantiles + 1), n_assets // n_quantiles), n_dates,
    )
    return pd.DataFrame(data, index=idx)


def _make_monotonic_factor_data(
    direction: float = 1.0,
    n_dates: int = 200,
    n_assets: int = 50,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Create factor_data with perfectly monotonic quantile returns.

    direction=1.0: Q1 lowest return, Q5 highest (positive monotonicity)
    direction=-1.0: Q1 highest return, Q5 lowest (negative monotonicity)
    """
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="h")
    assets = [f"A_{i}" for i in range(n_assets)]
    tuples = [(d, a) for d in dates for a in assets]
    idx = pd.MultiIndex.from_tuples(tuples, names=["date", "asset"])
    n = len(tuples)

    # Assign quantiles evenly
    quantiles = np.tile(
        np.repeat(range(1, n_quantiles + 1), n_assets // n_quantiles),
        n_dates,
    )
    # Forward returns proportional to quantile
    base_ret = (quantiles - (n_quantiles + 1) / 2) * 0.002 * direction
    noise = np.random.RandomState(42).randn(n) * 0.0005

    return pd.DataFrame(
        {
            "1h": base_ret + noise,
            "4h": base_ret * 2 + noise,
            "factor": np.random.RandomState(42).randn(n),
            "factor_quantile": quantiles,
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════
# Factor Signal Quality Metrics
# ═══════════════════════════════════════════════════════════════


class TestWinRate:
    """Tests for compute_win_rate."""

    def test_all_positive(self):
        ic = _make_ic_df({"1h": [0.1, 0.2, 0.3, 0.4, 0.5]})
        result = compute_win_rate(ic)
        assert result["1h"] == pytest.approx(1.0)

    def test_all_negative(self):
        ic = _make_ic_df({"1h": [-0.1, -0.2, -0.3]})
        result = compute_win_rate(ic)
        assert result["1h"] == pytest.approx(0.0)

    def test_mixed(self):
        ic = _make_ic_df({"1h": [0.1, -0.05, 0.08, -0.11, 0.06, -0.02]})
        result = compute_win_rate(ic)
        # 3 positive out of 6
        assert result["1h"] == pytest.approx(3 / 6)

    def test_with_nan(self):
        ic = _make_ic_df({"1h": [0.1, np.nan, -0.05, 0.08, np.nan, -0.02]})
        result = compute_win_rate(ic)
        # 2 positive out of 4 non-NaN
        assert result["1h"] == pytest.approx(2 / 4)

    def test_per_period_independent(self):
        ic = _make_ic_df({
            "1h": [0.1, 0.2, -0.1],
            "4h": [-0.1, -0.2, -0.3],
        })
        result = compute_win_rate(ic)
        assert result["1h"] == pytest.approx(2 / 3)
        assert result["4h"] == pytest.approx(0.0)

    def test_zero_ic_not_counted_as_win(self):
        ic = _make_ic_df({"1h": [0.0, 0.0, 0.1]})
        result = compute_win_rate(ic)
        assert result["1h"] == pytest.approx(1 / 3)


class TestCoverage:
    """Tests for compute_coverage."""

    def test_full_coverage(self):
        fd = _make_factor_data(n_dates=10, n_assets=5)
        result = compute_coverage(fd, total_timestamps=10, total_assets=5)
        assert result == pytest.approx(1.0)

    def test_partial_coverage(self):
        fd = _make_factor_data(n_dates=10, n_assets=5)
        # Simulate larger universe
        result = compute_coverage(fd, total_timestamps=20, total_assets=5)
        assert result == pytest.approx(0.5)

    def test_empty_factor_data(self):
        fd = _make_factor_data(n_dates=10, n_assets=5).iloc[:0]
        result = compute_coverage(fd, total_timestamps=10, total_assets=5)
        assert result == pytest.approx(0.0)


class TestIcLinearity:
    """Tests for compute_ic_linearity (R² of cumulative IC)."""

    def test_constant_positive_ic(self):
        """Constant IC → perfect linear cumulative IC → R² ≈ 1.0."""
        ic = _make_ic_df({"1h": [0.05] * 200})
        result = compute_ic_linearity(ic)
        assert result["1h"] == pytest.approx(1.0, abs=0.001)

    def test_random_ic(self):
        """IID random IC → cumulative IC is random walk → low R²."""
        rng = np.random.RandomState(42)
        ic = _make_ic_df({"1h": rng.randn(500).tolist()})
        result = compute_ic_linearity(ic)
        # Random walk cumsum has R² typically 0.0-0.5
        assert result["1h"] < 0.7

    def test_structural_break(self):
        """Positive then negative IC → cumulative IC bends → lower R²."""
        vals = [0.05] * 100 + [-0.05] * 100
        ic = _make_ic_df({"1h": vals})
        result = compute_ic_linearity(ic)
        # Bent curve has lower R² than straight line
        assert result["1h"] < 0.95

    def test_insufficient_data(self):
        ic = _make_ic_df({"1h": [0.1, 0.2]})
        result = compute_ic_linearity(ic)
        assert np.isnan(result["1h"])


class TestIcAr1:
    """Tests for compute_ic_ar1 (lag-1 autocorrelation)."""

    def test_high_persistence(self):
        """AR(1) with rho=0.9 → AR(1) close to 0.9."""
        rng = np.random.RandomState(42)
        n = 2000
        x = np.zeros(n)
        x[0] = rng.randn()
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.randn() * 0.1
        ic = _make_ic_df({"1h": x.tolist()})
        result = compute_ic_ar1(ic)
        assert result["1h"] == pytest.approx(0.9, abs=0.05)

    def test_iid_near_zero(self):
        """IID noise → AR(1) ≈ 0."""
        rng = np.random.RandomState(42)
        ic = _make_ic_df({"1h": rng.randn(1000).tolist()})
        result = compute_ic_ar1(ic)
        assert abs(result["1h"]) < 0.1

    def test_insufficient_data(self):
        ic = _make_ic_df({"1h": [0.1, 0.2]})
        result = compute_ic_ar1(ic)
        assert np.isnan(result["1h"])


class TestIcHalfLife:
    """Tests for compute_ic_half_life."""

    def test_fast_decay(self):
        """Highly autocorrelated IC → finite half-life."""
        rng = np.random.RandomState(42)
        n = 2000
        # AR(1) with rho=0.95 → strong autocorrelation, fast-ish decay
        ic_vals = np.zeros(n)
        ic_vals[0] = rng.randn()
        for i in range(1, n):
            ic_vals[i] = 0.95 * ic_vals[i - 1] + rng.randn() * 0.1

        ic = _make_ic_df({"1h": ic_vals.tolist()})
        result = compute_ic_half_life(ic)
        # Should get a finite positive half-life
        assert np.isfinite(result["1h"])
        assert result["1h"] > 0

    def test_no_autocorrelation(self):
        """IID noise → NaN or very large half-life."""
        rng = np.random.RandomState(42)
        ic = _make_ic_df({"1h": rng.randn(500).tolist()})
        result = compute_ic_half_life(ic)
        # Either NaN (fit fails) or very large
        assert np.isnan(result["1h"]) or result["1h"] > 100

    def test_insufficient_data(self):
        """Too few data points → NaN."""
        ic = _make_ic_df({"1h": [0.1, 0.2, 0.3]})
        result = compute_ic_half_life(ic)
        assert np.isnan(result["1h"])


class TestMonotonicity:
    """Tests for compute_monotonicity."""

    def test_perfect_positive(self):
        fd = _make_monotonic_factor_data(direction=1.0)
        result = compute_monotonicity(fd)
        assert result["1h"] == pytest.approx(1.0, abs=0.05)

    def test_perfect_negative(self):
        fd = _make_monotonic_factor_data(direction=-1.0)
        result = compute_monotonicity(fd)
        assert result["1h"] == pytest.approx(-1.0, abs=0.05)

    def test_per_period(self):
        fd = _make_monotonic_factor_data(direction=1.0)
        result = compute_monotonicity(fd)
        # Both periods should be monotonic
        assert result["1h"] == pytest.approx(1.0, abs=0.05)
        assert result["4h"] == pytest.approx(1.0, abs=0.05)


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.factor_metrics is False

    def test_custom(self):
        cfg = MetricsConfig(factor_metrics=True)
        assert cfg.factor_metrics is True
