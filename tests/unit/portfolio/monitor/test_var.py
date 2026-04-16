# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Historical Simulation VaR/CVaR pure function."""

import numpy as np
import pytest

from nautilus_quants.portfolio.monitor.var import compute_historical_var


def test_normal_var_approximation():
    """For N(0, σ²), 95% VaR ≈ 1.645σ."""
    rng = np.random.default_rng(42)
    sigma = 0.02
    n_inst = 1
    t = 10000
    returns = rng.normal(0, sigma, (t, n_inst))
    weights = np.array([1.0])
    result = compute_historical_var(returns, weights, alpha=0.05)
    assert abs(result["var_pct"] - 1.645 * sigma) < 0.002


def test_cvar_greater_than_var():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, (5000, 3))
    weights = np.array([0.4, 0.3, 0.3])
    result = compute_historical_var(returns, weights, alpha=0.05)
    assert result["cvar_pct"] >= result["var_pct"]


def test_lookback_truncation():
    """Only last `lookback` rows are used."""
    rng = np.random.default_rng(42)
    # First 900 rows: zero returns, last 100: volatile
    calm = np.zeros((900, 2))
    volatile = rng.normal(0, 0.05, (100, 2))
    returns = np.vstack([calm, volatile])
    weights = np.array([0.5, 0.5])

    full = compute_historical_var(returns, weights, alpha=0.05, lookback=None)
    short = compute_historical_var(returns, weights, alpha=0.05, lookback=100)
    assert short["var_pct"] > full["var_pct"]
    assert short["sample_count"] == 100


def test_dict_weights_with_instruments():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, (200, 3))
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    instruments = ("A", "B", "C")
    result = compute_historical_var(returns, weights, instruments=instruments, alpha=0.05)
    assert result["var_pct"] > 0


def test_dict_weights_without_instruments_raises():
    with pytest.raises(ValueError, match="instruments required"):
        compute_historical_var(np.zeros((10, 2)), {"A": 0.5}, alpha=0.05)


def test_equity_conversion():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, (500, 1))
    weights = np.array([1.0])
    result = compute_historical_var(returns, weights, alpha=0.05, equity=100000.0)
    assert result["var_usdt"] is not None
    assert abs(result["var_usdt"] - result["var_pct"] * 100000.0) < 1e-6


def test_insufficient_data_returns_zeros():
    returns = np.zeros((5, 2))
    weights = np.array([0.5, 0.5])
    result = compute_historical_var(returns, weights, alpha=0.05)
    assert result["var_pct"] == 0.0
    assert result["sample_count"] == 5
