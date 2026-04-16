# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for concentration metrics pure function."""

from nautilus_quants.portfolio.monitor.concentration import compute_concentration


def test_empty_portfolio():
    result = compute_concentration({})
    assert result["n_effective"] == 0.0
    assert result["herfindahl"] == 0.0
    assert result["max_abs_weight"] == 0.0


def test_single_position():
    result = compute_concentration({"BTC": 1.0})
    assert result["n_effective"] == 1.0
    assert result["herfindahl"] == 1.0
    assert result["max_abs_weight"] == 1.0


def test_equal_weight_n_effective():
    """N equal-weight positions → n_effective = N."""
    n = 10
    weights = {f"INST_{i}": 0.1 for i in range(n)}
    result = compute_concentration(weights)
    assert abs(result["n_effective"] - n) < 1e-10
    assert abs(result["herfindahl"] - 1.0 / n) < 1e-10


def test_short_positions_counted():
    """Short positions contribute to concentration via |w|."""
    weights = {"A": 0.5, "B": -0.5}
    result = compute_concentration(weights)
    assert result["n_effective"] == 2.0
    assert result["max_abs_weight"] == 0.5


def test_concentrated_portfolio():
    """One dominant position → n_effective close to 1."""
    weights = {"A": 0.9, "B": 0.05, "C": 0.05}
    result = compute_concentration(weights)
    assert result["n_effective"] < 2.0
    assert result["max_abs_weight"] == 0.9


def test_zero_weight_ignored():
    weights = {"A": 0.5, "B": 0.0, "C": 0.5}
    result = compute_concentration(weights)
    assert result["n_effective"] == 2.0
