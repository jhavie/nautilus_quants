# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for P&L attribution pure function."""

import numpy as np
import pytest

from nautilus_quants.portfolio.monitor.attribution import compute_pnl_attribution


@pytest.fixture()
def simple_data():
    """3 instruments, 2 factors, known exact decomposition."""
    w = np.array([0.4, 0.3, 0.3])
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])  # (3, 2)
    f_t = np.array([0.02, -0.01])  # factor returns
    u_t = np.array([0.001, 0.002, -0.001])  # specific returns
    factor_names = ("momentum", "value")
    return w, X, f_t, u_t, factor_names


def test_total_equals_sum_of_parts(simple_data):
    w, X, f_t, u_t, names = simple_data
    result = compute_pnl_attribution(w, X, f_t, u_t, names)
    factor_sum = sum(result["by_factor"].values())
    assert abs(result["period_return_total"] - (factor_sum + result["specific"])) < 1e-12


def test_factor_contributions_match_manual(simple_data):
    w, X, f_t, u_t, names = simple_data
    result = compute_pnl_attribution(w, X, f_t, u_t, names)
    # exposure = X' w = [0.4*1+0.3*0+0.3*0.5, 0.4*0+0.3*1+0.3*0.5] = [0.55, 0.45]
    assert abs(result["by_factor"]["momentum"] - 0.55 * 0.02) < 1e-12
    assert abs(result["by_factor"]["value"] - 0.45 * (-0.01)) < 1e-12


def test_specific_contribution(simple_data):
    w, X, f_t, u_t, names = simple_data
    result = compute_pnl_attribution(w, X, f_t, u_t, names)
    expected = float(np.sum(w * u_t))
    assert abs(result["specific"] - expected) < 1e-12


def test_zero_weights_produce_zero_return():
    w = np.zeros(3)
    X = np.ones((3, 2))
    f_t = np.array([0.05, -0.03])
    u_t = np.array([0.01, 0.02, -0.01])
    result = compute_pnl_attribution(w, X, f_t, u_t, ("a", "b"))
    assert result["period_return_total"] == 0.0
    assert result["specific"] == 0.0


def test_single_factor():
    w = np.array([0.5, 0.5])
    X = np.array([[1.0], [1.0]])
    f_t = np.array([0.03])
    u_t = np.array([0.0, 0.0])
    result = compute_pnl_attribution(w, X, f_t, u_t, ("beta",))
    assert abs(result["by_factor"]["beta"] - 0.03) < 1e-12
