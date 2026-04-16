# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for risk decomposition (variance budget) pure function."""

import numpy as np

from nautilus_quants.portfolio.monitor.risk_decomposition import compute_risk_budget


def test_vol_shares_sum_to_one():
    """factor_vol_share + specific_vol_share ≈ 1.0."""
    w = np.array([0.3, 0.3, 0.4])
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    cov_b = np.array([[0.04, 0.01], [0.01, 0.02]])
    var_u = np.array([0.01, 0.02, 0.015])
    names = ("f1", "f2")

    result = compute_risk_budget(w, X, cov_b, var_u, names)
    total_share = sum(result["factor_vol_share"].values()) + result["specific_vol_share"]
    assert abs(total_share - 1.0) < 1e-10


def test_total_vol_positive():
    w = np.array([0.5, 0.5])
    X = np.array([[1.0], [1.0]])
    cov_b = np.array([[0.04]])
    var_u = np.array([0.01, 0.01])
    result = compute_risk_budget(w, X, cov_b, var_u, ("beta",))
    assert result["total_vol"] > 0.0


def test_zero_weights_produce_zero_vol():
    w = np.zeros(3)
    X = np.ones((3, 2))
    cov_b = np.eye(2) * 0.04
    var_u = np.array([0.01, 0.01, 0.01])
    result = compute_risk_budget(w, X, cov_b, var_u, ("a", "b"))
    assert result["total_vol"] == 0.0


def test_single_factor_dominance():
    """With one huge factor and tiny specific, factor share ≈ 1."""
    w = np.array([0.5, 0.5])
    X = np.array([[1.0], [1.0]])
    cov_b = np.array([[1.0]])  # very large factor variance
    var_u = np.array([1e-8, 1e-8])  # tiny specific
    result = compute_risk_budget(w, X, cov_b, var_u, ("dominant",))
    assert result["factor_vol_share"]["dominant"] > 0.99


def test_factor_names_in_output():
    w = np.array([0.5, 0.5])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    cov_b = np.eye(2) * 0.04
    var_u = np.array([0.01, 0.01])
    names = ("btc_beta", "size")
    result = compute_risk_budget(w, X, cov_b, var_u, names)
    assert set(result["factor_vol_share"].keys()) == {"btc_beta", "size"}
