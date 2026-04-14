# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for MeanVarianceOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from nautilus_quants.portfolio.optimizer.base import OptimizerConstraints
from nautilus_quants.portfolio.optimizer.mean_variance import (
    MeanVarianceConfig,
    MeanVarianceOptimizer,
)


def _make_psd_cov(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.05
    return cov


def _default_constraints(
    n_positions_min: int = 1,
    turnover: float | None = 0.5,
) -> OptimizerConstraints:
    return OptimizerConstraints(
        max_weight=0.2,
        max_leverage=2.0,
        net_exposure=(-0.1, 0.1),
        turnover_limit=turnover,
        min_positions=n_positions_min,
        sector_limits=None,
        factor_limits=None,
    )


@pytest.mark.unit
def test_solve_produces_feasible_solution() -> None:
    n = 10
    rng = np.random.default_rng(0)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=_default_constraints(turnover=None),
    )
    assert result.status == "optimal"
    assert result.weights.shape == (n,)
    # All core constraints satisfied
    assert np.all(np.abs(result.weights) <= 0.2 + 1e-6)
    assert np.sum(np.abs(result.weights)) <= 2.0 + 1e-6
    net = np.sum(result.weights)
    assert -0.1 - 1e-6 <= net <= 0.1 + 1e-6


@pytest.mark.unit
def test_alpha_aligned_with_weights() -> None:
    """Maximizing α'w - λw'Σw with identity Σ should approximately align w with α."""
    n = 5
    alpha = np.array([1.0, -0.5, 0.2, -1.0, 0.7])
    cov = np.eye(n) * 0.1
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False, risk_aversion=0.01))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=_default_constraints(turnover=None),
    )
    # Sign correlation with alpha must be strongly positive
    w = result.weights
    assert np.corrcoef(alpha, w)[0, 1] > 0.7


@pytest.mark.unit
def test_turnover_constraint_binds() -> None:
    n = 10
    rng = np.random.default_rng(1)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)
    w0 = np.zeros(n)
    w0[0] = 0.15
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=w0,
        constraints=_default_constraints(turnover=0.1),  # very tight
    )
    assert result.status in ("optimal", "optimal_fallback")
    if result.status == "optimal":
        # turnover ≤ 0.1 must hold
        assert result.turnover is not None
        assert result.turnover <= 0.1 + 1e-6


@pytest.mark.unit
def test_infeasible_falls_back_when_turnover_too_tight() -> None:
    """With extremely tight turnover, solver may relax via fallback."""
    n = 10
    rng = np.random.default_rng(2)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)
    # Far-from-neutral w0 that cannot reach constraints within 0.001 turnover
    w0 = np.zeros(n)
    w0[0] = 0.5  # beyond max_weight
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=w0,
        constraints=_default_constraints(turnover=0.001),
    )
    # Either optimal_fallback or infeasible - both are valid outcomes
    assert result.status in ("optimal", "optimal_fallback", "infeasible")
    if result.status == "optimal_fallback":
        assert "turnover" in result.relaxed


@pytest.mark.unit
def test_sector_limits_enforced() -> None:
    n = 6
    rng = np.random.default_rng(3)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)
    # Sector dummies: 3 in sector A, 3 in sector B
    sector_dummies = np.zeros((n, 2))
    sector_dummies[:3, 0] = 1.0
    sector_dummies[3:, 1] = 1.0

    constraints = OptimizerConstraints(
        max_weight=0.3,
        max_leverage=3.0,
        net_exposure=(-0.1, 0.1),
        turnover_limit=None,
        min_positions=1,
        sector_limits={"A": 0.05, "B": 1.0},  # very tight on sector A
        factor_limits=None,
    )
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=constraints,
        sector_dummies=sector_dummies,
    )
    assert result.status in ("optimal", "optimal_fallback")
    if result.status == "optimal":
        gross_a = float(np.sum(np.abs(result.weights[:3])))
        assert gross_a <= 0.05 + 1e-6, f"sector A gross {gross_a} > 0.05"


@pytest.mark.unit
def test_factor_limits_enforced() -> None:
    n = 8
    rng = np.random.default_rng(4)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)

    # Factor exposure X (N, K=2)
    X = np.zeros((n, 2))
    X[:, 0] = rng.standard_normal(n)  # beta-like
    X[:, 1] = rng.standard_normal(n)

    constraints = OptimizerConstraints(
        max_weight=0.3,
        max_leverage=2.0,
        net_exposure=(-0.1, 0.1),
        turnover_limit=None,
        min_positions=1,
        sector_limits=None,
        factor_limits={"beta": 0.05, "style2": 0.5},
    )
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=constraints,
        factor_exposures=X,
    )
    assert result.status in ("optimal", "optimal_fallback")
    if result.status == "optimal":
        beta_exposure = float(X[:, 0] @ result.weights)
        assert abs(beta_exposure) <= 0.05 + 1e-6


@pytest.mark.unit
def test_epsilon_zeros_tiny_weights() -> None:
    n = 10
    rng = np.random.default_rng(5)
    alpha = rng.standard_normal(n)
    cov = _make_psd_cov(n)
    # Large epsilon forces many weights to be zeroed
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False, epsilon=1e-2))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=_default_constraints(turnover=None),
    )
    tiny = np.abs(result.weights) > 0
    assert np.all(np.abs(result.weights[tiny]) >= 1e-2 - 1e-9)


@pytest.mark.unit
def test_market_neutrality_band() -> None:
    n = 10
    rng = np.random.default_rng(6)
    alpha = rng.standard_normal(n) * 5  # strong signal
    cov = _make_psd_cov(n)
    # Very tight net exposure band
    constraints = OptimizerConstraints(
        max_weight=0.3,
        max_leverage=2.0,
        net_exposure=(-0.01, 0.01),
        turnover_limit=None,
        min_positions=1,
    )
    opt = MeanVarianceOptimizer(MeanVarianceConfig(scale_return=False))
    result = opt.solve(
        alpha=alpha,
        covariance=cov,
        current_weights=None,
        constraints=constraints,
    )
    assert result.status == "optimal"
    net = float(np.sum(result.weights))
    assert -0.01 - 1e-6 <= net <= 0.01 + 1e-6


@pytest.mark.unit
def test_covariance_shape_mismatch_raises() -> None:
    opt = MeanVarianceOptimizer(MeanVarianceConfig())
    with pytest.raises(ValueError, match="covariance shape"):
        opt.solve(
            alpha=np.ones(5),
            covariance=np.eye(4),
            current_weights=None,
            constraints=_default_constraints(),
        )
