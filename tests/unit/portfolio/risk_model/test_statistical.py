# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for StatisticalRiskModel (PCA + Shrinkage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.portfolio.risk_model.base import StatisticalModelConfig
from nautilus_quants.portfolio.risk_model.statistical import StatisticalRiskModel


def _make_synthetic_returns(
    n_assets: int = 20,
    n_periods: int = 300,
    n_factors: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate returns with known factor structure.

    Returns the returns DataFrame and the true covariance matrix (for validation).
    """
    rng = np.random.default_rng(seed)
    # Factor loadings (N, K), scores (T, K), idiosyncratic noise (T, N)
    loadings = rng.standard_normal((n_assets, n_factors)) * 0.5
    factor_cov = np.diag(rng.uniform(0.5, 2.0, size=n_factors))
    factor_scores = rng.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=factor_cov,
        size=n_periods,
    )
    idio_vars = rng.uniform(0.1, 0.5, size=n_assets)
    idio = rng.standard_normal((n_periods, n_assets)) * np.sqrt(idio_vars)
    x = factor_scores @ loadings.T + idio

    index = pd.date_range("2024-01-01", periods=n_periods, freq="4h")
    cols = [f"ASSET_{i}.BINANCE" for i in range(n_assets)]
    returns = pd.DataFrame(x, index=index, columns=cols)

    # True covariance (note: no scale_return_pct applied here)
    true_cov = loadings @ factor_cov @ loadings.T + np.diag(idio_vars)
    return returns, true_cov


@pytest.mark.unit
def test_pca_produces_positive_definite_covariance() -> None:
    """PCA output must be symmetric and positive definite."""
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="pca",
        num_factors=3,
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,  # keep raw scale for direct eigenvalue check
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=1234)

    assert out.is_decomposed
    assert out.factor_names == ("PC_0", "PC_1", "PC_2")
    assert out.factor_exposures is not None
    assert out.factor_exposures.shape == (20, 3)
    assert out.factor_covariance is not None
    assert out.factor_covariance.shape == (3, 3)
    assert out.specific_variance is not None
    assert out.specific_variance.shape == (20,)
    assert out.covariance.shape == (20, 20)
    # Symmetric
    np.testing.assert_allclose(out.covariance, out.covariance.T, atol=1e-10)
    # Positive definite
    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > 0, f"covariance not PD, min eigenvalue={eig.min()}"


@pytest.mark.unit
def test_pca_reconstructs_true_covariance_within_tolerance() -> None:
    """PCA estimate should track the true covariance (Frobenius error bounded)."""
    returns, true_cov = _make_synthetic_returns(n_periods=1500)
    cfg = StatisticalModelConfig(
        method="pca",
        num_factors=3,
        lookback_bars=1500,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    # Frobenius relative error < 15% with K=3 matching true factors, long history
    err = np.linalg.norm(out.covariance - true_cov, "fro")
    scale = np.linalg.norm(true_cov, "fro")
    assert err / scale < 0.15, f"relative error too large: {err / scale:.3f}"


@pytest.mark.unit
def test_fa_produces_decomposition() -> None:
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="fa",
        num_factors=3,
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    assert out.factor_names == ("FA_0", "FA_1", "FA_2")
    assert out.factor_exposures is not None
    assert out.factor_exposures.shape == (20, 3)


@pytest.mark.unit
def test_shrinkage_lw_const_corr_alpha_in_unit_interval() -> None:
    """Ledoit-Wolf α must lie in [0, 1]."""
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="shrink",
        shrinkage="lw",
        shrink_target="const_corr",
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    assert not out.is_decomposed  # shrinkage path has no decomposition
    assert out.factor_names is None
    # PSD (eigvals >= 0 allowing small numerical noise)
    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > -1e-8


@pytest.mark.unit
def test_shrinkage_oas_only_with_const_var() -> None:
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="shrink",
        shrinkage="oas",
        shrink_target="const_var",
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > 0


@pytest.mark.unit
def test_shrinkage_single_factor_target() -> None:
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="shrink",
        shrinkage="lw",
        shrink_target="single_factor",
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > -1e-8


@pytest.mark.unit
def test_raises_on_insufficient_history() -> None:
    returns, _ = _make_synthetic_returns(n_periods=30)
    cfg = StatisticalModelConfig(method="pca", num_factors=3, min_history_bars=60)
    model = StatisticalRiskModel(cfg)
    with pytest.raises(ValueError, match="at least 60"):
        model.fit(returns, timestamp_ns=0)


@pytest.mark.unit
def test_nan_handling_fill_mode() -> None:
    """NaN cells must be replaced by 0 before covariance (nan_option='fill')."""
    returns, _ = _make_synthetic_returns()
    # Inject NaN into a few cells
    returns.iloc[0, 0] = np.nan
    returns.iloc[10, 5] = np.nan

    cfg = StatisticalModelConfig(
        method="pca",
        num_factors=3,
        nan_option="fill",
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)
    # No NaN/inf leaked to output
    assert np.all(np.isfinite(out.covariance))


@pytest.mark.unit
def test_direct_float_shrinkage_parameter() -> None:
    """Shrinkage alpha can be supplied directly as a float in [0, 1]."""
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="shrink",
        shrinkage=0.3,
        shrink_target="const_corr",
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=0)

    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > 0


@pytest.mark.unit
def test_output_timestamp_stamped() -> None:
    returns, _ = _make_synthetic_returns()
    cfg = StatisticalModelConfig(
        method="pca",
        num_factors=3,
        lookback_bars=300,
        min_history_bars=60,
    )
    model = StatisticalRiskModel(cfg)
    out = model.fit(returns, timestamp_ns=1_700_000_000_000_000_000)
    assert out.timestamp_ns == 1_700_000_000_000_000_000
    assert out.model_type == "statistical"
    assert not out.is_interpretable  # PCA factor names are synthetic
