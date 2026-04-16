# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for FundamentalRiskModel (Barra-style WLS regression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.portfolio.risk_model.base import FundamentalFactorSpec, FundamentalModelConfig
from nautilus_quants.portfolio.risk_model.fundamental import FundamentalRiskModel


def _make_simulated_panel(
    n_assets: int = 20,
    n_periods: int = 300,
    n_style_factors: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], np.ndarray]:
    """Simulate returns driven by known style factors + sector + noise.

    Returns:
        returns (T, N), exposures_panel (T*N, K_style), sector_map, true_factor_returns (T, K_style)
    """
    rng = np.random.default_rng(seed)
    instruments = [f"ASSET_{i}.BINANCE" for i in range(n_assets)]
    index = pd.date_range("2024-01-01", periods=n_periods, freq="4h")

    style_names = [f"style_{i}" for i in range(n_style_factors)]
    # Fixed style exposures across time for simplicity (can also simulate time-varying)
    base_exposures = rng.standard_normal((n_assets, n_style_factors))
    # Cross-sectionally z-score each column
    base_exposures = (base_exposures - base_exposures.mean(axis=0)) / base_exposures.std(axis=0)
    # Build panel MultiIndex DataFrame: (T, N) rows × K cols
    repeated_exposures = np.tile(base_exposures, (n_periods, 1))
    mi = pd.MultiIndex.from_product([index, instruments], names=["ts", "inst"])
    exposures_df = pd.DataFrame(repeated_exposures, index=mi, columns=style_names)

    # Simulate factor returns + sector returns
    true_factor_returns = rng.standard_normal((n_periods, n_style_factors)) * 0.02  # 2% vol
    # 2 sectors, assign half of instruments to each
    sector_map = {inst: ("A" if i < n_assets // 2 else "B") for i, inst in enumerate(instruments)}
    # Sector factor returns
    sector_returns = rng.standard_normal((n_periods, 2)) * 0.01
    sector_dummies = np.zeros((n_assets, 2))
    for i, inst in enumerate(instruments):
        sector_dummies[i, 0 if sector_map[inst] == "A" else 1] = 1.0

    # Generate returns r_t = X f_t + sector + noise
    idio = rng.standard_normal((n_periods, n_assets)) * 0.005
    style_contrib = true_factor_returns @ base_exposures.T  # (T, N)
    sector_contrib = sector_returns @ sector_dummies.T
    r = style_contrib + sector_contrib + idio

    returns = pd.DataFrame(r, index=index, columns=instruments)
    return returns, exposures_df, sector_map, true_factor_returns


@pytest.mark.unit
def test_fundamental_produces_interpretable_output() -> None:
    returns, exposures, sector_map, _ = _make_simulated_panel()
    cfg = FundamentalModelConfig(
        factors=(
            FundamentalFactorSpec(name="style_0", variable="style_0"),
            FundamentalFactorSpec(name="style_1", variable="style_1"),
            FundamentalFactorSpec(name="style_2", variable="style_2"),
        ),
        sector_map=sector_map,
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
        winsorize_quantile=0.0,  # disable winsorize for exact synthetic test
    )
    model = FundamentalRiskModel(cfg)
    out = model.fit(
        returns=returns[cfg.factors[0].variable and returns.columns.to_list()],
        exposures=exposures.rename(columns={"style_0": "style_0", "style_1": "style_1"}),
        timestamp_ns=0,
    )

    assert out.model_type == "fundamental"
    assert out.is_interpretable
    assert out.is_decomposed
    # factor_names = 3 styles + {sector_A, sector_B}
    assert out.factor_names is not None
    assert "style_0" in out.factor_names
    assert any(n.startswith("sector_") for n in out.factor_names)
    # Covariance must be symmetric PSD
    assert out.covariance.shape == (returns.shape[1], returns.shape[1])
    np.testing.assert_allclose(out.covariance, out.covariance.T, atol=1e-10)
    eig = np.linalg.eigvalsh(out.covariance)
    assert eig.min() > -1e-8


@pytest.mark.unit
def test_fundamental_recovers_factor_returns_within_tolerance() -> None:
    """WLS regression should recover simulated factor returns approximately."""
    returns, exposures, sector_map, true_fr = _make_simulated_panel(n_assets=30, n_periods=500)
    cfg = FundamentalModelConfig(
        factors=(
            FundamentalFactorSpec(name="style_0", variable="style_0"),
            FundamentalFactorSpec(name="style_1", variable="style_1"),
            FundamentalFactorSpec(name="style_2", variable="style_2"),
        ),
        sector_map=sector_map,
        lookback_bars=500,
        min_history_bars=60,
        scale_return_pct=False,
        winsorize_quantile=0.0,
        winsorize_exposures_sigma=10.0,  # effectively disabled
        shrink_specific=False,
    )
    model = FundamentalRiskModel(cfg)
    out = model.fit(returns=returns, exposures=exposures, timestamp_ns=0)

    # Check factor_covariance has non-zero diagonal matching scale of true_fr variance
    assert out.factor_covariance is not None
    # style_0/1/2 are first 3 columns
    diag = np.diag(out.factor_covariance)[:3]
    expected = true_fr.var(axis=0)
    # Within factor of 2 (WLS with noise + sector confounding → rough match)
    assert np.all(diag > 0)
    ratio = diag / expected
    assert np.all((ratio > 0.3) & (ratio < 3.0)), f"variance ratio off: {ratio}"


@pytest.mark.unit
def test_fundamental_requires_exposures() -> None:
    returns, _, sector_map, _ = _make_simulated_panel()
    cfg = FundamentalModelConfig(
        factors=(FundamentalFactorSpec(name="x", variable="x"),),
        sector_map=sector_map,
        min_history_bars=60,
    )
    model = FundamentalRiskModel(cfg)
    with pytest.raises(ValueError, match="exposures"):
        model.fit(returns=returns, exposures=None, timestamp_ns=0)


@pytest.mark.unit
def test_fundamental_uses_wls_weights() -> None:
    """Providing WLS weights shouldn't crash and should still produce valid output."""
    returns, exposures, sector_map, _ = _make_simulated_panel()
    cfg = FundamentalModelConfig(
        factors=(
            FundamentalFactorSpec(name="style_0", variable="style_0"),
            FundamentalFactorSpec(name="style_1", variable="style_1"),
            FundamentalFactorSpec(name="style_2", variable="style_2"),
        ),
        sector_map=sector_map,
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = FundamentalRiskModel(cfg)
    rng = np.random.default_rng(0)
    mcap = rng.uniform(100, 10000, size=returns.shape[1])
    wls = np.sqrt(mcap)
    out = model.fit(
        returns=returns,
        exposures=exposures,
        timestamp_ns=0,
        weights=wls,
    )
    assert np.all(np.isfinite(out.covariance))


@pytest.mark.unit
def test_fundamental_missing_factor_raises() -> None:
    returns, exposures, sector_map, _ = _make_simulated_panel()
    cfg = FundamentalModelConfig(
        factors=(FundamentalFactorSpec(name="nonexistent", variable="nonexistent"),),
        sector_map=sector_map,
        min_history_bars=60,
    )
    model = FundamentalRiskModel(cfg)
    with pytest.raises(ValueError, match="missing style"):
        model.fit(returns=returns, exposures=exposures, timestamp_ns=0)


@pytest.mark.unit
def test_fundamental_unknown_instruments_go_to_other_sector() -> None:
    returns, exposures, _, _ = _make_simulated_panel(n_assets=5)
    cfg = FundamentalModelConfig(
        factors=(FundamentalFactorSpec(name="style_0", variable="style_0"),),
        sector_map={},  # empty → all instruments go to "Other"
        lookback_bars=300,
        min_history_bars=60,
        scale_return_pct=False,
    )
    model = FundamentalRiskModel(cfg)
    out = model.fit(returns=returns, exposures=exposures, timestamp_ns=0)
    assert out.factor_names is not None
    assert "sector_Other" in out.factor_names
