# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Regime-conditional IC analysis for factor composites.

Splits factor IC time series by market regime (BTC-driven) and
computes per-regime ICIR to quantify regime sensitivity of each
factor. Used to determine optimal per-regime composite weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from nautilus_quants.factors.config import FactorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeICResult:
    """Per-factor IC statistics split by regime."""

    factor_name: str
    ic_bull: np.ndarray
    ic_bear: np.ndarray

    @property
    def icir_bull(self) -> float:
        std = self.ic_bull.std()
        return float(self.ic_bull.mean() / std) if std > 0 else 0.0

    @property
    def icir_bear(self) -> float:
        std = self.ic_bear.std()
        return float(self.ic_bear.mean() / std) if std > 0 else 0.0

    @property
    def icir_diff(self) -> float:
        return self.icir_bull - self.icir_bear

    @property
    def win_rate_bull(self) -> float:
        return float((self.ic_bull > 0).mean()) if len(self.ic_bull) else 0.0

    @property
    def win_rate_bear(self) -> float:
        return float((self.ic_bear > 0).mean()) if len(self.ic_bear) else 0.0


@dataclass(frozen=True)
class RegimeAnalysisReport:
    """Full regime IC analysis results."""

    factor_results: list[RegimeICResult]
    composite_result: RegimeICResult
    regime_series: pd.Series
    quarterly_icir: pd.DataFrame


def detect_regime_ema(
    btc_close: pd.Series,
    ema_span: int = 20,
) -> pd.Series:
    """Detect market regime from BTC close prices using EMA of returns.

    Args:
        btc_close: BTC close price series (DatetimeIndex).
        ema_span: EMA span for return smoothing.

    Returns:
        Series of "bull"/"bear" labels indexed by timestamp.
    """
    returns = btc_close.pct_change().dropna()
    ema = returns.ewm(span=ema_span).mean()
    return pd.Series(
        np.where(ema > 0, "bull", "bear"),
        index=ema.index,
        name="regime",
    )


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional normalize: (x - row_mean) / row_std."""
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def _compute_ic_series(
    factor_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    min_obs: int = 20,
) -> list[tuple[pd.Timestamp, float]]:
    """Compute Spearman IC at each timestamp.

    Args:
        factor_df: Factor values (timestamps × instruments).
        fwd_returns: Forward returns (timestamps × instruments).
        min_obs: Minimum cross-sectional observations per timestamp.

    Returns:
        List of (timestamp, ic_value) tuples.
    """
    results = []
    common_dates = factor_df.index.intersection(fwd_returns.index)
    for dt in common_dates:
        f = factor_df.loc[dt].dropna()
        r = fwd_returns.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_obs:
            continue
        corr, _ = stats.spearmanr(f[common].values, r[common].values)
        if not np.isnan(corr):
            results.append((dt, corr))
    return results


def compute_regime_ic(
    factor_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    regime: pd.Series,
    min_obs: int = 20,
) -> list[RegimeICResult]:
    """Compute per-factor IC split by regime.

    Args:
        factor_dfs: {factor_name: DataFrame(timestamps × instruments)}.
        fwd_returns: Forward returns DataFrame.
        regime: Series of regime labels ("bull"/"bear") indexed by timestamp.
        min_obs: Minimum observations per IC computation.

    Returns:
        List of RegimeICResult for each factor.
    """
    results = []
    for name, df in factor_dfs.items():
        ic_series = _compute_ic_series(df, fwd_returns, min_obs)
        ic_bull, ic_bear = [], []
        for dt, ic in ic_series:
            if dt in regime.index:
                if regime.loc[dt] == "bull":
                    ic_bull.append(ic)
                else:
                    ic_bear.append(ic)
        results.append(
            RegimeICResult(
                factor_name=name,
                ic_bull=np.array(ic_bull),
                ic_bear=np.array(ic_bear),
            )
        )
    return results


def compute_regime_aware_weights(
    regime_results: list[RegimeICResult],
    min_weight: float = 0.02,
) -> dict[str, dict[str, float]]:
    """Compute ICIR-proportional weights per regime.

    Args:
        regime_results: Per-factor regime IC results.
        min_weight: Floor for each weight (prevents zeroing out factors).

    Returns:
        {"bull": {factor: weight}, "bear": {factor: weight}}.
    """
    weight_map: dict[str, dict[str, float]] = {}
    for regime_label, icir_attr in [("bull", "icir_bull"), ("bear", "icir_bear")]:
        raw = {
            r.factor_name: max(getattr(r, icir_attr), min_weight)
            for r in regime_results
        }
        total = sum(raw.values())
        weight_map[regime_label] = {k: v / total for k, v in raw.items()}
    return weight_map


def run_regime_analysis(
    factor_dfs: dict[str, pd.DataFrame],
    pricing: pd.DataFrame,
    btc_instrument: str = "BTCUSDT.BINANCE",
    ema_span: int = 20,
    weights: dict[str, float] | None = None,
) -> RegimeAnalysisReport:
    """Run full regime-conditional IC analysis.

    Args:
        factor_dfs: {factor_name: DataFrame(timestamps × instruments)}.
        pricing: Close prices DataFrame (timestamps × instruments).
        btc_instrument: BTC instrument ID for regime detection.
        ema_span: EMA span for regime proxy.
        weights: Composite weights (default: equal weight).

    Returns:
        RegimeAnalysisReport with all results.
    """
    factor_names = list(factor_dfs.keys())
    if weights is None:
        weights = {n: 1.0 / len(factor_names) for n in factor_names}

    # Regime detection
    btc_close = pricing[btc_instrument].dropna()
    regime = detect_regime_ema(btc_close, ema_span)
    logger.info(
        "Regime detected: bull=%.1f%%, bear=%.1f%%, switches=%d",
        (regime == "bull").mean() * 100,
        (regime == "bear").mean() * 100,
        (regime != regime.shift()).sum(),
    )

    # Forward returns (1-bar)
    fwd_returns = pricing.pct_change().shift(-1)

    # Per-factor IC by regime
    factor_results = compute_regime_ic(factor_dfs, fwd_returns, regime)

    # Composite IC by regime
    norm_dfs = {n: _cs_normalize(df) for n, df in factor_dfs.items()}
    composite_df = sum(weights[n] * norm_dfs[n] for n in factor_names)
    composite_results = compute_regime_ic(
        {"composite": composite_df}, fwd_returns, regime
    )
    composite_result = composite_results[0]

    # Quarterly ICIR comparison
    ic_series = _compute_ic_series(composite_df, fwd_returns)
    ic_df = pd.DataFrame(ic_series, columns=["date", "ic"]).set_index("date")
    ic_df.index = pd.to_datetime(ic_df.index)
    ic_df["quarter"] = ic_df.index.to_period("Q")

    quarterly_rows = []
    for q, group in ic_df.groupby("quarter"):
        if len(group) < 10:
            continue
        ic_mean = group["ic"].mean()
        ic_std = group["ic"].std()
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        quarterly_rows.append(
            {"quarter": str(q), "n": len(group), "ic_mean": ic_mean, "icir": icir}
        )
    quarterly_icir = pd.DataFrame(quarterly_rows)

    return RegimeAnalysisReport(
        factor_results=factor_results,
        composite_result=composite_result,
        regime_series=regime,
        quarterly_icir=quarterly_icir,
    )
