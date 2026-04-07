# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Regime detection feature engineering.

Constructs feature matrices for Statistical Jump Model regime detection
from BTC/ETH price and volume data.

Features based on:
- Yu, Mulvey, Nie (JPM 2026): EMA returns, downside deviation, Sortino ratio
- Cortese et al. (2023): RSI, volume-return correlation (validated for crypto)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema_returns(close: pd.Series, span: int) -> pd.Series:
    """EMA of log returns."""
    ret = np.log(close / close.shift(1))
    return ret.ewm(span=span).mean()


def _downside_deviation(close: pd.Series, window: int) -> pd.Series:
    """Rolling downside deviation (log scale)."""
    ret = np.log(close / close.shift(1))
    neg_ret = ret.clip(upper=0)
    return (neg_ret.pow(2).rolling(window).mean()).pow(0.5)


def _sortino_ratio(close: pd.Series, window: int) -> pd.Series:
    """Rolling Sortino ratio."""
    ret = np.log(close / close.shift(1))
    mean_ret = ret.rolling(window).mean()
    dd = _downside_deviation(close, window)
    return mean_ret / dd.replace(0, np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _volume_return_corr(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling correlation between returns and volume."""
    ret = close.pct_change()
    return ret.rolling(window).corr(volume)


def build_regime_features(
    btc_close: pd.Series,
    btc_volume: pd.Series | None = None,
    eth_close: pd.Series | None = None,
    feature_set: str = "full",
) -> pd.DataFrame:
    """Build feature matrix for regime detection.

    Args:
        btc_close: BTC close prices (DatetimeIndex).
        btc_volume: BTC volume (optional, enables volume features).
        eth_close: ETH close prices (optional, enables relative features).
        feature_set: Feature set to use:
            - "full": All features (paper + Cortese)
            - "returns_only": EMA returns only (minimal)
            - "paper": Paper features (EMA returns + downside dev + Sortino)

    Returns:
        Feature DataFrame (timestamps × features), NaN rows dropped.
    """
    features = {}

    # EMA returns at multiple horizons (paper + Cortese)
    for span in [7, 14, 21]:
        features[f"ema{span}_ret"] = _ema_returns(btc_close, span)

    if feature_set in ("full", "paper"):
        # Downside deviation (paper)
        for window in [5 * 6, 21 * 6]:  # 5-day and 21-day in 4h bars
            label = window // 6
            features[f"dd_{label}d"] = _downside_deviation(btc_close, window)

        # Sortino ratio (paper)
        for window in [5 * 6, 10 * 6, 21 * 6]:
            label = window // 6
            features[f"sortino_{label}d"] = _sortino_ratio(btc_close, window)

    if feature_set == "full":
        # RSI (Cortese)
        features["rsi_14"] = _rsi(btc_close, window=14)

        # Volume-return correlation (Cortese)
        if btc_volume is not None:
            features["vol_ret_corr"] = _volume_return_corr(
                btc_close, btc_volume, window=20
            )

        # BTC-ETH relative strength (crypto-specific)
        if eth_close is not None:
            btc_ret = btc_close.pct_change()
            eth_ret = eth_close.pct_change()
            features["btc_eth_spread"] = (btc_ret - eth_ret).ewm(span=14).mean()

    df = pd.DataFrame(features)
    return df.dropna()
