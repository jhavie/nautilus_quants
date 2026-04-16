# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Returns preprocessing utilities — NaN handling, winsorize, centering.

Ported from Qlib RiskModel.base._preprocess() + prepare_riskdata.py
(winsorize at 2.5%/97.5% quantiles before covariance estimation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAN_IGNORE = "ignore"
NAN_FILL = "fill"
NAN_MASK = "mask"


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price panel to simple returns: r[t] = p[t]/p[t-1] - 1.

    Returns has T-1 rows compared to input (first row dropped).
    """
    if prices.shape[0] < 2:
        raise ValueError(f"prices must have >= 2 rows, got {prices.shape[0]}")
    return prices.pct_change().iloc[1:]


def winsorize_returns(
    returns: pd.DataFrame,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> pd.DataFrame:
    """Clip returns column-wise at (lower, upper) quantiles.

    Matches Qlib prepare_riskdata.py:
        ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)

    Parameters
    ----------
    returns : pd.DataFrame
        (T, N) returns matrix.
    lower_quantile : float
        Lower clipping quantile, default 0.025.
    upper_quantile : float
        Upper clipping quantile, default 0.975.

    Returns
    -------
    pd.DataFrame
        Winsorized returns (same shape).
    """
    if lower_quantile <= 0.0 and upper_quantile >= 1.0:
        return returns
    lo = returns.quantile(lower_quantile)
    hi = returns.quantile(upper_quantile)
    return returns.clip(lower=lo, upper=hi, axis=1)


def handle_nan(
    x: np.ndarray,
    nan_option: str = NAN_FILL,
) -> np.ndarray | np.ma.MaskedArray:
    """Handle NaN values per chosen strategy.

    Parameters
    ----------
    x : np.ndarray
        Input matrix (T, N).
    nan_option : str
        "ignore" — pass through (NaN propagates via numpy).
        "fill" — replace NaN with 0 via np.nan_to_num.
        "mask" — return np.ma.MaskedArray for pairwise counting in covariance.

    Returns
    -------
    np.ndarray | np.ma.MaskedArray
        Cleaned matrix.
    """
    if nan_option == NAN_FILL:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if nan_option == NAN_MASK:
        return np.ma.masked_invalid(x)
    if nan_option == NAN_IGNORE:
        return x
    raise ValueError(f"unknown nan_option: {nan_option!r}")


def center_columns(
    x: np.ndarray | np.ma.MaskedArray,
    assume_centered: bool = False,
) -> np.ndarray | np.ma.MaskedArray:
    """Subtract column-wise mean (nanmean-aware).

    Parameters
    ----------
    x : np.ndarray | np.ma.MaskedArray
        Input matrix (T, N).
    assume_centered : bool
        If True, return x unchanged (data already zero-mean).

    Returns
    -------
    np.ndarray | np.ma.MaskedArray
        Centered matrix.
    """
    if assume_centered:
        return x
    if isinstance(x, np.ma.MaskedArray):
        return x - np.ma.mean(x, axis=0)
    return x - np.nanmean(x, axis=0)


def scale_to_percent(x: np.ndarray | np.ma.MaskedArray) -> np.ndarray | np.ma.MaskedArray:
    """Multiply returns by 100 (Qlib convention for PCA/Shrinkage numerical stability)."""
    return x * 100.0


def preprocess_returns(
    returns: pd.DataFrame,
    *,
    winsorize_quantile: float = 0.025,
    nan_option: str = NAN_FILL,
    scale_pct: bool = True,
    assume_centered: bool = False,
) -> tuple[np.ndarray | np.ma.MaskedArray, tuple[str, ...]]:
    """Full preprocessing pipeline: winsorize → NaN handling → scale → center.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw returns (T, N). Index = timestamps, columns = instruments.
    winsorize_quantile : float
        Lower quantile for clipping (upper = 1 - this).
    nan_option : str
        "ignore" | "fill" | "mask".
    scale_pct : bool
        Multiply returns by 100 for numerical stability.
    assume_centered : bool
        Skip mean subtraction if True.

    Returns
    -------
    tuple[np.ndarray | np.ma.MaskedArray, tuple[str, ...]]
        Preprocessed matrix and the instrument column order.
    """
    winsorized = winsorize_returns(
        returns,
        lower_quantile=winsorize_quantile,
        upper_quantile=1.0 - winsorize_quantile,
    )
    instruments = tuple(str(c) for c in winsorized.columns)
    x = winsorized.to_numpy(dtype=np.float64)
    x = handle_nan(x, nan_option)
    if scale_pct:
        x = scale_to_percent(x)
    x = center_columns(x, assume_centered=assume_centered)
    return x, instruments
