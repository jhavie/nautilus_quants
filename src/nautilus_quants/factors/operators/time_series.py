# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Time-Series Operators.

Operators that work on historical data for a single instrument,
computing rolling statistics, delays, and other time-series transformations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nautilus_quants.factors.operators.base import (
    TimeSeriesOperator,
    register_operator,
)


@register_operator
class TsMean(TimeSeriesOperator):
    """Rolling mean (moving average)."""
    
    name = "ts_mean"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """
        Compute rolling mean.
        
        Args:
            data: Historical data array
            window: Lookback window size
            
        Returns:
            Mean of the last `window` values
        """
        if len(data) < window:
            return float('nan')
        return float(np.mean(data[-window:]))


@register_operator
class TsSum(TimeSeriesOperator):
    """Rolling sum."""
    
    name = "ts_sum"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute rolling sum."""
        if len(data) < window:
            return float('nan')
        return float(np.sum(data[-window:]))


@register_operator
class TsStd(TimeSeriesOperator):
    """Rolling standard deviation."""
    
    name = "ts_std"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute rolling standard deviation (sample std, ddof=1)."""
        if len(data) < window:
            return float('nan')
        return float(np.std(data[-window:], ddof=1))


@register_operator
class TsMin(TimeSeriesOperator):
    """Rolling minimum."""
    
    name = "ts_min"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute rolling minimum."""
        if len(data) < window:
            return float('nan')
        return float(np.min(data[-window:]))


@register_operator
class TsMax(TimeSeriesOperator):
    """Rolling maximum."""
    
    name = "ts_max"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute rolling maximum."""
        if len(data) < window:
            return float('nan')
        return float(np.max(data[-window:]))


@register_operator
class TsRank(TimeSeriesOperator):
    """Rolling rank (percentile rank of current value in window)."""
    
    name = "ts_rank"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """
        Compute time-series rank.
        
        Returns the percentile rank of the current value within
        the lookback window (0.0 to 1.0).
        """
        if len(data) < window:
            return float('nan')
        
        window_data = data[-window:]
        current = window_data[-1]
        
        # Count values less than current
        rank = np.sum(window_data < current)
        # Normalize to [0, 1]
        return float(rank / (window - 1)) if window > 1 else 0.5


@register_operator
class TsArgmax(TimeSeriesOperator):
    """Index of maximum value in window (WorldQuant semantics: 1 = oldest, window = most recent)."""
    
    name = "ts_argmax"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """
        Compute position of maximum value (WorldQuant semantics).
        
        Returns position in window where maximum occurred:
        - 1 = oldest day in window
        - window = most recent day (today)
        
        This follows WorldQuant Alpha101 convention:
        ts_argmax(close, 31) == 31 means today is the maximum.
        """
        if len(data) < window:
            return float('nan')
        
        window_data = data[-window:]
        idx = np.argmax(window_data)
        return float(idx + 1)  # 1-indexed, 1 = oldest, window = newest


@register_operator
class TsArgmin(TimeSeriesOperator):
    """Index of minimum value in window (WorldQuant semantics: 1 = oldest, window = most recent)."""
    
    name = "ts_argmin"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """
        Compute position of minimum value (WorldQuant semantics).
        
        Returns position in window where minimum occurred:
        - 1 = oldest day in window
        - window = most recent day (today)
        
        This follows WorldQuant Alpha101 convention:
        ts_argmin(close, 31) == 31 means today is the minimum.
        """
        if len(data) < window:
            return float('nan')
        
        window_data = data[-window:]
        idx = np.argmin(window_data)
        return float(idx + 1)  # 1-indexed, 1 = oldest, window = newest


@register_operator
class Delta(TimeSeriesOperator):
    """Difference from n periods ago: x[t] - x[t-n]."""
    
    name = "delta"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute difference from n periods ago."""
        # Handle scalar input (from nested function calls)
        if isinstance(data, (int, float)):
            return float('nan')  # Can't compute delta on scalar
        if len(data) <= window:
            return float('nan')
        return float(data[-1] - data[-window - 1])


@register_operator
class Delay(TimeSeriesOperator):
    """Lagged value: x[t-n]."""
    
    name = "delay"
    min_args = 2
    max_args = 2
    
    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Return value from n periods ago."""
        # Handle scalar input (from nested function calls)
        if isinstance(data, (int, float)):
            return float(data)  # For scalar, just return the value
        if len(data) <= window:
            return float('nan')
        return float(data[-window - 1])


@register_operator
class Correlation(TimeSeriesOperator):
    """Rolling correlation between two series."""
    
    name = "correlation"
    min_args = 3
    max_args = 3
    
    def compute(
        self, 
        data: np.ndarray, 
        window: int,
        data2: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray:
        """
        Compute rolling correlation.
        
        Note: data2 should be passed via kwargs or as third positional arg.
        """
        if data2 is None:
            data2 = kwargs.get('data2')
        if data2 is None:
            return float('nan')
        
        if len(data) < window or len(data2) < window:
            return float('nan')
        
        x = data[-window:]
        y = data2[-window:]
        
        # Handle constant arrays
        if np.std(x) == 0 or np.std(y) == 0:
            return float('nan')
        
        return float(np.corrcoef(x, y)[0, 1])


@register_operator
class Covariance(TimeSeriesOperator):
    """Rolling covariance between two series."""
    
    name = "covariance"
    min_args = 3
    max_args = 3
    
    def compute(
        self, 
        data: np.ndarray, 
        window: int,
        data2: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float | np.ndarray:
        """Compute rolling covariance."""
        if data2 is None:
            data2 = kwargs.get('data2')
        if data2 is None:
            return float('nan')
        
        if len(data) < window or len(data2) < window:
            return float('nan')
        
        x = data[-window:]
        y = data2[-window:]
        
        return float(np.cov(x, y, ddof=1)[0, 1])


# Convenience function wrappers for use in evaluator
def ts_mean(data: np.ndarray, window: int) -> float:
    """Wrapper for TsMean operator."""
    return TsMean().compute(data, int(window))  # type: ignore


def ts_sum(data: np.ndarray, window: int) -> float:
    """Wrapper for TsSum operator."""
    return TsSum().compute(data, int(window))  # type: ignore


def ts_std(data: np.ndarray, window: int) -> float:
    """Wrapper for TsStd operator."""
    return TsStd().compute(data, int(window))  # type: ignore


def ts_min(data: np.ndarray, window: int) -> float:
    """Wrapper for TsMin operator."""
    return TsMin().compute(data, int(window))  # type: ignore


def ts_max(data: np.ndarray, window: int) -> float:
    """Wrapper for TsMax operator."""
    return TsMax().compute(data, int(window))  # type: ignore


def ts_rank(data: np.ndarray, window: int) -> float:
    """Wrapper for TsRank operator."""
    return TsRank().compute(data, int(window))  # type: ignore


def ts_argmax(data: np.ndarray, window: int) -> float:
    """Wrapper for TsArgmax operator."""
    return TsArgmax().compute(data, int(window))  # type: ignore


def ts_argmin(data: np.ndarray, window: int) -> float:
    """Wrapper for TsArgmin operator."""
    return TsArgmin().compute(data, int(window))  # type: ignore


def delta(data: np.ndarray, window: int) -> float:
    """Wrapper for Delta operator."""
    return Delta().compute(data, int(window))  # type: ignore


def delay(data: np.ndarray, window: int) -> float:
    """Wrapper for Delay operator."""
    return Delay().compute(data, int(window))  # type: ignore


def correlation(data1: np.ndarray, data2: np.ndarray, window: int) -> float:
    """Wrapper for Correlation operator."""
    return Correlation().compute(data1, int(window), data2=data2)  # type: ignore


def covariance(data1: np.ndarray, data2: np.ndarray, window: int) -> float:
    """Wrapper for Covariance operator."""
    return Covariance().compute(data1, int(window), data2=data2)  # type: ignore


# Export all function wrappers
TIME_SERIES_OPERATORS = {
    "ts_mean": ts_mean,
    "ts_sum": ts_sum,
    "ts_std": ts_std,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_rank": ts_rank,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "delta": delta,
    "delay": delay,
    "correlation": correlation,
    "covariance": covariance,
}
