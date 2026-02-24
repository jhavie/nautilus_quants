# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Time-Series Operators.

Operators that work on historical data for a single instrument,
computing rolling statistics, delays, and other time-series transformations.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

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

    def make_incremental(self, window: int) -> "IncrementalMean":
        """Return O(1) incremental rolling mean state."""
        return IncrementalMean(window)

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.rolling(int(window)).mean()


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

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.rolling(int(window)).sum()


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

    def make_incremental(self, window: int) -> "IncrementalStd":
        """Return O(1) incremental rolling standard deviation state."""
        return IncrementalStd(window)

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.rolling(int(window)).std(ddof=1)


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

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.rolling(int(window)).min()


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

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.rolling(int(window)).max()


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

    def make_incremental(self, window: int) -> "IncrementalDelta":
        """Return O(1) incremental delta state."""
        return IncrementalDelta(window)

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.diff(int(window))


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

    def make_incremental(self, window: int) -> "IncrementalDelay":
        """Return O(1) incremental lag state."""
        return IncrementalDelay(window)

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        return data.shift(int(window))


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

    def make_incremental(self, window: int) -> "IncrementalCorr":
        """Return O(1) incremental Pearson correlation state."""
        return IncrementalCorr(window)

    def compute_vectorized(
        self, data: pd.Series, window: int, data2: pd.Series | None = None, **kwargs: Any,
    ) -> pd.Series:
        if data2 is None:
            return pd.Series(np.nan, index=data.index)
        return data.rolling(int(window)).corr(data2)


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

    def compute_vectorized(
        self, data: pd.Series, window: int, data2: pd.Series | None = None, **kwargs: Any,
    ) -> pd.Series:
        if data2 is None:
            return pd.Series(np.nan, index=data.index)
        return data.rolling(int(window)).cov(data2, ddof=1)


# ---------------------------------------------------------------------------
# Incremental (O(1) per bar) operator implementations
# ---------------------------------------------------------------------------


class IncrementalMean:
    """O(1) sliding-window mean using running sum.

    Equivalent to TsMean.compute(data[-window:], window) at every step,
    but maintains a running sum instead of recomputing over the full window.
    """

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0

    def push(self, value: float) -> float:
        """Push one new value and return updated rolling mean (nan during warmup)."""
        v = float(value)
        if len(self._buf) == self._window:
            self._sum -= self._buf[0]
        self._buf.append(v)
        self._sum += v
        if len(self._buf) < self._window:
            return float("nan")
        return self._sum / self._window

    def reset(self) -> None:
        """Clear all state."""
        self._buf.clear()
        self._sum = 0.0


class IncrementalStd:
    """O(1) sliding-window sample standard deviation (ddof=1).

    Equivalent to TsStd.compute(data[-window:], window) at every step,
    using running sum and sum-of-squares for Welford-like update.
    """

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    def push(self, value: float) -> float:
        """Push one new value and return updated rolling std (nan during warmup)."""
        v = float(value)
        if len(self._buf) == self._window:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(v)
        self._sum += v
        self._sum_sq += v * v

        n = len(self._buf)
        if n < self._window:
            return float("nan")

        # Sample variance: (sum_sq - n * mean^2) / (n - 1)
        mean = self._sum / n
        var = (self._sum_sq - n * mean * mean) / (n - 1)
        if var < 0.0:
            var = 0.0  # Numerical precision clamp
        return math.sqrt(var)

    def reset(self) -> None:
        """Clear all state."""
        self._buf.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


class IncrementalDelay:
    """O(1) sliding lag returning the value from `lag` steps ago.

    Equivalent to Delay.compute(data[:i+1], lag) at every step i,
    using a deque of maxlen=lag+1.
    """

    def __init__(self, lag: int) -> None:
        self._lag = lag
        self._buf: deque[float] = deque(maxlen=lag + 1)

    def push(self, value: float) -> float:
        """Push one new value and return the value from `lag` steps ago (nan during warmup)."""
        self._buf.append(float(value))
        if len(self._buf) <= self._lag:
            return float("nan")
        return self._buf[0]

    def reset(self) -> None:
        """Clear all state."""
        self._buf.clear()


class IncrementalDelta:
    """O(1) sliding delta: x[t] - x[t-lag].

    Equivalent to Delta.compute(data[:i+1], lag) at every step i,
    using an IncrementalDelay to maintain the lagged value.
    """

    def __init__(self, lag: int) -> None:
        self._delay = IncrementalDelay(lag)

    def push(self, value: float) -> float:
        """Push one value and return x[t] - x[t-lag] (nan during warmup)."""
        old = self._delay.push(value)
        return float(value) - old  # NaN propagates naturally when old is nan

    def reset(self) -> None:
        """Clear all state."""
        self._delay.reset()


class IncrementalCorr:
    """O(1) sliding-window Pearson correlation using online running statistics.

    Equivalent to Correlation.compute(x[:i+1], window, data2=y[:i+1]) at every
    step i, using the one-pass formula:

        r = (n * sum_xy - sum_x * sum_y)
            / sqrt((n * sum_x2 - sum_x^2) * (n * sum_y2 - sum_y^2))

    This is mathematically identical to numpy.corrcoef for same input window.
    Numerical tolerance vs batch: abs error < 1e-8 for typical price/volume data.
    """

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf_x: deque[float] = deque(maxlen=window)
        self._buf_y: deque[float] = deque(maxlen=window)
        self._sum_x: float = 0.0
        self._sum_y: float = 0.0
        self._sum_xy: float = 0.0
        self._sum_x2: float = 0.0
        self._sum_y2: float = 0.0

    def push(self, x: float, y: float) -> float:
        """Push one new (x, y) pair and return updated Pearson r (nan during warmup)."""
        xv, yv = float(x), float(y)
        if len(self._buf_x) == self._window:
            ox, oy = self._buf_x[0], self._buf_y[0]
            self._sum_x -= ox
            self._sum_y -= oy
            self._sum_xy -= ox * oy
            self._sum_x2 -= ox * ox
            self._sum_y2 -= oy * oy

        self._buf_x.append(xv)
        self._buf_y.append(yv)
        self._sum_x += xv
        self._sum_y += yv
        self._sum_xy += xv * yv
        self._sum_x2 += xv * xv
        self._sum_y2 += yv * yv

        n = len(self._buf_x)
        if n < self._window:
            return float("nan")

        # One-pass Pearson r
        num = n * self._sum_xy - self._sum_x * self._sum_y
        den_x = n * self._sum_x2 - self._sum_x * self._sum_x
        den_y = n * self._sum_y2 - self._sum_y * self._sum_y

        if den_x <= 0.0 or den_y <= 0.0:
            return float("nan")  # Constant series

        denom = math.sqrt(den_x * den_y)
        return num / denom

    def reset(self) -> None:
        """Clear all state."""
        self._buf_x.clear()
        self._buf_y.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0


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

# Instance registry for vectorized evaluator (lookup by operator name)
TS_OPERATOR_INSTANCES: dict[str, TimeSeriesOperator] = {
    "ts_mean": TsMean(),
    "ts_sum": TsSum(),
    "ts_std": TsStd(),
    "ts_min": TsMin(),
    "ts_max": TsMax(),
    "ts_rank": TsRank(),
    "ts_argmax": TsArgmax(),
    "ts_argmin": TsArgmin(),
    "delta": Delta(),
    "delay": Delay(),
    "correlation": Correlation(),
    "covariance": Covariance(),
}
