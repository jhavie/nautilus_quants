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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        return data.rolling(int(window)).max()


@register_operator
class TsRank(TimeSeriesOperator):
    """Rolling rank — popbo/academic semantics.

    compute(): returns average rank normalized to [1/d, 1] (incremental engine).
    compute_panel(): returns scipy.rankdata(method='min')[-1], raw rank in [1, d].

    This aligns with popbo/alphas101.py:
        def rolling_rank(na):
            return rankdata(na, method='min')[-1]
    """

    name = "ts_rank"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Return average rank of last value in window, normalized to [1/d, 1]."""
        if len(data) < window:
            return float('nan')

        window_data = data[-window:]
        current = window_data[-1]

        # Average rank: (count_less + (count_equal + 1) / 2) / window
        less = np.sum(window_data < current)
        equal = np.sum(window_data == current)
        avg_rank = less + (equal + 1) / 2
        return float(avg_rank / window)

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """popbo semantics: rankdata(method='min')[-1], returns [1, d].

        Optimized: numpy batch comparison replaces per-window scipy.rankdata.
        rank_min(last) = count(window < last) + 1
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)  # [N]
            last = win[-1:]  # [1, N]
            rank = np.nansum(win < last, axis=0).astype(float) + 1.0
            rank[has_nan] = np.nan
            result[t] = rank

        return pd.DataFrame(result, index=data.index, columns=data.columns)


@register_operator
class TsArgmax(TimeSeriesOperator):
    """Index of maximum value in window — popbo/academic semantics.

    Returns np.argmax + 1 (1-indexed from oldest):
    - 1 = oldest day in window
    - window = most recent day (today)

    Aligns with popbo: df.rolling(window).apply(np.argmax) + 1
    """

    name = "ts_argmax"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute position of maximum value (1-indexed from oldest)."""
        if len(data) < window:
            return float('nan')

        window_data = data[-window:]
        idx = np.argmax(window_data)
        return float(idx + 1)  # 1-indexed, 1 = oldest, window = newest

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """popbo semantics: np.argmax + 1 (1-indexed from oldest).

        Optimized: numpy batch argmax replaces per-window lambda.
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            r = np.argmax(win, axis=0).astype(float) + 1.0
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


@register_operator
class TsArgmin(TimeSeriesOperator):
    """Index of minimum value in window — popbo/academic semantics.

    Returns np.argmin + 1 (1-indexed from oldest):
    - 1 = oldest day in window
    - window = most recent day (today)

    Aligns with popbo: df.rolling(window).apply(np.argmin) + 1
    """

    name = "ts_argmin"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute position of minimum value (1-indexed from oldest)."""
        if len(data) < window:
            return float('nan')

        window_data = data[-window:]
        idx = np.argmin(window_data)
        return float(idx + 1)  # 1-indexed, 1 = oldest, window = newest

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """popbo semantics: np.argmin + 1 (1-indexed from oldest).

        Optimized: numpy batch argmin replaces per-window lambda.
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            r = np.argmin(win, axis=0).astype(float) + 1.0
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
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
        # Popbo-aligned: fillna(0) treats undefined correlation (constant series) as uncorrelated
        return data.rolling(int(window)).corr(data2).fillna(0).replace([np.inf, -np.inf], 0)

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """DataFrame-level rolling Pearson correlation.

        Uses pandas DataFrame.rolling().corr(other_df) which computes all
        columns simultaneously (C-optimized), avoiding per-column Python loop.

        Popbo-aligned: fillna(0) treats undefined correlation (constant series)
        as uncorrelated (0).
        """
        data2 = kwargs.get("data2")
        if data2 is None:
            return pd.DataFrame(np.nan, index=data.index, columns=data.columns)
        w = int(window)
        result = data.rolling(w).corr(data2)
        return result.fillna(0).replace([np.inf, -np.inf], 0)


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

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """DataFrame-level rolling covariance.

        Uses pandas DataFrame.rolling().cov(other_df) which computes all
        columns simultaneously (C-optimized), avoiding per-column Python loop.
        """
        data2 = kwargs.get("data2")
        if data2 is None:
            return pd.DataFrame(np.nan, index=data.index, columns=data.columns)
        w = int(window)
        return data.rolling(w).cov(data2, ddof=1)


@register_operator
class DecayLinear(TimeSeriesOperator):
    """Linear weighted moving average (LWMA).

    Weights are [1, 2, ..., d] normalized to sum to 1.
    Most recent value has highest weight.
    """

    name = "decay_linear"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute linearly weighted moving average (popbo-aligned form)."""
        if isinstance(data, (int, float)):
            return float('nan')
        if len(data) < window:
            return float('nan')
        w = int(window)
        weights = np.array(range(1, w + 1), dtype=float)
        sum_weights = float(np.sum(weights))
        return float(np.sum(weights * data[-w:]) / sum_weights)

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        """Vectorized LWMA computation (popbo-aligned form)."""
        w = int(window)
        weights = np.array(range(1, w + 1), dtype=float)
        sum_weights = float(np.sum(weights))
        return data.rolling(w).apply(lambda x: np.sum(weights * x) / sum_weights, raw=True)

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """popbo-aligned: exact same computation form as popbo's decay_linear.

        popbo uses: np.sum(weights * x) / sum_weights  (un-normalized weights)
        Optimized: numpy batch weighted sum replaces per-window lambda.
        """
        w = int(window)
        weights = np.arange(1, w + 1, dtype=float)  # [w]
        sum_weights = weights.sum()

        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            # np.sum(weights * x) / sum_weights — popbo evaluation order
            r = np.sum(weights[:, None] * win, axis=0) / sum_weights
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


@register_operator
class TsProduct(TimeSeriesOperator):
    """Rolling product over window."""

    name = "ts_product"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        """Compute rolling product."""
        if isinstance(data, (int, float)):
            return float('nan')
        if len(data) < window:
            return float('nan')
        return float(np.prod(data[-int(window):]))

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        """Vectorized rolling product."""
        return data.rolling(int(window)).apply(np.prod, raw=True)

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """Optimized: numpy batch product replaces per-window lambda."""
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            r = np.prod(win, axis=0)
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


# ---------------------------------------------------------------------------
# WorldQuant BRAIN operators (wq_ prefix) — BRAIN platform semantics
# ---------------------------------------------------------------------------


@register_operator
class WqTsRank(TimeSeriesOperator):
    """WorldQuant BRAIN ts_rank: (scipy_rank - 1) / (d - 1), value range [0, 1].

    Example: wq_ts_rank([200, 0, 100], d=3) → 0.5 (100 is the median value)
    BRAIN docs: ts_rank returns 0 for smallest, 1 for largest.
    """

    name = "wq_ts_rank"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        if len(data) < window:
            return float("nan")
        from scipy.stats import rankdata

        arr = data[-window:]
        raw = float(rankdata(arr)[-1])
        return (raw - 1) / (window - 1) if window > 1 else 0.5

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """BRAIN semantics: (average_rank - 1) / (d - 1), value range [0, 1].

        Optimized: numpy batch comparison replaces per-window scipy.rankdata.
        average_rank(last) = count_less + (count_equal + 1) / 2
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            last = win[-1:]  # [1, N]
            count_less = np.nansum(win < last, axis=0).astype(float)
            count_equal = np.nansum(win == last, axis=0).astype(float)
            avg_rank = count_less + (count_equal + 1.0) / 2.0
            r = (avg_rank - 1.0) / (w - 1.0) if w > 1 else np.full(N, 0.5)
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


@register_operator
class WqTsArgmax(TimeSeriesOperator):
    """WorldQuant BRAIN ts_argmax: 0-indexed from today. Today=0, yesterday=1.

    Example: wq_ts_argmax([4,9,5,8,2,6], d=6) → 4 (max=9 was 4 days ago)
    Note: BRAIN input order is [today, yesterday, ...] but pandas data is
    [oldest, ..., today], so we compute: d - 1 - np.argmax(arr).
    """

    name = "wq_ts_argmax"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        if len(data) < window:
            return float("nan")
        arr = data[-window:]
        return float(len(arr) - 1 - np.argmax(arr))

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """BRAIN semantics: d - 1 - argmax (0-indexed from today).

        Optimized: numpy batch argmax replaces per-window lambda.
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            r = (w - 1.0 - np.argmax(win, axis=0)).astype(float)
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


@register_operator
class WqTsArgmin(TimeSeriesOperator):
    """WorldQuant BRAIN ts_argmin: 0-indexed from today. Today=0, yesterday=1.

    Example: wq_ts_argmin([4,9,5,8,2,6], d=6) → 1 (min=2 was 1 day ago)
    """

    name = "wq_ts_argmin"
    min_args = 2
    max_args = 2

    def compute(self, data: np.ndarray, window: int, **kwargs: Any) -> float | np.ndarray:
        if len(data) < window:
            return float("nan")
        arr = data[-window:]
        return float(len(arr) - 1 - np.argmin(arr))

    def compute_panel(self, data: pd.DataFrame, window: int, **kwargs: Any) -> pd.DataFrame:
        """BRAIN semantics: d - 1 - argmin (0-indexed from today).

        Optimized: numpy batch argmin replaces per-window lambda.
        """
        w = int(window)
        values = data.values  # [T, N]
        T, N = values.shape
        result = np.full((T, N), np.nan)

        for t in range(w - 1, T):
            win = values[t - w + 1:t + 1]  # [w, N]
            has_nan = np.any(np.isnan(win), axis=0)
            r = (w - 1.0 - np.argmin(win, axis=0)).astype(float)
            r[has_nan] = np.nan
            result[t] = r

        return pd.DataFrame(result, index=data.index, columns=data.columns)


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


def decay_linear(data: np.ndarray, window: int) -> float:
    """Wrapper for DecayLinear operator."""
    return DecayLinear().compute(data, int(window))  # type: ignore


def ts_product(data: np.ndarray, window: int) -> float:
    """Wrapper for TsProduct operator."""
    return TsProduct().compute(data, int(window))  # type: ignore


def wq_ts_rank(data: np.ndarray, window: int) -> float:
    """Wrapper for WqTsRank operator (BRAIN semantics)."""
    return WqTsRank().compute(data, int(window))  # type: ignore


def wq_ts_argmax(data: np.ndarray, window: int) -> float:
    """Wrapper for WqTsArgmax operator (BRAIN semantics)."""
    return WqTsArgmax().compute(data, int(window))  # type: ignore


def wq_ts_argmin(data: np.ndarray, window: int) -> float:
    """Wrapper for WqTsArgmin operator (BRAIN semantics)."""
    return WqTsArgmin().compute(data, int(window))  # type: ignore


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
    "decay_linear": decay_linear,
    "ts_product": ts_product,
    # Aliases (direct references, no wrapper functions)
    "stddev": ts_std,
    "sma": ts_mean,
    "ts_delta": delta,
    "ts_delay": delay,
    "ts_corr": correlation,
    "ts_covariance": covariance,
    "ts_std_dev": ts_std,
    "ts_decay_linear": decay_linear,
    "product": ts_product,
    "ts_arg_max": ts_argmax,
    "ts_arg_min": ts_argmin,
    # WorldQuant BRAIN operators (wq_ prefix)
    "wq_ts_rank": wq_ts_rank,
    "wq_ts_argmax": wq_ts_argmax,
    "wq_ts_argmin": wq_ts_argmin,
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
    "decay_linear": DecayLinear(),
    "ts_product": TsProduct(),
    # Aliases pointing to canonical operator instances
    "stddev": TsStd(),
    "sma": TsMean(),
    "ts_delta": Delta(),
    "ts_delay": Delay(),
    "ts_corr": Correlation(),
    "ts_covariance": Covariance(),
    "ts_std_dev": TsStd(),
    "ts_decay_linear": DecayLinear(),
    "product": TsProduct(),
    "ts_arg_max": TsArgmax(),
    "ts_arg_min": TsArgmin(),
    # WorldQuant BRAIN operators (wq_ prefix)
    "wq_ts_rank": WqTsRank(),
    "wq_ts_argmax": WqTsArgmax(),
    "wq_ts_argmin": WqTsArgmin(),
}
