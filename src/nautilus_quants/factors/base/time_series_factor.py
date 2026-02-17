# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Time-Series Factor Base Class.

Specialized factor class for time-series computations on a single instrument.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from nautilus_quants.factors.base.factor import Factor

if TYPE_CHECKING:
    from nautilus_quants.factors.types import FactorInput


class TimeSeriesFactor(Factor):
    """
    Base class for time-series factors.
    
    Time-series factors operate on historical data for a single instrument,
    computing rolling statistics, momentum indicators, etc.
    
    Subclasses should implement the `compute_from_history` method which
    receives the historical data arrays directly.
    
    Attributes:
        lookback: Required history length for computation
    """
    
    def __init__(
        self,
        name: str,
        lookback: int,
        description: str = "",
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            warmup_period=lookback,
        )
        self.lookback = lookback
    
    def compute(self, data: FactorInput, var_cache: dict | None = None) -> float:
        """
        Compute factor value from input data.
        
        Extracts history and delegates to compute_from_history.
        """
        # Get close price history (most common case)
        close_history = data.history.get("close")
        
        if close_history is None or len(close_history) < self.lookback:
            return float('nan')
        
        return self.compute_from_history(
            close=close_history,
            open_=data.history.get("open"),
            high=data.history.get("high"),
            low=data.history.get("low"),
            volume=data.history.get("volume"),
        )
    
    @abstractmethod
    def compute_from_history(
        self,
        close: np.ndarray,
        open_: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> float:
        """
        Compute factor value from historical data arrays.
        
        Args:
            close: Close price history
            open_: Open price history (optional)
            high: High price history (optional)
            low: Low price history (optional)
            volume: Volume history (optional)
            
        Returns:
            Computed factor value
        """
        pass


class MomentumFactor(TimeSeriesFactor):
    """Simple momentum factor: (close[t] - close[t-n]) / close[t-n]."""
    
    def __init__(self, lookback: int = 20, name: str | None = None) -> None:
        super().__init__(
            name=name or f"momentum_{lookback}",
            lookback=lookback,
            description=f"{lookback}-period momentum",
        )
    
    def compute_from_history(
        self,
        close: np.ndarray,
        open_: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> float:
        """Compute momentum as percentage return over lookback period."""
        if len(close) < self.lookback + 1:
            return float('nan')
        
        current = close[-1]
        past = close[-self.lookback - 1]
        
        if past == 0:
            return float('nan')
        
        return float((current - past) / past)


class VolatilityFactor(TimeSeriesFactor):
    """Rolling volatility (standard deviation of returns)."""
    
    def __init__(self, lookback: int = 20, name: str | None = None) -> None:
        super().__init__(
            name=name or f"volatility_{lookback}",
            lookback=lookback + 1,  # Need extra point for returns
            description=f"{lookback}-period volatility",
        )
        self._returns_lookback = lookback
    
    def compute_from_history(
        self,
        close: np.ndarray,
        open_: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> float:
        """Compute volatility as std of log returns."""
        if len(close) < self._returns_lookback + 1:
            return float('nan')
        
        # Compute log returns
        prices = close[-(self._returns_lookback + 1):]
        returns = np.diff(np.log(prices))
        
        return float(np.std(returns, ddof=1))


class MeanReversionFactor(TimeSeriesFactor):
    """Mean reversion factor: (close - SMA) / std."""
    
    def __init__(self, lookback: int = 20, name: str | None = None) -> None:
        super().__init__(
            name=name or f"mean_reversion_{lookback}",
            lookback=lookback,
            description=f"{lookback}-period mean reversion (z-score)",
        )
    
    def compute_from_history(
        self,
        close: np.ndarray,
        open_: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> float:
        """Compute z-score from moving average."""
        if len(close) < self.lookback:
            return float('nan')
        
        window = close[-self.lookback:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        
        if std == 0:
            return 0.0
        
        current = close[-1]
        return float((current - mean) / std)
