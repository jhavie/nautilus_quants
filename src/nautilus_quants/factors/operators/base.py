# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Operator Base Classes.

This module defines the abstract base class for all operators used in
factor expression evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nautilus_quants.factors.types import FactorInput


class OperatorType(Enum):
    """Type classification for operators."""
    TIME_SERIES = "time_series"      # Operates on single instrument over time
    CROSS_SECTIONAL = "cross_sectional"  # Operates across instruments at one time
    MATH = "math"                    # Pure mathematical operations
    ELEMENT_WISE = "element_wise"    # Element-wise array operations


class Operator(ABC):
    """
    Abstract base class for all operators.
    
    Operators are the building blocks of factor expressions. They take
    input values and compute results based on their specific logic.
    
    Attributes:
        name: Operator identifier used in expressions (e.g., 'ts_mean')
        operator_type: Classification of the operator
        min_args: Minimum number of arguments
        max_args: Maximum number of arguments (None for unlimited)
    """
    
    name: str
    operator_type: OperatorType
    min_args: int = 1
    max_args: int | None = None
    
    def __init__(self) -> None:
        """Initialize the operator."""
        pass
    
    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> float | np.ndarray:
        """
        Compute the operator result.
        
        Args:
            *args: Positional arguments (operator-specific)
            **kwargs: Keyword arguments (operator-specific)
            
        Returns:
            Computed result (scalar or array)
        """
        pass
    
    def validate_args(self, args: tuple[Any, ...]) -> None:
        """
        Validate the number of arguments.
        
        Args:
            args: Arguments to validate
            
        Raises:
            ValueError: If argument count is invalid
        """
        n_args = len(args)
        
        if n_args < self.min_args:
            raise ValueError(
                f"Operator '{self.name}' requires at least {self.min_args} "
                f"arguments, got {n_args}"
            )
        
        if self.max_args is not None and n_args > self.max_args:
            raise ValueError(
                f"Operator '{self.name}' accepts at most {self.max_args} "
                f"arguments, got {n_args}"
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class TimeSeriesOperator(Operator):
    """
    Base class for time-series operators.
    
    Time-series operators work on historical data for a single instrument,
    computing values like moving averages, delays, and rolling statistics.
    """
    
    operator_type = OperatorType.TIME_SERIES

    @abstractmethod
    def compute(
        self,
        data: np.ndarray,
        window: int,
        **kwargs: Any,
    ) -> float | np.ndarray:
        """
        Compute the time-series operator result.

        Args:
            data: Historical data array (most recent at the end)
            window: Lookback window size
            **kwargs: Additional operator-specific arguments

        Returns:
            Computed result
        """
        pass

    def compute_vectorized(self, data: pd.Series, window: int, **kwargs: Any) -> pd.Series:
        """Vectorized computation over full time-series.

        Default: rolling apply wrapping scalar ``compute``. Override for performance.
        """
        return data.rolling(window).apply(
            lambda x: self.compute(x.values, window, **kwargs), raw=False,
        )


class CrossSectionalOperator(Operator):
    """
    Base class for cross-sectional operators.
    
    Cross-sectional operators work across multiple instruments at a single
    point in time, computing values like rankings and z-scores.
    """
    
    operator_type = OperatorType.CROSS_SECTIONAL

    @abstractmethod
    def compute(
        self,
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute the cross-sectional operator result.

        Args:
            values: Dict of {instrument_id: value} for all instruments
            **kwargs: Additional operator-specific arguments

        Returns:
            Dict of {instrument_id: computed_value}
        """
        pass

    def compute_vectorized(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Vectorized computation across instruments for all timestamps.

        Default: apply scalar ``compute`` row-by-row. Override for performance.
        """
        def _apply_row(row: pd.Series) -> pd.Series:
            values = row.dropna().to_dict()
            if not values:
                return row
            result = self.compute(values, **kwargs)
            return pd.Series(result, index=row.index)
        return df.apply(_apply_row, axis=1)


class MathOperator(Operator):
    """
    Base class for mathematical operators.
    
    Math operators perform pure mathematical computations like log, abs, sign.
    """
    
    operator_type = OperatorType.MATH
    
    @abstractmethod
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """
        Compute the mathematical operation.
        
        Args:
            value: Input value (scalar or array)
            **kwargs: Additional operator-specific arguments
            
        Returns:
            Computed result
        """
        pass


# Operator Registry
_OPERATOR_REGISTRY: dict[str, type[Operator]] = {}


def register_operator(cls: type[Operator]) -> type[Operator]:
    """
    Decorator to register an operator class.
    
    Args:
        cls: Operator class to register
        
    Returns:
        The same class (for decorator chaining)
        
    Example:
        ```python
        @register_operator
        class TsMean(TimeSeriesOperator):
            name = "ts_mean"
            ...
        ```
    """
    if not hasattr(cls, 'name') or not cls.name:
        raise ValueError(f"Operator class {cls.__name__} must have a 'name' attribute")
    
    _OPERATOR_REGISTRY[cls.name] = cls
    return cls


def get_operator(name: str) -> Operator:
    """
    Get an operator instance by name.
    
    Args:
        name: Operator name (e.g., 'ts_mean', 'cs_rank')
        
    Returns:
        Operator instance
        
    Raises:
        KeyError: If operator is not registered
    """
    if name not in _OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator: '{name}'")
    
    return _OPERATOR_REGISTRY[name]()


def get_all_operators() -> dict[str, type[Operator]]:
    """
    Get all registered operators.
    
    Returns:
        Dict of {name: operator_class}
    """
    return _OPERATOR_REGISTRY.copy()


def list_operators() -> list[str]:
    """
    List all registered operator names.
    
    Returns:
        Sorted list of operator names
    """
    return sorted(_OPERATOR_REGISTRY.keys())
