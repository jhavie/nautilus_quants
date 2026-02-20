# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Base Classes.

This module defines the abstract base class for all factors and
common factor implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nautilus_quants.factors.types import FactorInput


@dataclass
class FactorMetadata:
    """Metadata for a factor."""
    name: str
    description: str = ""
    category: str = ""
    version: str = "1.0"


class Factor(ABC):
    """
    Abstract base class for all factors.
    
    A Factor takes input data and computes a numeric value that can be
    used for trading signals or portfolio construction.
    
    Attributes:
        name: Unique identifier for the factor
        description: Human-readable description
        warmup_period: Minimum history required before valid output
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        warmup_period: int = 0,
    ) -> None:
        self.name = name
        self.description = description
        self.warmup_period = warmup_period
        self._is_warmed_up = False
        self._compute_count = 0
    
    @abstractmethod
    def compute(self, data: FactorInput, var_cache: dict[str, Any] | None = None) -> float:
        """
        Compute the factor value for given input.
        
        Args:
            data: Factor input containing current and historical data
            var_cache: Pre-computed variable values from the engine.
                ExpressionFactor uses this to avoid redundant variable
                evaluation. Other subclasses may ignore it.
            
        Returns:
            Computed factor value (float)
        """
        pass
    
    def update(self, data: FactorInput, var_cache: dict[str, Any] | None = None) -> float:
        """
        Update factor with new data and return computed value.
        
        This method handles warmup period tracking and calls compute().
        
        Args:
            data: Factor input containing current and historical data
            var_cache: Pre-computed variable values from the engine.
            
        Returns:
            Computed factor value, or NaN if not warmed up
        """
        self._compute_count += 1
        
        # Check warmup
        if not self._is_warmed_up:
            if self._compute_count >= self.warmup_period:
                self._is_warmed_up = True
            else:
                return float('nan')
        
        try:
            return self.compute(data, var_cache=var_cache)
        except TypeError:
            # Backward compatibility: subclass hasn't adopted var_cache parameter
            return self.compute(data)
    
    @property
    def is_warmed_up(self) -> bool:
        """Whether the factor has enough history for valid output."""
        return self._is_warmed_up
    
    def reset(self) -> None:
        """Reset factor state."""
        self._is_warmed_up = False
        self._compute_count = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ExpressionFactor(Factor):
    """
    Factor defined by an Alpha101-style expression.
    
    Parses and evaluates expressions like "ts_mean(close, 20) / ts_std(close, 20)".
    
    Example:
        ```python
        factor = ExpressionFactor(
            name="zscore_20",
            expression="(close - ts_mean(close, 20)) / ts_std(close, 20)",
            warmup_period=20,
        )
        value = factor.compute(data)
        ```
    """
    
    def __init__(
        self,
        name: str,
        expression: str,
        description: str = "",
        warmup_period: int = 0,
        parameters: dict[str, Any] | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name, description, warmup_period)
        self.expression = expression
        self.parameters = parameters or {}
        self.variables = variables or {}
        
        # Lazy import to avoid circular dependency
        from nautilus_quants.factors.expression import parse_expression
        
        # Parse the expression
        self._ast = parse_expression(expression)
        
        # Parse variable expressions
        self._variable_asts: dict[str, Any] = {}
        for var_name, var_expr in self.variables.items():
            self._variable_asts[var_name] = parse_expression(var_expr)
    
    def compute(self, data: FactorInput, var_cache: dict[str, Any] | None = None) -> float:
        """Compute factor value by evaluating the expression.
        
        Args:
            data: Factor input containing current bar and history.
            var_cache: Pre-computed variable values from the engine.
                When provided, skips redundant per-factor variable
                evaluation — the main performance optimization.
        """
        from nautilus_quants.factors.expression import EvaluationContext, Evaluator
        from nautilus_quants.factors.operators.math import MATH_OPERATORS
        from nautilus_quants.factors.operators.time_series import TIME_SERIES_OPERATORS
        
        # Build operator dict
        operators: dict[str, Any] = {}
        operators.update(MATH_OPERATORS)
        operators.update(TIME_SERIES_OPERATORS)
        
        # Build variable dict from input data
        variables: dict[str, Any] = {
            "open": data.history.get("open", np.array([data.open])),
            "high": data.history.get("high", np.array([data.high])),
            "low": data.history.get("low", np.array([data.low])),
            "close": data.history.get("close", np.array([data.close])),
            "volume": data.history.get("volume", np.array([data.volume])),
        }
        # Inject extra bar fields (e.g. quote_volume, count) from history
        for key, arr in data.history.items():
            if key not in variables:
                variables[key] = arr

        # Add parameters
        variables.update(self.parameters)
        
        # Inject pre-computed variable values from engine cache
        if var_cache:
            variables.update(var_cache)

        # Create evaluation context
        context = EvaluationContext(
            variables=variables,
            operators=operators,
            parameters=self.parameters,
        )
        evaluator = Evaluator(context)

        # Evaluate variable expressions if not pre-computed by engine
        if not var_cache:
            for var_name, var_ast in self._variable_asts.items():
                var_value = evaluator.evaluate(var_ast)
                context.set_variable(var_name, var_value)

        # Evaluate main expression
        result = evaluator.evaluate(self._ast)
        
        # Ensure scalar output
        if isinstance(result, np.ndarray):
            result = float(result[-1]) if len(result) > 0 else float('nan')
        
        return float(result)
