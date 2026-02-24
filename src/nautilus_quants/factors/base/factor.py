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

from nautilus_quants.factors.expression import EvaluationContext, Evaluator, parse_expression
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TIME_SERIES_OPERATORS

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
    
    def update(self, data: FactorInput, var_cache: dict[str, Any] | None = None, extra_operators: dict[str, Any] | None = None) -> float:
        """
        Update factor with new data and return computed value.

        This method handles warmup period tracking and calls compute().

        Args:
            data: Factor input containing current and historical data
            var_cache: Pre-computed variable values from the engine.
            extra_operators: Incremental operator closures (forwarded to compute()).

        Returns:
            Computed factor value, or NaN if not warmed up
        """
        self._compute_count += 1

        if extra_operators is not None:
            # ── Incremental mode ──────────────────────────────────────────────
            # Always call compute() so incremental operators receive this bar's
            # value even during Factor warmup.  Then apply Factor-level warmup
            # gating to the *result*: boolean expressions (e.g. close > ts_max)
            # evaluate to 0.0 rather than NaN when inputs are NaN, so operator-
            # level NaN propagation alone is insufficient.
            try:
                raw = self.compute(data, var_cache=var_cache, extra_operators=extra_operators)
            except TypeError:
                try:
                    raw = self.compute(data, var_cache=var_cache)
                except TypeError:
                    raw = self.compute(data)

            if not self._is_warmed_up:
                if self._compute_count >= self.warmup_period:
                    self._is_warmed_up = True
                else:
                    return float('nan')  # Operators fed; result gated until warm
            return raw

        else:
            # ── Batch mode ────────────────────────────────────────────────────
            # Gate computation before calling compute(); DataSynchronizer
            # accumulates history in the background regardless.
            if not self._is_warmed_up:
                if self._compute_count >= self.warmup_period:
                    self._is_warmed_up = True
                else:
                    return float('nan')

            try:
                return self.compute(data, var_cache=var_cache, extra_operators=None)
            except TypeError:
                try:
                    return self.compute(data, var_cache=var_cache)
                except TypeError:
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
        
        # Parse the expression
        self._ast = parse_expression(expression)
        
        # Parse variable expressions
        self._variable_asts: dict[str, Any] = {}
        for var_name, var_expr in self.variables.items():
            self._variable_asts[var_name] = parse_expression(var_expr)

        # Pre-build batch operators (static, never changes)
        self._batch_operators: dict[str, Any] = dict(MATH_OPERATORS)
        self._batch_operators.update(TIME_SERIES_OPERATORS)
    
    def compute(
        self,
        data: FactorInput,
        var_cache: dict[str, Any] | None = None,
        extra_operators: dict[str, Any] | None = None,
    ) -> float:
        """Compute factor value by evaluating the expression.

        Args:
            data: Factor input containing current bar and history.
            var_cache: Pre-computed variable values from the engine.
                When provided, skips redundant per-factor variable
                evaluation — the main performance optimization.
            extra_operators: When provided, replaces TIME_SERIES_OPERATORS
                with the given callables and evaluates variables as scalars
                (incremental mode).  Each callable in ``extra_operators``
                must accept the same positional arguments as the
                corresponding function in TIME_SERIES_OPERATORS but may
                receive scalar inputs instead of numpy arrays.
        """
        # Build operator dict — incremental operators override batch ones
        if extra_operators is not None:
            operators: dict[str, Any] = dict(MATH_OPERATORS)
            operators.update(extra_operators)
        else:
            operators = self._batch_operators  # reuse pre-built, no copy

        # Build variable dict
        if extra_operators is not None:
            # ── Incremental / scalar mode ──────────────────────────────
            # Variables are current-bar scalars; TS operators maintain
            # their own O(1) state and do NOT need the full history array.
            variables: dict[str, Any] = {
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "close": data.close,
                "volume": data.volume,
            }
            for key, arr in data.history.items():
                if key not in variables:
                    variables[key] = float(arr[-1]) if len(arr) > 0 else float("nan")
        else:
            # ── Batch / array mode (existing behaviour) ────────────────
            variables = {
                "open": data.history.get("open", np.array([data.open])),
                "high": data.history.get("high", np.array([data.high])),
                "low": data.history.get("low", np.array([data.low])),
                "close": data.history.get("close", np.array([data.close])),
                "volume": data.history.get("volume", np.array([data.volume])),
            }
            for key, arr in data.history.items():
                if key not in variables:
                    variables[key] = arr

        # Add parameters
        variables.update(self.parameters)

        # Inject pre-computed variable values from engine cache (batch mode only)
        if var_cache and extra_operators is None:
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
            result = float(result[-1]) if len(result) > 0 else float("nan")

        return float(result)

    def update_with_evaluator(self, evaluator: Evaluator) -> float:
        """快速路径：使用预构建 Evaluator 的增量 O(1) 评估。

        调用前：调用方已将 OHLCV 及 variable 值更新至 evaluator.context.variables。
        行为等价于 Factor.update(data, extra_operators=<non-None>)。
        """
        self._compute_count += 1

        try:
            result = evaluator.evaluate(self._ast)
        except Exception:
            if not self._is_warmed_up and self._compute_count >= self.warmup_period:
                self._is_warmed_up = True
            return float("nan")

        if isinstance(result, np.ndarray):
            result = float(result[-1]) if len(result) > 0 else float("nan")
        else:
            result = float(result)

        if not self._is_warmed_up:
            if self._compute_count >= self.warmup_period:
                self._is_warmed_up = True
            else:
                return float("nan")

        return result
