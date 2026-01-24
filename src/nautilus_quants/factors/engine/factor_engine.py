# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Engine - Core computation engine for factors.

The FactorEngine is a Nautilus Actor that receives bar data,
computes factor values, and publishes results via MessageBus.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from nautilus_quants.factors.base.factor import ExpressionFactor, Factor
from nautilus_quants.factors.config import FactorConfig, load_factor_config
from nautilus_quants.factors.engine.data_synchronizer import DataSynchronizer
from nautilus_quants.factors.engine.dependency_resolver import DependencyResolver
from nautilus_quants.factors.expression import EvaluationContext, Evaluator, parse_expression
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TIME_SERIES_OPERATORS
from nautilus_quants.factors.types import FactorInput, FactorValues

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar


class FactorEngine:
    """
    Core engine for factor computation.
    
    Manages factor registration, data synchronization, and computation.
    Can be used standalone or integrated with Nautilus as an Actor.
    
    Attributes:
        config: Factor configuration
        factors: Registered factor instances
        synchronizer: Multi-instrument data synchronizer
    
    Example:
        ```python
        engine = FactorEngine()
        engine.load_config("config/factors.yaml")
        
        # Process bars
        for bar in bars:
            result = engine.on_bar(bar)
            if result:
                print(result.factors)
        ```
    """
    
    def __init__(
        self,
        config: FactorConfig | None = None,
        max_history: int = 500,
    ) -> None:
        """
        Initialize the FactorEngine.
        
        Args:
            config: Optional factor configuration
            max_history: Maximum history to maintain per instrument
        """
        self.config = config
        self.max_history = max_history
        
        # Core components
        self.synchronizer = DataSynchronizer(max_history=max_history)
        self.resolver = DependencyResolver()
        
        # Factor storage
        self._factors: dict[str, Factor] = {}
        self._variables: dict[str, str] = {}
        self._parameters: dict[str, Any] = {}
        
        # Performance tracking
        self._compute_times: list[float] = []
        self._total_computes: int = 0
        self._warning_threshold_ms: float = 0.5
        self._enable_timing: bool = True
        
        # Build operators dict
        self._operators: dict[str, Any] = {}
        self._operators.update(MATH_OPERATORS)
        self._operators.update(TIME_SERIES_OPERATORS)
        
        # Load config if provided
        if config:
            self._apply_config(config)
    
    def load_config(self, path: str) -> None:
        """
        Load factor configuration from file.
        
        Args:
            path: Path to YAML configuration file
        """
        self.config = load_factor_config(path)
        self._apply_config(self.config)
    
    def _apply_config(self, config: FactorConfig) -> None:
        """Apply configuration to engine."""
        # Load parameters
        self._parameters = config.parameters.copy()
        
        # Load variables
        for var_name, var_expr in config.variables.items():
            self.register_variable(var_name, var_expr)
        
        # Load factors
        for factor_def in config.factors:
            self.register_expression_factor(
                name=factor_def.name,
                expression=factor_def.expression,
                description=factor_def.description,
            )
        
        # Performance settings
        self._warning_threshold_ms = config.performance.warning_threshold_ms
        self._enable_timing = config.performance.enable_timing
    
    def register_factor(self, factor: Factor) -> None:
        """
        Register a factor instance.
        
        Args:
            factor: Factor to register
        """
        self._factors[factor.name] = factor
    
    def register_expression_factor(
        self,
        name: str,
        expression: str,
        description: str = "",
        warmup_period: int = 0,
    ) -> None:
        """
        Register a factor defined by expression.
        
        Args:
            name: Factor name
            expression: Alpha101-style expression
            description: Optional description
            warmup_period: Warmup period before valid output
        """
        # Determine warmup from expression if not specified
        if warmup_period == 0:
            warmup_period = self._estimate_warmup(expression)
        
        factor = ExpressionFactor(
            name=name,
            expression=expression,
            description=description,
            warmup_period=warmup_period,
            parameters=self._parameters,
            variables=self._variables,
        )
        
        self._factors[name] = factor
        self.resolver.add_factor(name, expression, factor)
    
    def register_variable(self, name: str, expression: str) -> None:
        """
        Register a reusable variable.
        
        Args:
            name: Variable name
            expression: Expression defining the variable
        """
        self._variables[name] = expression
        self.resolver.add_variable(name, expression)
    
    def _estimate_warmup(self, expression: str) -> int:
        """Estimate warmup period from expression."""
        import re
        
        # Find all numeric arguments to time-series functions
        pattern = r'(ts_\w+|delay|delta)\s*\([^,]+,\s*(\d+)'
        matches = re.findall(pattern, expression)
        
        if matches:
            return max(int(m[1]) for m in matches) + 1
        return 1
    
    def on_bar(self, bar: Bar) -> FactorValues | None:
        """
        Process a bar and compute factor values.
        
        Args:
            bar: The bar to process
            
        Returns:
            FactorValues if computation succeeded, None otherwise
        """
        start_time = time.perf_counter() if self._enable_timing else 0
        
        # Update synchronizer
        instrument_id = str(bar.bar_type.instrument_id)
        self.synchronizer.on_bar(bar)
        
        # Get instrument data
        inst_data = self.synchronizer.get_instrument_data(instrument_id)
        if inst_data is None:
            return None
        
        # Build factor input
        arrays = inst_data.get_arrays()
        factor_input = FactorInput(
            instrument_id=bar.bar_type.instrument_id,
            timestamp_ns=bar.ts_event,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            history=arrays,
        )
        
        # Compute all factors
        factor_results: dict[str, dict[str, float]] = {}
        
        for factor_name, factor in self._factors.items():
            try:
                value = factor.update(factor_input)
                if factor_name not in factor_results:
                    factor_results[factor_name] = {}
                factor_results[factor_name][instrument_id] = value
            except Exception as e:
                # Log error but continue with other factors
                factor_results.setdefault(factor_name, {})[instrument_id] = float('nan')
        
        # Track timing
        if self._enable_timing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._compute_times.append(elapsed_ms)
            self._total_computes += 1
            
            if elapsed_ms > self._warning_threshold_ms:
                pass  # Would log warning in production
        
        return FactorValues(
            ts_event=bar.ts_event,
            factors=factor_results,
        )
    
    def compute_factors(
        self,
        instrument_id: str,
        history: dict[str, np.ndarray],
        current_bar: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute all factors for given data.
        
        This is a lower-level method for direct computation without
        going through the bar processing pipeline.
        
        Args:
            instrument_id: Instrument identifier
            history: Historical data arrays
            current_bar: Current bar OHLCV data
            
        Returns:
            Dict of {factor_name: value}
        """
        factor_input = FactorInput(
            instrument_id=None,  # type: ignore
            timestamp_ns=0,
            open=current_bar.get("open", current_bar.get("close", 0)),
            high=current_bar.get("high", current_bar.get("close", 0)),
            low=current_bar.get("low", current_bar.get("close", 0)),
            close=current_bar.get("close", 0),
            volume=current_bar.get("volume", 0),
            history=history,
        )
        
        results: dict[str, float] = {}
        for factor_name, factor in self._factors.items():
            try:
                results[factor_name] = factor.compute(factor_input)
            except Exception:
                results[factor_name] = float('nan')
        
        return results
    
    def evaluate_expression(
        self,
        expression: str,
        history: dict[str, np.ndarray],
    ) -> float:
        """
        Evaluate a single expression with given data.
        
        Args:
            expression: Expression to evaluate
            history: Historical data arrays
            
        Returns:
            Computed value
        """
        ast = parse_expression(expression)
        
        # Build context
        variables: dict[str, Any] = {}
        variables.update(history)
        variables.update(self._parameters)
        
        # Evaluate variable expressions first
        evaluator = Evaluator(EvaluationContext(
            variables=variables,
            operators=self._operators,
            parameters=self._parameters,
        ))
        
        for var_name, var_expr in self._variables.items():
            var_ast = parse_expression(var_expr)
            var_value = evaluator.evaluate(var_ast)
            evaluator.context.set_variable(var_name, var_value)
        
        result = evaluator.evaluate(ast)
        
        if isinstance(result, np.ndarray):
            return float(result[-1]) if len(result) > 0 else float('nan')
        return float(result)
    
    @property
    def factor_names(self) -> list[str]:
        """Get list of registered factor names."""
        return list(self._factors.keys())
    
    @property
    def variable_names(self) -> list[str]:
        """Get list of registered variable names."""
        return list(self._variables.keys())
    
    def get_factor(self, name: str) -> Factor | None:
        """Get a factor by name."""
        return self._factors.get(name)
    
    def get_execution_order(self) -> list[str]:
        """Get factor execution order based on dependencies."""
        return self.resolver.resolve()
    
    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        if not self._compute_times:
            return {
                "mean_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0,
                "total_computes": 0,
            }
        
        return {
            "mean_ms": float(np.mean(self._compute_times)),
            "max_ms": float(np.max(self._compute_times)),
            "min_ms": float(np.min(self._compute_times)),
            "p95_ms": float(np.percentile(self._compute_times, 95)),
            "total_computes": self._total_computes,
        }
    
    def reset(self) -> None:
        """Reset engine state."""
        self.synchronizer.reset()
        for factor in self._factors.values():
            factor.reset()
        self._compute_times.clear()
        self._total_computes = 0
