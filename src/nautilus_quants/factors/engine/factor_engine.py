# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor Engine - Core computation engine for factors.

The FactorEngine is a Nautilus Actor that receives bar data,
computes factor values, and publishes results via MessageBus.
"""

from __future__ import annotations

import time
from collections import deque as _deque
from typing import TYPE_CHECKING, Any

import numpy as np

from nautilus_quants.factors.base.factor import ExpressionFactor, Factor
from nautilus_quants.factors.config import FactorConfig, load_factor_config
from nautilus_quants.factors.engine.dependency_resolver import DependencyResolver
from nautilus_quants.factors.expression import EvaluationContext, Evaluator, parse_expression
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import (
    TIME_SERIES_OPERATORS,
    TS_OPERATOR_INSTANCES,
)
from nautilus_quants.factors.types import FactorInput

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar


class FactorEngine:
    """
    Core engine for factor computation.
    
    Manages factor registration and incremental O(1) computation per bar.
    Can be used standalone or integrated with Nautilus as an Actor.

    Attributes:
        config: Factor configuration
        factors: Registered factor instances
    
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
        self.resolver = DependencyResolver()

        # Factor storage
        self._factors: dict[str, Factor] = {}
        self._variables: dict[str, str] = {}
        self._variable_asts: dict[str, Any] = {}
        self._parameters: dict[str, Any] = {}

        # Performance tracking
        self._compute_times: list[float] = []
        self._total_computes: int = 0
        self._warning_threshold_ms: float = 0.5
        self._enable_timing: bool = True

        # Build operators dict (math → time-series → CS stubs)
        self._operators: dict[str, Any] = {}
        self._operators.update(MATH_OPERATORS)
        self._operators.update(TIME_SERIES_OPERATORS)

        # Incremental operator state: instrument_id -> {call_site_key -> Incremental* instance}
        self._incremental_states: dict[str, dict[str, Any]] = {}
        # Extra bar fields (e.g. BinanceBar.quote_volume) to extract as scalars
        self._extra_fields: list[str] = []

        # Reuse single counter list across bar evaluations (avoids list allocation per bar)
        self._call_counter: list[int] = [0]
        # Per-instrument, per-factor cached Evaluator + ctx_vars dict
        # instrument_id -> factor_name -> (Evaluator, ctx_vars_dict)
        self._eval_cache: dict[str, dict[str, tuple]] = {}

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
        
        # Pre-parse AST for the variable
        self._variable_asts[name] = parse_expression(expression)
    
    def _estimate_warmup(self, expression: str) -> int:
        """Estimate warmup period from expression."""
        import re
        
        # Find all numeric arguments to time-series functions
        pattern = r'(ts_\w+|delay|delta)\s*\([^,]+,\s*(\d+)'
        matches = re.findall(pattern, expression)
        
        if matches:
            return max(int(m[1]) for m in matches) + 1
        return 1
    
    def _compute_variable_cache(
        self,
        factor_input: FactorInput,
    ) -> dict[str, Any]:
        """Compute all variable expressions once for a given bar.
        
        Returns a dict of {var_name: computed_value} that can be passed
        to each factor's compute() as var_cache, avoiding redundant
        per-factor variable evaluation.
        """
        if not self._variable_asts:
            return {}

        # Build base variables from input data
        variables: dict[str, Any] = {
            "open": factor_input.history.get("open", np.array([factor_input.open])),
            "high": factor_input.history.get("high", np.array([factor_input.high])),
            "low": factor_input.history.get("low", np.array([factor_input.low])),
            "close": factor_input.history.get("close", np.array([factor_input.close])),
            "volume": factor_input.history.get("volume", np.array([factor_input.volume])),
        }
        # Inject extra bar fields (e.g. quote_volume, count) from history
        for key, arr in factor_input.history.items():
            if key not in variables:
                variables[key] = arr
        variables.update(self._parameters)
        
        evaluator = Evaluator(EvaluationContext(
            variables=variables,
            operators=self._operators,
            parameters=self._parameters,
        ))
        
        # Evaluate each variable in declaration order (later vars may reference earlier ones)
        cache: dict[str, Any] = {}
        for var_name, var_ast in self._variable_asts.items():
            var_value = evaluator.evaluate(var_ast)
            evaluator.context.set_variable(var_name, var_value)
            cache[var_name] = var_value
        
        return cache

    def _make_inc_operators(
        self,
        inst_states: dict[str, Any],
        factor_name: str,
        call_counter: list[int],
    ) -> dict[str, Any]:
        """Build operator closures for incremental factor evaluation.

        For each TS operator registered in TS_OPERATOR_INSTANCES, dispatches to
        one of four closure factories based on arity and incremental availability:

        - Single-data + make_incremental() → O(1) push path
        - Single-data + no make_incremental() → deque-backed O(window) fallback
        - Two-data + make_incremental() → O(1) push(x, y) path
        - Two-data + no make_incremental() → deque-backed O(window) fallback

        call_counter must be reset to 0 before each factor evaluation so that
        the n-th operator call within a factor always maps to the same key.

        New TS operators gain incremental support automatically by overriding
        make_incremental() in their operator class; no changes here are needed.
        """

        def _single_inc(op: Any, name: str) -> Any:
            def closure(data: Any, window: Any) -> float:
                call_counter[0] += 1
                w = int(window)
                key = f"{factor_name}/{name}_{w}_{call_counter[0]}"
                if key not in inst_states:
                    inst_states[key] = op.make_incremental(w)
                return inst_states[key].push(float(data))
            return closure

        def _single_deque(batch_fn: Any, name: str) -> Any:
            def closure(data: Any, window: Any) -> float:
                call_counter[0] += 1
                w = int(window)
                key = f"{factor_name}/{name}_{w}_{call_counter[0]}"
                if key not in inst_states:
                    inst_states[key] = _deque(maxlen=w)
                inst_states[key].append(float(data))
                if len(inst_states[key]) < w:
                    return float("nan")
                return float(batch_fn(np.array(inst_states[key]), w))
            return closure

        def _two_data_inc(op: Any, name: str) -> Any:
            def closure(data1: Any, data2: Any, window: Any) -> float:
                call_counter[0] += 1
                w = int(window)
                key = f"{factor_name}/{name}_{w}_{call_counter[0]}"
                if key not in inst_states:
                    inst_states[key] = op.make_incremental(w)
                return inst_states[key].push(float(data1), float(data2))
            return closure

        def _two_data_deque(batch_fn: Any, name: str) -> Any:
            def closure(data1: Any, data2: Any, window: Any) -> float:
                call_counter[0] += 1
                w = int(window)
                k1 = f"{factor_name}/{name}_d0_{w}_{call_counter[0]}"
                k2 = f"{factor_name}/{name}_d1_{w}_{call_counter[0]}"
                if k1 not in inst_states:
                    inst_states[k1] = _deque(maxlen=w)
                    inst_states[k2] = _deque(maxlen=w)
                inst_states[k1].append(float(data1))
                inst_states[k2].append(float(data2))
                if len(inst_states[k1]) < w:
                    return float("nan")
                return float(batch_fn(np.array(inst_states[k1]), np.array(inst_states[k2]), w))
            return closure

        ops: dict[str, Any] = {}
        for op_name, op_instance in TS_OPERATOR_INSTANCES.items():
            is_two_data = op_instance.min_args >= 3
            has_incremental = op_instance.make_incremental(1) is not None
            if is_two_data:
                if has_incremental:
                    ops[op_name] = _two_data_inc(op_instance, op_name)
                else:
                    ops[op_name] = _two_data_deque(TIME_SERIES_OPERATORS[op_name], op_name)
            else:
                if has_incremental:
                    ops[op_name] = _single_inc(op_instance, op_name)
                else:
                    ops[op_name] = _single_deque(TIME_SERIES_OPERATORS[op_name], op_name)
        return ops

    def on_bar(self, bar: Bar) -> dict[str, dict[str, float]]:
        """
        Process a bar and compute factor values.

        Args:
            bar: The bar to process

        Returns:
            dict of {factor_name: {instrument_id: value}} for all computed factors.
        """
        start_time = time.perf_counter() if self._enable_timing else 0

        instrument_id = str(bar.bar_type.instrument_id)

        # Per-instrument incremental state dict (lazily created)
        inst_states = self._incremental_states.setdefault(instrument_id, {})

        # Reuse counter list to avoid per-bar allocation
        call_counter = self._call_counter
        # Per-instrument eval cache (populated lazily on first bar per instrument/factor)
        inst_evals = self._eval_cache.setdefault(instrument_id, {})

        # ── Compute all factors ────────────────────────────────────────────
        factor_results: dict[str, dict[str, float]] = {}

        # Pre-extract OHLCV + extra_fields once per bar (all factors share)
        _bar_open   = float(bar.open)
        _bar_high   = float(bar.high)
        _bar_low    = float(bar.low)
        _bar_close  = float(bar.close)
        _bar_volume = float(bar.volume)
        _bar_extra  = {ef: float(getattr(bar, ef, 0.0)) for ef in self._extra_fields}

        for factor_name, factor in self._factors.items():
            try:
                if isinstance(factor, ExpressionFactor):
                    # ── Incremental O(1) path (cached Evaluator) ───────────
                    call_counter[0] = 0

                    if factor_name not in inst_evals:
                        # First bar for this (instrument, factor): build and cache evaluator
                        inc_ops = self._make_inc_operators(inst_states, factor_name, call_counter)
                        merged_ops: dict[str, Any] = dict(MATH_OPERATORS)
                        merged_ops.update(inc_ops)
                        ctx_vars: dict[str, Any] = {
                            "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
                        }
                        for ef in self._extra_fields:
                            ctx_vars[ef] = 0.0
                        ctx_vars.update(self._parameters)
                        context = EvaluationContext(
                            variables=ctx_vars,
                            operators=merged_ops,
                            parameters=self._parameters,
                        )
                        evaluator = Evaluator(context)
                        inst_evals[factor_name] = (evaluator, ctx_vars)

                    evaluator, ctx_vars = inst_evals[factor_name]

                    # Update OHLCV scalars in-place using pre-extracted values
                    ctx_vars["open"]   = _bar_open
                    ctx_vars["high"]   = _bar_high
                    ctx_vars["low"]    = _bar_low
                    ctx_vars["close"]  = _bar_close
                    ctx_vars["volume"] = _bar_volume
                    if _bar_extra:
                        ctx_vars.update(_bar_extra)

                    # Evaluate engine-level variable expressions in incremental mode
                    for var_name, var_ast in factor._variable_asts.items():
                        ctx_vars[var_name] = evaluator.evaluate(var_ast)

                    # Fast path: skip compute() init overhead
                    value = factor.update_with_evaluator(evaluator)
                else:
                    continue  # non-ExpressionFactor not supported in incremental mode

                if factor_name not in factor_results:
                    factor_results[factor_name] = {}
                factor_results[factor_name][instrument_id] = value
            except Exception:
                factor_results.setdefault(factor_name, {})[instrument_id] = float("nan")

        # Track timing
        if self._enable_timing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._compute_times.append(elapsed_ms)
            self._total_computes += 1

        return factor_results
    
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
        
        # Build variable cache once
        var_cache = self._compute_variable_cache(factor_input)
        
        results: dict[str, float] = {}
        for factor_name, factor in self._factors.items():
            try:
                results[factor_name] = factor.compute(factor_input, var_cache=var_cache)
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
    
    def set_extra_fields(self, fields: list[str]) -> None:
        """Set extra bar fields to track (e.g. BinanceBar extended fields)."""
        self._extra_fields = list(fields)

    def reset(self) -> None:
        """Reset engine state."""
        for factor in self._factors.values():
            factor.reset()
        self._compute_times.clear()
        self._total_computes = 0
        self._incremental_states.clear()
        self._eval_cache.clear()
        self._call_counter[0] = 0
