# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorEngine — Unified factor engine using panel (matrix) architecture.

Replaces the two-phase engine with a single engine that evaluates all
expressions (including nested CS/TS operators) correctly via panel
``pd.DataFrame[T x N]`` intermediates.

Usage::

    engine = FactorEngine(config=factor_config, max_history=500)

    # On each bar:
    engine.on_bar("AAPL", {"open": 150, "close": 155, ...}, ts=1)
    engine.on_bar("GOOGL", {"open": 2800, "close": 2850, ...}, ts=1)

    # When all instruments report for a timestamp:
    results = engine.flush_and_compute(ts=1)
    # results = {"alpha001": {"AAPL": 0.42, "GOOGL": -0.15}, ...}
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from nautilus_quants.factors.config import FactorConfig, load_factor_config
from nautilus_quants.factors.engine.buffer import Buffer
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.expression.ast import ASTNode
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

logger = logging.getLogger(__name__)


class FactorEngine:
    """Unified panel-based factor engine.

    Accumulates bar data in a :class:`Buffer` and evaluates factor
    ASTs via :class:`Evaluator`.  All intermediate results are
    ``pd.DataFrame[T x N]`` which naturally handles any nesting of
    cross-sectional (row-wise) and time-series (column-wise) operators.

    Parameters
    ----------
    config : FactorConfig, optional
        Factor configuration (factors, variables, parameters).
    max_history : int
        Rolling window size for the panel buffer.
    extra_fields : tuple[str, ...]
        Additional bar fields beyond OHLCV (e.g. ``("quote_volume",)``).
    """

    def __init__(
        self,
        config: FactorConfig | None = None,
        max_history: int = 500,
        extra_fields: tuple[str, ...] = (),
    ) -> None:
        self._buffer = Buffer(max_history=max_history, extra_fields=extra_fields)
        self._max_history = max_history

        # Pre-parsed factor ASTs
        self._factors: dict[str, ASTNode] = {}
        self._factor_descriptions: dict[str, str] = {}

        # Pre-parsed variable ASTs
        self._variables: dict[str, ASTNode] = {}
        self._variable_order: list[str] = []

        # Config parameters
        self._parameters: dict[str, Any] = {}

        # Operator instances (shared, pre-allocated)
        self._ts_ops = dict(TS_OPERATOR_INSTANCES)
        self._cs_ops = dict(CS_OPERATOR_INSTANCES)
        self._math_ops = dict(MATH_OPERATORS)

        # Performance tracking
        self._compute_times: list[float] = []
        self._total_computes: int = 0
        self._enable_timing: bool = True

        # Cached evaluator (reused across flushes to avoid re-creation overhead)
        self._evaluator: Evaluator | None = None

        if config:
            self._apply_config(config)

    def load_config(self, path: str) -> None:
        """Load factor configuration from a YAML file."""
        config = load_factor_config(path)
        self._apply_config(config)

    def _apply_config(self, config: FactorConfig) -> None:
        """Apply a FactorConfig to the engine."""
        self._parameters = config.parameters.copy()

        # Register variables (order matters)
        for var_name, var_expr in config.variables.items():
            self.register_variable(var_name, var_expr)

        # Register factors
        for factor_def in config.all_factors:
            self.register_expression_factor(
                name=factor_def.name,
                expression=factor_def.expression,
                description=factor_def.description,
            )

        # Performance settings
        self._enable_timing = config.performance.enable_timing

    def register_expression_factor(
        self,
        name: str,
        expression: str,
        description: str = "",
    ) -> None:
        """Register a factor expression.

        Parameters
        ----------
        name : str
            Factor name (e.g. "alpha001").
        expression : str
            Alpha101-style expression string.
        description : str
            Optional human-readable description.
        """
        self._factors[name] = parse_expression(expression)
        self._factor_descriptions[name] = description

    def register_variable(self, name: str, expression: str) -> None:
        """Register a reusable variable expression.

        Variables are evaluated before factors and their results are
        injected into the panel fields for subsequent factor evaluation.
        """
        self._variables[name] = parse_expression(expression)
        if name not in self._variable_order:
            self._variable_order.append(name)

    def on_bar(
        self, instrument_id: str, bar_data: dict[str, float], ts: int
    ) -> None:
        """Append a bar to the panel buffer.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        bar_data : dict
            OHLCV data (and optional extra fields).
        ts : int
            Timestamp (nanoseconds).
        """
        self._buffer.append(instrument_id, ts, bar_data)

    def flush_and_compute(self, ts: int) -> dict[str, dict[str, float]]:
        """Flush the timestamp and compute all factors.

        Flushes staged data for ``ts``, builds panel DataFrames, evaluates
        all variable and factor expressions, and returns the latest-row
        values for each instrument.

        Parameters
        ----------
        ts : int
            Timestamp to flush and compute.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{factor_name: {instrument_id: value}}``.
            Instruments with NaN values are excluded.
        """
        pc = time.perf_counter

        t0 = pc()

        # Phase: flush
        self._buffer.flush_timestamp(ts)
        t1 = pc()

        # Phase: to_panel
        panel_fields: dict[str, pd.DataFrame | float] = self._buffer.to_panel()  # type: ignore[assignment]
        t2 = pc()

        # Inject config parameters so they're accessible as variables
        for p_name, p_val in self._parameters.items():
            panel_fields[p_name] = p_val

        # Reuse evaluator — only update the fields reference
        if self._evaluator is None:
            self._evaluator = Evaluator(
                panel_fields=panel_fields,
                ts_ops=self._ts_ops,
                cs_ops=self._cs_ops,
                math_ops=self._math_ops,
                parameters=self._parameters,
            )
        else:
            self._evaluator.update_fields(panel_fields)
        evaluator = self._evaluator

        # Phase: variables
        for var_name in self._variable_order:
            var_ast = self._variables[var_name]
            try:
                var_result = evaluator.evaluate(var_ast)
                panel_fields[var_name] = var_result
            except Exception:
                logger.debug("Variable '%s' evaluation failed", var_name, exc_info=True)
        t3 = pc()

        # Phase: factors
        results: dict[str, dict[str, float]] = {}
        instruments = self._buffer.instruments

        for name, ast in self._factors.items():
            try:
                result_df = evaluator.evaluate(ast)
                # Inject result into panel_fields so subsequent factors can
                # reference it (e.g. momentum_3h_norm references momentum_3h).
                # NaN propagates naturally: instruments with undefined values
                # are excluded from cross-sectional ops and final output.
                if isinstance(result_df, pd.DataFrame):
                    panel_fields[name] = result_df
                    # Extract last row via numpy — avoid Series creation
                    vals = result_df.values[-1]  # numpy array view, no copy
                    mask = ~np.isnan(vals)
                    if mask.any():
                        cols = result_df.columns
                        results[name] = {
                            cols[i]: float(vals[i])
                            for i in range(len(vals))
                            if mask[i]
                        }
                    else:
                        results[name] = {}
                elif isinstance(result_df, (int, float)):
                    panel_fields[name] = result_df
                    val = float(result_df)
                    if not np.isnan(val):
                        results[name] = {inst: val for inst in instruments}
                    else:
                        results[name] = {}
                else:
                    results[name] = {}
            except Exception:
                logger.warning("Factor '%s' evaluation failed", name, exc_info=True)
                results[name] = {}
        t4 = pc()

        # Track timing
        if self._enable_timing:
            elapsed_ms = (t4 - t0) * 1000
            self._compute_times.append(elapsed_ms)
            self._total_computes += 1

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flush_timestamp(self, ts: int) -> None:
        """Flush staged data for *ts* without computing factors.

        Used when factor values come from cache but the buffer must
        stay consistent for potential fallback computation on later
        timestamps that miss the cache.
        """
        self._buffer.flush_timestamp(ts)

    @property
    def factor_names(self) -> list[str]:
        """Return list of registered factor names."""
        return list(self._factors.keys())

    @property
    def variable_names(self) -> list[str]:
        """Return list of registered variable names."""
        return list(self._variable_order)

    @property
    def instruments(self) -> list[str]:
        """Return list of known instrument IDs."""
        return self._buffer.instruments

    def set_extra_fields(self, fields: list[str]) -> None:
        """Set extra bar fields to track beyond OHLCV."""
        self._buffer = Buffer(
            max_history=self._max_history,
            extra_fields=tuple(fields),
        )

    def get_performance_stats(self) -> dict[str, float]:
        """Return performance statistics."""
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
        """Reset all engine state."""
        self._buffer.reset()
        self._compute_times.clear()
        self._total_computes = 0
