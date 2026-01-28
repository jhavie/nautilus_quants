# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Cross-Sectional Factor Engine.

Automatically detects and computes cross-sectional factors.
Implements Phase 2 of the two-phase factor computation architecture.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from nautilus_quants.factors.operators.cross_sectional import CROSS_SECTIONAL_OPERATORS
from nautilus_quants.factors.operators.time_series import TIME_SERIES_OPERATORS

if TYPE_CHECKING:
    from nautilus_quants.factors.config import FactorConfig, FactorDefinition


class CsFactorEngine:
    """
    Engine for cross-sectional factor computation.

    Automatically detects which factors are cross-sectional based on
    their expressions (presence of cross-sectional operators like
    normalize, winsorize, cs_rank, etc.).

    The engine implements the second phase of a two-phase architecture:
    - Phase 1: Time-series factors computed per instrument (FactorEngine)
    - Phase 2: Cross-sectional factors computed across instruments (this engine)
    """

    def __init__(self, config: FactorConfig | None = None) -> None:
        """
        Initialize the cross-sectional factor engine.

        Args:
            config: Factor configuration containing factor definitions.
        """
        self._all_factors: list[FactorDefinition] = []
        self._cs_factors: list[FactorDefinition] = []
        self._ts_factor_names: set[str] = set()

        # Get operator names from existing modules (no separate list)
        self._cs_operator_names = set(CROSS_SECTIONAL_OPERATORS.keys())
        self._ts_operator_names = set(TIME_SERIES_OPERATORS.keys())
        self._operators = CROSS_SECTIONAL_OPERATORS

        if config:
            self._load_and_classify(config)

    def _load_and_classify(self, config: FactorConfig) -> None:
        """Load factors and classify into time-series vs cross-sectional."""
        self._all_factors = list(config.factors)

        # Pass 1: Detect by operators in expression
        cs_names: set[str] = set()
        ts_names: set[str] = set()

        for factor in self._all_factors:
            if self._has_cs_operator(factor.expression):
                cs_names.add(factor.name)
            else:
                ts_names.add(factor.name)

        # Pass 2: Propagate cross-sectional property
        # If a factor references a cross-sectional factor, it's also cross-sectional
        changed = True
        while changed:
            changed = False
            for factor in self._all_factors:
                if factor.name in ts_names:
                    refs = self._extract_references(factor.expression)
                    if any(ref in cs_names for ref in refs):
                        ts_names.remove(factor.name)
                        cs_names.add(factor.name)
                        changed = True

        self._ts_factor_names = ts_names

        # Topological sort cross-sectional factors by dependency
        self._cs_factors = self._topological_sort(
            [f for f in self._all_factors if f.name in cs_names],
            cs_names,
        )

    def _has_cs_operator(self, expression: str) -> bool:
        """Check if expression contains cross-sectional operators."""
        pattern = r'\b(\w+)\s*\('
        matches = re.findall(pattern, expression)
        return any(m in self._cs_operator_names for m in matches)

    def _extract_references(self, expression: str) -> set[str]:
        """Extract variable references from expression."""
        # Match identifiers that are not function calls
        pattern = r'\b([a-zA-Z_]\w*)\b(?!\s*\()'
        matches = set(re.findall(pattern, expression))

        # Remove known non-factor names
        exclude = {
            'true', 'false', 'True', 'False',
            'and', 'or', 'not',
            'open', 'high', 'low', 'close', 'volume',
        }
        return matches - exclude

    def _topological_sort(
        self,
        factors: list[FactorDefinition],
        cs_names: set[str],
    ) -> list[FactorDefinition]:
        """Sort factors by dependency order."""
        name_to_factor = {f.name: f for f in factors}
        sorted_factors: list[FactorDefinition] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name not in name_to_factor:
                return
            factor = name_to_factor[name]
            # Visit dependencies first
            for ref in self._extract_references(factor.expression):
                if ref in cs_names:
                    visit(ref)
            visited.add(name)
            sorted_factors.append(factor)

        for factor in factors:
            visit(factor.name)

        return sorted_factors

    def compute(
        self,
        ts_factor_values: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Compute all cross-sectional factors.

        Args:
            ts_factor_values: Time-series factor values from Phase 1.
                Format: {factor_name: {instrument_id: value}}

        Returns:
            Cross-sectional factor values.
                Format: {factor_name: {instrument_id: value}}
        """
        all_values = dict(ts_factor_values)
        cs_results: dict[str, dict[str, float]] = {}

        for factor_def in self._cs_factors:
            result = self._evaluate_expression(
                factor_def.expression,
                all_values,
            )
            cs_results[factor_def.name] = result
            all_values[factor_def.name] = result

        return cs_results

    def _evaluate_expression(
        self,
        expression: str,
        factor_values: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Evaluate a cross-sectional expression."""
        expression = expression.strip()

        # Handle weighted sum (e.g., "0.6 * a + 0.4 * b")
        if self._is_weighted_sum(expression):
            return self._eval_weighted_sum(expression, factor_values)

        # Handle function call
        if "(" in expression:
            return self._eval_function(expression, factor_values)

        # Handle simple variable reference
        return factor_values.get(expression, {})

    def _is_weighted_sum(self, expr: str) -> bool:
        """Check if expression is a weighted sum like '0.6 * a + 0.4 * b'."""
        depth = 0
        for char in expr:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char in "+-" and depth == 0:
                return True
        return False

    def _eval_weighted_sum(
        self,
        expression: str,
        factor_values: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Evaluate weighted sum expression."""
        # Pattern: coefficient * factor_name
        pattern = r'([+-]?\s*[\d.]+)\s*\*\s*(\w+)'
        matches = re.findall(pattern, expression)

        if not matches:
            return {}

        # Get all instrument IDs
        all_instruments: set[str] = set()
        for _, factor_name in matches:
            if factor_name in factor_values:
                all_instruments.update(factor_values[factor_name].keys())

        result: dict[str, float] = {}
        for inst_id in all_instruments:
            total = 0.0
            valid = True
            for weight_str, factor_name in matches:
                weight = float(weight_str.replace(" ", ""))
                values = factor_values.get(factor_name, {})
                value = values.get(inst_id, float('nan'))
                if np.isnan(value):
                    valid = False
                    break
                total += weight * value
            result[inst_id] = total if valid else float('nan')

        return result

    def _eval_function(
        self,
        expression: str,
        factor_values: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Evaluate function call expression."""
        match = re.match(r'(\w+)\s*\((.*)\)', expression, re.DOTALL)
        if not match:
            return {}

        func_name = match.group(1)
        args_str = match.group(2)

        # Get operator (try direct name and cs_ prefix)
        operator = self._operators.get(func_name)
        if operator is None:
            operator = self._operators.get(f"cs_{func_name}")

        if operator is None:
            return {}

        args = self._parse_args(args_str, factor_values)
        if not args or not isinstance(args[0], dict):
            return {}

        return operator(args[0], *args[1:])

    def _parse_args(
        self,
        args_str: str,
        factor_values: dict[str, dict[str, float]],
    ) -> list[Any]:
        """Parse function arguments."""
        args: list[Any] = []
        depth = 0
        current = ""

        for char in args_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                args.append(self._parse_single_arg(current.strip(), factor_values))
                current = ""
            else:
                current += char

        if current.strip():
            args.append(self._parse_single_arg(current.strip(), factor_values))

        return args

    def _parse_single_arg(
        self,
        arg: str,
        factor_values: dict[str, dict[str, float]],
    ) -> Any:
        """Parse a single argument."""
        if arg.lower() == "true":
            return True
        if arg.lower() == "false":
            return False

        try:
            return float(arg) if "." in arg else int(arg)
        except ValueError:
            pass

        if "(" in arg:
            return self._eval_function(arg, factor_values)

        return factor_values.get(arg, {})

    @property
    def ts_factor_names(self) -> list[str]:
        """Get time-series factor names (computed by FactorEngine)."""
        return list(self._ts_factor_names)

    @property
    def cs_factor_names(self) -> list[str]:
        """Get cross-sectional factor names (computed by this engine)."""
        return [f.name for f in self._cs_factors]
