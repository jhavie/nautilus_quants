# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
PanelEvaluator — AST evaluator for panel (matrix) factor computation.

Evaluates Alpha101-style AST expressions where every intermediate result
is a ``pd.DataFrame[T x N]`` (rows=timestamps, cols=instruments) or a scalar.

This enables natural handling of nested CS/TS operators:
    correlation(rank(open), rank(volume), 10)
    → rank operates row-wise (CS), correlation operates column-wise (TS)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from nautilus_quants.factors.expression.ast import (
    ASTNode,
    ASTVisitor,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    StringNode,
    TernaryNode,
    UnaryOpNode,
    VariableNode,
)
from nautilus_quants.factors.operators.base import (
    CrossSectionalOperator,
    TimeSeriesOperator,
)


class PanelEvaluationError(Exception):
    """Raised when panel expression evaluation fails."""

    pass


# Operators that take two data series: op(x, y, window).
_TWO_DATA_OPS: frozenset[str] | None = None


def _get_two_data_ops() -> frozenset[str]:
    """Lazy-load two-data operator names from TS_OPERATOR_INSTANCES."""
    global _TWO_DATA_OPS
    if _TWO_DATA_OPS is None:
        from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

        _TWO_DATA_OPS = frozenset(
            name for name, op in TS_OPERATOR_INSTANCES.items() if op.min_args >= 3
        )
    return _TWO_DATA_OPS


class PanelEvaluator(ASTVisitor):
    """Evaluate AST nodes over panel DataFrames.

    Variables are bound to ``pd.DataFrame[T x N]`` (full panel per field).
    Function calls dispatch to operator ``compute_panel()`` methods:

    - **TS operators** → column-wise (rolling along time axis)
    - **CS operators** → row-wise (across instruments at each timestamp)
    - **Math operators** → element-wise (numpy broadcasting on DataFrames)

    Parameters
    ----------
    panel_fields : dict[str, pd.DataFrame | float]
        Mapping of variable name → panel DataFrame (or scalar parameter).
    ts_ops : dict[str, TimeSeriesOperator]
        Time-series operator instances (name → instance).
    cs_ops : dict[str, CrossSectionalOperator]
        Cross-sectional operator instances (name → instance).
    math_ops : dict[str, Callable]
        Math operator callables (name → function).
    parameters : dict[str, Any]
        Config parameters accessible as variables.
    """

    def __init__(
        self,
        panel_fields: dict[str, pd.DataFrame | float],
        ts_ops: dict[str, TimeSeriesOperator],
        cs_ops: dict[str, CrossSectionalOperator],
        math_ops: dict[str, Callable[..., Any]],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._fields = panel_fields
        self._ts_ops = ts_ops
        self._cs_ops = cs_ops
        self._math_ops = math_ops
        self._parameters = parameters or {}

    def evaluate(self, node: ASTNode) -> pd.DataFrame | float:
        """Evaluate an AST node, returning a panel DataFrame or scalar."""
        return node.accept(self)

    # ------------------------------------------------------------------
    # Visitor methods
    # ------------------------------------------------------------------

    def visit_number(self, node: NumberNode) -> float:
        return node.value

    def visit_string(self, node: StringNode) -> str:
        return node.value  # type: ignore[return-value]

    # Boolean literals recognised in Alpha101-style expressions
    _BOOL_LITERALS: dict[str, float] = {"true": 1.0, "false": 0.0}

    def visit_variable(self, node: VariableNode) -> pd.DataFrame | float:
        name = node.name
        if name in self._fields:
            return self._fields[name]
        if name in self._parameters:
            return self._parameters[name]
        if name in self._BOOL_LITERALS:
            return self._BOOL_LITERALS[name]
        raise PanelEvaluationError(f"Unknown variable: '{name}'")

    def visit_binary_op(self, node: BinaryOpNode) -> pd.DataFrame | float:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        op = node.operator

        # Arithmetic
        if op == "+":
            return self._add(left, right)
        if op == "-":
            return self._sub(left, right)
        if op == "*":
            return self._mul(left, right)
        if op == "/":
            return self._div(left, right)
        if op == "^":
            return self._pow(left, right)

        # Comparison → returns float (1.0 / 0.0)
        if op == ">":
            return self._cmp(left, right, op)
        if op == "<":
            return self._cmp(left, right, op)
        if op == ">=":
            return self._cmp(left, right, op)
        if op == "<=":
            return self._cmp(left, right, op)
        if op == "==":
            return self._cmp(left, right, op)
        if op == "!=":
            return self._cmp(left, right, op)

        # Logical
        if op == "&&":
            return self._logical_and(left, right)
        if op == "||":
            return self._logical_or(left, right)

        raise PanelEvaluationError(f"Unknown binary operator: '{op}'")

    def visit_unary_op(self, node: UnaryOpNode) -> pd.DataFrame | float:
        operand = self.evaluate(node.operand)
        if node.operator == "-":
            return -operand  # type: ignore[operator]
        if node.operator == "!":
            if isinstance(operand, pd.DataFrame):
                return (~operand.astype(bool)).astype(float)
            return float(not bool(operand))
        raise PanelEvaluationError(f"Unknown unary operator: '{node.operator}'")

    def visit_ternary(self, node: TernaryNode) -> pd.DataFrame | float:
        condition = self.evaluate(node.condition)
        true_val = self.evaluate(node.true_expr)
        false_val = self.evaluate(node.false_expr)

        if isinstance(condition, pd.DataFrame):
            cond_arr = condition.astype(bool).values
            t_arr = true_val.values if isinstance(true_val, pd.DataFrame) else true_val
            f_arr = (
                false_val.values if isinstance(false_val, pd.DataFrame) else false_val
            )
            result = np.where(cond_arr, t_arr, f_arr)
            return pd.DataFrame(
                result, index=condition.index, columns=condition.columns
            )

        # Scalar condition: short-circuit
        return true_val if bool(condition) else false_val

    def visit_function_call(self, node: FunctionCallNode) -> pd.DataFrame | float:
        func_name = node.name

        # ------ Cross-sectional operators ------
        if func_name in self._cs_ops:
            op_instance = self._cs_ops[func_name]
            args = [self.evaluate(arg) for arg in node.arguments]
            data = args[0]
            extra_args = args[1:]

            # CS operators take (data, *extra_positional) → e.g. rank(x), normalize(x, useStd, limit)
            if isinstance(data, pd.DataFrame):
                # Unpack remaining args as keyword-compatible positional args
                return op_instance.compute_panel(data, *extra_args)
            # Scalar fallback
            return data

        # ------ Time-series operators ------
        if func_name in self._ts_ops:
            op_instance = self._ts_ops[func_name]
            args = [self.evaluate(arg) for arg in node.arguments]
            data = args[0]
            extra_kwargs: dict[str, Any] = {}

            two_data_ops = _get_two_data_ops()
            if func_name in two_data_ops and len(args) > 2:
                # correlation(x, y, window) → data=x, data2=y, window=window
                extra_kwargs["data2"] = args[1]
                window = int(args[2]) if not isinstance(args[2], str) else 1
            elif len(args) > 1 and isinstance(args[1], (pd.DataFrame, pd.Series)):
                # Element-wise operation: ts_min(df, df) → min(df, df)
                _TS_ELEMENTWISE = {"ts_min": "min", "ts_max": "max"}
                math_name = _TS_ELEMENTWISE.get(func_name)
                if math_name and math_name in self._math_ops:
                    result = self._math_ops[math_name](*args)
                    if isinstance(result, np.ndarray) and not isinstance(
                        result, (pd.DataFrame, pd.Series)
                    ):
                        for a in args:
                            if isinstance(a, pd.DataFrame):
                                return pd.DataFrame(
                                    result, index=a.index, columns=a.columns
                                )
                    return result
                raise PanelEvaluationError(
                    f"TS operator '{func_name}' received non-scalar window"
                )
            else:
                window = int(args[1]) if len(args) > 1 and not isinstance(args[1], str) else 1

            if isinstance(data, pd.DataFrame):
                return op_instance.compute_panel(data, window, **extra_kwargs)
            # Scalar data with TS operator (degenerate case)
            return data

        # ------ Math operators ------
        if func_name in self._math_ops:
            math_fn = self._math_ops[func_name]
            args = [self.evaluate(arg) for arg in node.arguments]
            result = math_fn(*args)
            # Ensure result preserves DataFrame type when input was DataFrame
            if isinstance(result, np.ndarray) and not isinstance(
                result, (pd.DataFrame, pd.Series)
            ):
                for a in args:
                    if isinstance(a, pd.DataFrame):
                        return pd.DataFrame(
                            result, index=a.index, columns=a.columns
                        )
            return result

        raise PanelEvaluationError(f"Unknown operator: '{func_name}'")

    # ------------------------------------------------------------------
    # Arithmetic helpers (handle DataFrame/scalar mixed ops)
    # ------------------------------------------------------------------

    @staticmethod
    def _add(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        return left + right  # type: ignore[operator]

    @staticmethod
    def _sub(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        return left - right  # type: ignore[operator]

    @staticmethod
    def _mul(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        return left * right  # type: ignore[operator]

    @staticmethod
    def _div(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        if isinstance(right, pd.DataFrame):
            return left / right.replace(0, np.nan)  # type: ignore[operator]
        if isinstance(right, (int, float)):
            if right == 0:
                if isinstance(left, pd.DataFrame):
                    return pd.DataFrame(
                        np.nan, index=left.index, columns=left.columns
                    )
                return float("nan")
        return left / right  # type: ignore[operator]

    @staticmethod
    def _pow(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            return np.power(left, right)  # type: ignore[arg-type]
        return np.power(left, right)

    @staticmethod
    def _cmp(
        left: pd.DataFrame | float, right: pd.DataFrame | float, op: str
    ) -> pd.DataFrame | float:
        if op == ">":
            r = left > right  # type: ignore[operator]
        elif op == "<":
            r = left < right  # type: ignore[operator]
        elif op == ">=":
            r = left >= right  # type: ignore[operator]
        elif op == "<=":
            r = left <= right  # type: ignore[operator]
        elif op == "==":
            r = left == right  # type: ignore[operator]
        elif op == "!=":
            r = left != right  # type: ignore[operator]
        else:
            raise PanelEvaluationError(f"Unknown comparison: '{op}'")

        if isinstance(r, pd.DataFrame):
            return r.astype(float)
        return float(r)

    @staticmethod
    def _logical_and(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            l_bool = left.astype(bool) if isinstance(left, pd.DataFrame) else bool(left)
            r_bool = (
                right.astype(bool) if isinstance(right, pd.DataFrame) else bool(right)
            )
            return (l_bool & r_bool).astype(float)  # type: ignore[union-attr]
        return float(bool(left) and bool(right))

    @staticmethod
    def _logical_or(
        left: pd.DataFrame | float, right: pd.DataFrame | float
    ) -> pd.DataFrame | float:
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            l_bool = left.astype(bool) if isinstance(left, pd.DataFrame) else bool(left)
            r_bool = (
                right.astype(bool) if isinstance(right, pd.DataFrame) else bool(right)
            )
            return (l_bool | r_bool).astype(float)  # type: ignore[union-attr]
        return float(bool(left) or bool(right))
