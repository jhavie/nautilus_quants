# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Expression complexity analysis for anti-overfitting constraints.

Provides AST-level metrics and configurable thresholds to reject
overly complex factor expressions before backtesting.

Inspired by QuantaAlpha's factor regulator (symbol length ≤ 250,
base features ≤ 6, duplicated subtree ≤ 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

# Time-series functions whose second (or third for two-data) arg is a window.
_TS_FUNCS: frozenset[str] = frozenset({
    "ts_mean", "sma", "ts_sum", "ts_std", "stddev", "ts_std_dev",
    "ts_min", "ts_max", "ts_rank", "ts_argmax", "ts_argmin",
    "ts_arg_max", "ts_arg_min", "ts_skew", "ts_product", "product",
    "ts_slope", "ts_rsquare", "ts_residual", "ts_percentile",
    "delta", "ts_delta", "delay", "ts_delay",
    "decay_linear", "ts_decay_linear", "ema",
    "wq_ts_rank", "wq_ts_argmax", "wq_ts_argmin",
})
_TWO_DATA_FUNCS: frozenset[str] = frozenset({
    "correlation", "ts_corr", "covariance", "ts_covariance",
})


@dataclass(frozen=True)
class ComplexityConstraints:
    """Configurable thresholds for expression complexity."""

    max_char_length: int = 200
    max_node_count: int = 30
    max_depth: int = 6
    max_func_nesting: int = 4
    max_variables: int = 5
    max_window: int = 720
    max_numeric_ratio: float = 0.3


@dataclass(frozen=True)
class ComplexityMetrics:
    """Measured complexity of a parsed expression."""

    char_length: int
    node_count: int
    depth: int
    func_nesting: int
    variables: frozenset[str]
    max_window: int
    number_nodes: int
    total_nodes: int

    @property
    def numeric_ratio(self) -> float:
        return self.number_nodes / self.total_nodes if self.total_nodes > 0 else 0.0


# ── internal result carried through the visitor ──────────────────────

@dataclass(frozen=True)
class _Info:
    """Intermediate metrics collected during AST traversal."""

    node_count: int
    depth: int
    func_nesting: int  # max nested FunctionCallNode depth on any path
    variables: frozenset[str]
    max_window: int
    number_nodes: int


def _merge(parent_extra_nodes: int, *children: _Info) -> _Info:
    """Merge child _Info values and add *parent_extra_nodes* for the parent."""
    return _Info(
        node_count=sum(c.node_count for c in children) + parent_extra_nodes,
        depth=max((c.depth for c in children), default=0) + 1,
        func_nesting=max((c.func_nesting for c in children), default=0),
        variables=frozenset().union(*(c.variables for c in children)),
        max_window=max((c.max_window for c in children), default=0),
        number_nodes=sum(c.number_nodes for c in children),
    )


class ComplexityVisitor(ASTVisitor):
    """Single-pass AST visitor that collects all complexity metrics."""

    def __init__(self) -> None:
        self._func_depth: int = 0  # current FunctionCallNode nesting

    # ── leaf nodes ───────────────────────────────────────────────

    def visit_number(self, node: NumberNode) -> _Info:
        return _Info(
            node_count=1, depth=1, func_nesting=0,
            variables=frozenset(), max_window=0, number_nodes=1,
        )

    def visit_string(self, node: StringNode) -> _Info:
        return _Info(
            node_count=1, depth=1, func_nesting=0,
            variables=frozenset(), max_window=0, number_nodes=0,
        )

    def visit_variable(self, node: VariableNode) -> _Info:
        return _Info(
            node_count=1, depth=1, func_nesting=self._func_depth,
            variables=frozenset({node.name}), max_window=0, number_nodes=0,
        )

    # ── composite nodes ──────────────────────────────────────────

    def visit_binary_op(self, node: BinaryOpNode) -> _Info:
        left: _Info = node.left.accept(self)
        right: _Info = node.right.accept(self)
        return _merge(1, left, right)

    def visit_unary_op(self, node: UnaryOpNode) -> _Info:
        child: _Info = node.operand.accept(self)
        return _merge(1, child)

    def visit_ternary(self, node: TernaryNode) -> _Info:
        cond: _Info = node.condition.accept(self)
        t: _Info = node.true_expr.accept(self)
        f: _Info = node.false_expr.accept(self)
        return _merge(1, cond, t, f)

    def visit_function_call(self, node: FunctionCallNode) -> _Info:
        self._func_depth += 1
        children: list[_Info] = [arg.accept(self) for arg in node.arguments]
        self._func_depth -= 1

        merged = _merge(1, *children) if children else _Info(
            node_count=1, depth=1, func_nesting=0,
            variables=frozenset(), max_window=0, number_nodes=0,
        )

        # Detect window parameter for time-series functions.
        window = merged.max_window
        name_lower = node.name.lower()
        if name_lower in _TWO_DATA_FUNCS:
            # correlation(x, y, window) — window is arg[2]
            if len(node.arguments) >= 3 and isinstance(node.arguments[2], NumberNode):
                window = max(window, int(node.arguments[2].value))
        elif name_lower in _TS_FUNCS:
            # ts_mean(x, window) — window is arg[1]
            if len(node.arguments) >= 2 and isinstance(node.arguments[1], NumberNode):
                window = max(window, int(node.arguments[1].value))

        return _Info(
            node_count=merged.node_count,
            depth=merged.depth,
            func_nesting=max(merged.func_nesting, self._func_depth + 1),
            variables=merged.variables,
            max_window=window,
            number_nodes=merged.number_nodes,
        )


# ── public API ───────────────────────────────────────────────────

def analyze_complexity(
    expr_or_ast: str | ASTNode,
    *,
    expr_str: str | None = None,
) -> ComplexityMetrics:
    """Analyze complexity of an expression string or pre-parsed AST.

    Parameters
    ----------
    expr_or_ast
        Either an expression string (will be parsed) or a pre-parsed ASTNode.
    expr_str
        Original expression string for char_length measurement.
        Required when *expr_or_ast* is an ASTNode.
    """
    if isinstance(expr_or_ast, str):
        from nautilus_quants.factors.expression.parser import parse_expression
        expr_str = expr_or_ast
        ast = parse_expression(expr_or_ast)
    else:
        ast = expr_or_ast
        if expr_str is None:
            raise ValueError("expr_str required when passing a pre-parsed ASTNode")

    visitor = ComplexityVisitor()
    info: _Info = ast.accept(visitor)

    return ComplexityMetrics(
        char_length=len(expr_str),
        node_count=info.node_count,
        depth=info.depth,
        func_nesting=info.func_nesting,
        variables=info.variables,
        max_window=info.max_window,
        number_nodes=info.number_nodes,
        total_nodes=info.node_count,
    )


def check_constraints(
    expr_or_ast: str | ASTNode,
    constraints: ComplexityConstraints,
    *,
    expr_str: str | None = None,
) -> list[str]:
    """Check expression against complexity constraints.

    Returns a list of violation descriptions (empty = all passed).
    """
    if isinstance(expr_or_ast, str):
        expr_str = expr_or_ast
    elif expr_str is None:
        raise ValueError("expr_str required when passing a pre-parsed ASTNode")

    violations: list[str] = []

    # Fast pre-check: char length (no parsing needed).
    assert expr_str is not None
    if len(expr_str) > constraints.max_char_length:
        violations.append(
            f"char_length={len(expr_str)} > max {constraints.max_char_length}"
        )

    metrics = analyze_complexity(expr_or_ast, expr_str=expr_str)

    if metrics.node_count > constraints.max_node_count:
        violations.append(
            f"node_count={metrics.node_count} > max {constraints.max_node_count}"
        )
    if metrics.depth > constraints.max_depth:
        violations.append(
            f"depth={metrics.depth} > max {constraints.max_depth}"
        )
    if metrics.func_nesting > constraints.max_func_nesting:
        violations.append(
            f"func_nesting={metrics.func_nesting} > max {constraints.max_func_nesting}"
        )
    if len(metrics.variables) > constraints.max_variables:
        violations.append(
            f"variables={len(metrics.variables)} > max {constraints.max_variables}"
        )
    if metrics.max_window > constraints.max_window:
        violations.append(
            f"max_window={metrics.max_window} > max {constraints.max_window}"
        )
    if metrics.numeric_ratio > constraints.max_numeric_ratio:
        violations.append(
            f"numeric_ratio={metrics.numeric_ratio:.2f} > max {constraints.max_numeric_ratio}"
        )

    return violations
