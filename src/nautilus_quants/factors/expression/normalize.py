# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Expression normalization, hashing, and template extraction.

Provides tools for:
- Canonical serialization (for dedup hashing)
- Expression template extraction (for parameter isolation)

Zero ``alpha/`` dependency — lives in ``factors/expression/``.
"""

from __future__ import annotations

import hashlib

from nautilus_quants.factors.expression.ast import (
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    StringNode,
    TernaryNode,
    UnaryOpNode,
    VariableNode,
)
from nautilus_quants.factors.expression.parser import parse_expression

# ── Operator precedence (higher = binds tighter) ──────────────────────────

_PREC: dict[str, int] = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5,
    "^": 6,
}

_RIGHT_ASSOC_OPS = frozenset({"-", "/"})


# ── Internals ─────────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    """Format a number for canonical output."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:g}"


def _normalize(node: ASTNode) -> ASTNode:
    """Normalize AST for canonical comparison.

    Rules:
      - ``UnaryOp('-', NumberNode(v))`` → ``NumberNode(-v)``
      - ``UnaryOp('-', other)``         → ``BinaryOp('*', NumberNode(-1), other)``
    """
    if isinstance(node, (NumberNode, StringNode, VariableNode)):
        return node

    if isinstance(node, UnaryOpNode):
        operand = _normalize(node.operand)
        if node.operator == "-":
            if isinstance(operand, NumberNode):
                return NumberNode(-operand.value)
            return BinaryOpNode("*", NumberNode(-1.0), operand)
        return UnaryOpNode(node.operator, operand)

    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.operator,
            _normalize(node.left),
            _normalize(node.right),
        )

    if isinstance(node, TernaryNode):
        return TernaryNode(
            _normalize(node.condition),
            _normalize(node.true_expr),
            _normalize(node.false_expr),
        )

    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            node.name,
            tuple(_normalize(arg) for arg in node.arguments),
        )

    return node  # pragma: no cover


def _serialize(node: ASTNode, parent_prec: int = 0) -> str:
    """Serialize normalized AST to canonical string with minimal parens."""
    if isinstance(node, NumberNode):
        return _fmt(node.value)

    if isinstance(node, StringNode):
        return f'"{node.value}"'

    if isinstance(node, VariableNode):
        return node.name

    if isinstance(node, BinaryOpNode):
        prec = _PREC.get(node.operator, 0)
        left = _serialize(node.left, prec)
        right_prec = prec + 1 if node.operator in _RIGHT_ASSOC_OPS else prec
        right = _serialize(node.right, right_prec)
        result = f"{left} {node.operator} {right}"
        if prec < parent_prec:
            result = f"({result})"
        return result

    if isinstance(node, UnaryOpNode):
        operand = _serialize(node.operand, 7)
        return f"{node.operator}{operand}"

    if isinstance(node, TernaryNode):
        c = _serialize(node.condition, 0)
        t = _serialize(node.true_expr, 0)
        f = _serialize(node.false_expr, 0)
        result = f"{c} ? {t} : {f}"
        if parent_prec > 0:
            result = f"({result})"
        return result

    if isinstance(node, FunctionCallNode):
        args = ", ".join(_serialize(a, 0) for a in node.arguments)
        return f"{node.name}({args})"

    raise TypeError(f"Unknown AST node: {type(node)}")  # pragma: no cover


def _templatize(
    node: ASTNode,
    values: list[float],
    counter: list[int],
    parent_prec: int = 0,
) -> str:
    """Like ``_serialize`` but replaces ``NumberNode`` with ``p0``, ``p1``, …"""
    if isinstance(node, NumberNode):
        idx = counter[0]
        counter[0] += 1
        values.append(node.value)
        return f"p{idx}"

    if isinstance(node, StringNode):
        return f'"{node.value}"'

    if isinstance(node, VariableNode):
        return node.name

    if isinstance(node, BinaryOpNode):
        prec = _PREC.get(node.operator, 0)
        left = _templatize(node.left, values, counter, prec)
        right_prec = prec + 1 if node.operator in _RIGHT_ASSOC_OPS else prec
        right = _templatize(node.right, values, counter, right_prec)
        result = f"{left} {node.operator} {right}"
        if prec < parent_prec:
            result = f"({result})"
        return result

    if isinstance(node, UnaryOpNode):
        operand = _templatize(node.operand, values, counter, 7)
        return f"{node.operator}{operand}"

    if isinstance(node, TernaryNode):
        c = _templatize(node.condition, values, counter, 0)
        t = _templatize(node.true_expr, values, counter, 0)
        f = _templatize(node.false_expr, values, counter, 0)
        result = f"{c} ? {t} : {f}"
        if parent_prec > 0:
            result = f"({result})"
        return result

    if isinstance(node, FunctionCallNode):
        args = ", ".join(
            _templatize(a, values, counter, 0) for a in node.arguments
        )
        return f"{node.name}({args})"

    raise TypeError(f"Unknown AST node: {type(node)}")  # pragma: no cover


# ── Public API ────────────────────────────────────────────────────────────


def normalize_expression(expr: str) -> str:
    """Parse expression to AST, normalize, and serialize to canonical string.

    Normalization rules:
      - ``-expr`` → ``-1 * expr`` (unary minus becomes explicit multiply)
      - ``-N``    → negative number literal
      - Whitespace and redundant parentheses removed
      - Numbers formatted consistently via ``:g`` format
    """
    ast = parse_expression(expr)
    normalized = _normalize(ast)
    return _serialize(normalized)


def expression_hash(expr: str) -> str:
    """Canonical hash of an expression (SHA-256, first 16 hex chars).

    Two expressions with the same hash compute the same value
    (assuming identical variable bindings).
    """
    canonical = normalize_expression(expr)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def expression_template(expr: str) -> tuple[str, list[float]]:
    """Extract all numeric literals as positional parameters.

    Returns ``(template_string, parameter_values)`` where every number
    is replaced by a positional placeholder ``p0``, ``p1``, …

    ``UnaryOp('-', expr)`` is first normalized to
    ``BinaryOp('*', NumberNode(-1), expr)`` so the ``-1`` becomes
    an extractable parameter.

    Examples::

        >>> expression_template("-1 * correlation(high, rank(volume), 5)")
        ('p0 * correlation(high, rank(volume), p1)', [-1.0, 5.0])

        >>> expression_template("-correlation(high, rank(volume), 10)")
        ('p0 * correlation(high, rank(volume), p1)', [-1.0, 10.0])

        >>> expression_template("correlation(high, rank(volume), 5)")
        ('correlation(high, rank(volume), p0)', [5.0])

        >>> expression_template("ts_mean(close, 5) / close")
        ('ts_mean(close, p0) / close', [5.0])
    """
    ast = parse_expression(expr)
    normalized = _normalize(ast)
    values: list[float] = []
    counter = [0]
    template = _templatize(normalized, values, counter)
    return template, values
