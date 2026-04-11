# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for expression complexity analysis."""

from __future__ import annotations

import pytest

from nautilus_quants.factors.expression.complexity import (
    ComplexityConstraints,
    ComplexityMetrics,
    analyze_complexity,
    check_constraints,
)


# ── analyze_complexity parametrized tests ────────────────────────


class TestAnalyzeComplexity:
    """Test analyze_complexity() on various expression patterns."""

    def test_single_variable(self):
        m = analyze_complexity("close")
        assert m.node_count == 1
        assert m.depth == 1
        assert m.func_nesting == 0
        assert m.variables == frozenset({"close"})
        assert m.max_window == 0
        assert m.number_nodes == 0

    def test_simple_function(self):
        m = analyze_complexity("ts_mean(close, 20)")
        assert m.node_count == 3  # func + var + num
        assert m.depth == 2  # func(leaf, leaf)
        assert m.func_nesting == 1
        assert m.variables == frozenset({"close"})
        assert m.max_window == 20

    def test_nested_functions(self):
        m = analyze_complexity("cs_rank(ts_mean(close, 20))")
        assert m.func_nesting == 2
        assert m.variables == frozenset({"close"})
        assert m.max_window == 20

    def test_binary_op(self):
        m = analyze_complexity("close + volume")
        assert m.node_count == 3  # op + var + var
        assert m.depth == 2
        assert m.variables == frozenset({"close", "volume"})

    def test_deep_nesting(self):
        expr = "cs_rank(ts_mean(close, 20) / ts_std(close, 20))"
        m = analyze_complexity(expr)
        assert m.func_nesting >= 1
        assert m.variables == frozenset({"close"})
        assert m.max_window == 20

    def test_correlation_window(self):
        m = analyze_complexity("correlation(high, volume, 5)")
        assert m.max_window == 5
        assert m.variables == frozenset({"high", "volume"})

    def test_large_window(self):
        m = analyze_complexity("ts_mean(close, 1000)")
        assert m.max_window == 1000

    def test_many_variables(self):
        expr = "close + open + high + low + volume + returns"
        m = analyze_complexity(expr)
        assert m.variables == frozenset(
            {"close", "open", "high", "low", "volume", "returns"}
        )
        assert len(m.variables) == 6

    def test_numeric_ratio(self):
        # 3 numbers, 1 variable = 4 nodes + ops
        expr = "1.0 + 2.0 + 3.0 + close"
        m = analyze_complexity(expr)
        assert m.number_nodes == 3
        assert m.numeric_ratio > 0

    def test_ternary(self):
        expr = "close > 0 ? close : 0"
        m = analyze_complexity(expr)
        assert m.depth >= 2
        assert m.variables == frozenset({"close"})

    def test_unary_neg(self):
        m = analyze_complexity("-close")
        assert m.node_count == 2  # unary + var
        assert m.variables == frozenset({"close"})

    def test_char_length(self):
        expr = "ts_mean(close, 20)"
        m = analyze_complexity(expr)
        assert m.char_length == len(expr)

    def test_complex_real_world(self):
        expr = (
            "cs_rank(delta(close, 6) / replace_zero(delay(close, 6)))"
            " * cs_rank(ts_mean(volume, 6) / ts_mean(volume, 42))"
        )
        m = analyze_complexity(expr)
        assert m.func_nesting >= 2
        assert m.variables == frozenset({"close", "volume"})
        assert m.max_window == 42

    def test_ema_window(self):
        m = analyze_complexity("ema(close, 12)")
        assert m.max_window == 12

    def test_two_data_covariance(self):
        m = analyze_complexity("covariance(close, volume, 30)")
        assert m.max_window == 30


# ── check_constraints tests ──────────────────────────────────────


class TestCheckConstraints:
    """Test constraint checking logic."""

    def test_simple_expression_passes(self):
        violations = check_constraints(
            "cs_rank(delta(close, 6))",
            ComplexityConstraints(),
        )
        assert violations == []

    def test_char_length_violation(self):
        long_expr = "cs_rank(" + " + ".join(["close"] * 50) + ")"
        violations = check_constraints(
            long_expr,
            ComplexityConstraints(max_char_length=100),
        )
        assert any("char_length" in v for v in violations)

    def test_depth_violation(self):
        # Deeply nested: f(f(f(f(f(f(f(close)))))))
        expr = "close"
        for func in ["abs", "sign", "abs", "sign", "abs", "sign", "abs"]:
            expr = f"{func}({expr})"
        violations = check_constraints(
            expr,
            ComplexityConstraints(max_depth=4),
        )
        assert any("depth" in v for v in violations)

    def test_func_nesting_violation(self):
        expr = "cs_rank(ts_mean(abs(sign(log(close))), 5))"
        violations = check_constraints(
            expr,
            ComplexityConstraints(max_func_nesting=2),
        )
        assert any("func_nesting" in v for v in violations)

    def test_variables_violation(self):
        expr = "close + open + high + low + volume + returns"
        violations = check_constraints(
            expr,
            ComplexityConstraints(max_variables=3),
        )
        assert any("variables" in v for v in violations)

    def test_window_violation(self):
        violations = check_constraints(
            "ts_mean(close, 1000)",
            ComplexityConstraints(max_window=720),
        )
        assert any("max_window" in v for v in violations)

    def test_numeric_ratio_violation(self):
        # Expression with many numbers: (1+2+3+4+5) / close
        expr = "(1 + 2 + 3 + 4 + 5) / close"
        violations = check_constraints(
            expr,
            ComplexityConstraints(max_numeric_ratio=0.1),
        )
        assert any("numeric_ratio" in v for v in violations)

    def test_multiple_violations(self):
        long_expr = "ts_mean(" + " + ".join(["close"] * 40) + ", 1000)"
        violations = check_constraints(
            long_expr,
            ComplexityConstraints(
                max_char_length=100,
                max_node_count=10,
                max_window=720,
            ),
        )
        assert len(violations) >= 2

    def test_default_constraints_pass_reasonable_expr(self):
        reasonable_exprs = [
            "cs_rank(delta(close, 6))",
            "cs_rank(ts_mean(volume, 6) / ts_mean(volume, 42))",
            "correlation(close, volume, 20)",
            "ts_rank(delta(close, 1) / replace_zero(delay(close, 1)), 42)",
        ]
        constraints = ComplexityConstraints()
        for expr in reasonable_exprs:
            violations = check_constraints(expr, constraints)
            assert violations == [], f"Unexpected violation for {expr}: {violations}"

    def test_node_count_violation(self):
        expr = " + ".join(["close"] * 20)
        violations = check_constraints(
            expr,
            ComplexityConstraints(max_node_count=10),
        )
        assert any("node_count" in v for v in violations)


# ── ComplexityConstraints defaults ───────────────────────────────


class TestComplexityConstraints:
    """Test dataclass defaults and immutability."""

    def test_defaults(self):
        c = ComplexityConstraints()
        assert c.max_char_length == 200
        assert c.max_node_count == 30
        assert c.max_depth == 6
        assert c.max_func_nesting == 4
        assert c.max_variables == 5
        assert c.max_window == 720
        assert c.max_numeric_ratio == 0.3

    def test_frozen(self):
        c = ComplexityConstraints()
        with pytest.raises(AttributeError):
            c.max_depth = 10  # type: ignore[misc]

    def test_custom_values(self):
        c = ComplexityConstraints(max_depth=3, max_window=100)
        assert c.max_depth == 3
        assert c.max_window == 100
        assert c.max_char_length == 200  # other defaults unchanged
