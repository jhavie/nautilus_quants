# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for expression parser."""

import pytest

from nautilus_quants.factors.expression.ast import (
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    TernaryNode,
    UnaryOpNode,
    VariableNode,
)
from nautilus_quants.factors.expression.parser import ExpressionParser, parse_expression


class TestBasicExpressions:
    """Tests for basic expression parsing."""

    def test_parse_number(self):
        """Test parsing numeric literals."""
        ast = parse_expression("42")
        assert isinstance(ast, NumberNode)
        assert ast.value == 42.0

    def test_parse_float(self):
        """Test parsing float literals."""
        ast = parse_expression("3.14159")
        assert isinstance(ast, NumberNode)
        assert ast.value == pytest.approx(3.14159)

    def test_parse_scientific_notation(self):
        """Test parsing scientific notation."""
        ast = parse_expression("1.5e-3")
        assert isinstance(ast, NumberNode)
        assert ast.value == pytest.approx(0.0015)

    def test_parse_variable(self):
        """Test parsing variable references."""
        ast = parse_expression("close")
        assert isinstance(ast, VariableNode)
        assert ast.name == "close"

    def test_parse_underscore_variable(self):
        """Test parsing variable with underscore."""
        ast = parse_expression("adv_20")
        assert isinstance(ast, VariableNode)
        assert ast.name == "adv_20"


class TestArithmeticExpressions:
    """Tests for arithmetic expression parsing."""

    def test_parse_addition(self):
        """Test parsing addition."""
        ast = parse_expression("1 + 2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "+"
        assert isinstance(ast.left, NumberNode)
        assert isinstance(ast.right, NumberNode)

    def test_parse_subtraction(self):
        """Test parsing subtraction."""
        ast = parse_expression("5 - 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "-"

    def test_parse_multiplication(self):
        """Test parsing multiplication."""
        ast = parse_expression("2 * 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "*"

    def test_parse_division(self):
        """Test parsing division."""
        ast = parse_expression("10 / 2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "/"

    def test_parse_power(self):
        """Test parsing power operator."""
        ast = parse_expression("2 ^ 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "^"

    def test_operator_precedence(self):
        """Test operator precedence (multiplication before addition)."""
        ast = parse_expression("1 + 2 * 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "+"
        # Right side should be 2 * 3
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.operator == "*"

    def test_parentheses(self):
        """Test parentheses override precedence."""
        ast = parse_expression("(1 + 2) * 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "*"
        # Left side should be 1 + 2
        assert isinstance(ast.left, BinaryOpNode)
        assert ast.left.operator == "+"


class TestComparisonExpressions:
    """Tests for comparison expression parsing."""

    def test_parse_greater_than(self):
        """Test parsing greater than."""
        ast = parse_expression("close > 100")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == ">"

    def test_parse_less_than(self):
        """Test parsing less than."""
        ast = parse_expression("volume < 1000")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "<"

    def test_parse_equals(self):
        """Test parsing equality."""
        ast = parse_expression("x == y")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "=="

    def test_parse_not_equals(self):
        """Test parsing not equals."""
        ast = parse_expression("a != b")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "!="

    def test_parse_greater_equal(self):
        """Test parsing greater than or equal."""
        ast = parse_expression("x >= 0")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == ">="

    def test_parse_less_equal(self):
        """Test parsing less than or equal."""
        ast = parse_expression("x <= 100")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "<="


class TestUnaryExpressions:
    """Tests for unary expression parsing."""

    def test_parse_negation(self):
        """Test parsing negation."""
        ast = parse_expression("-5")
        assert isinstance(ast, UnaryOpNode)
        assert ast.operator == "-"
        assert isinstance(ast.operand, NumberNode)

    def test_parse_logical_not(self):
        """Test parsing logical not."""
        ast = parse_expression("!condition")
        assert isinstance(ast, UnaryOpNode)
        assert ast.operator == "!"
        assert isinstance(ast.operand, VariableNode)


class TestFunctionCalls:
    """Tests for function call parsing."""

    def test_parse_simple_function(self):
        """Test parsing simple function call."""
        ast = parse_expression("ts_mean(close, 20)")
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "ts_mean"
        assert len(ast.arguments) == 2
        assert isinstance(ast.arguments[0], VariableNode)
        assert isinstance(ast.arguments[1], NumberNode)

    def test_parse_nested_function(self):
        """Test parsing nested function calls."""
        ast = parse_expression("delay(ts_max(close, 30), 1)")
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "delay"
        assert len(ast.arguments) == 2
        # First arg is another function call
        assert isinstance(ast.arguments[0], FunctionCallNode)
        assert ast.arguments[0].name == "ts_max"

    def test_parse_function_with_expression_arg(self):
        """Test function with expression as argument."""
        ast = parse_expression("log(close / open)")
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "log"
        assert len(ast.arguments) == 1
        assert isinstance(ast.arguments[0], BinaryOpNode)


class TestTernaryExpressions:
    """Tests for ternary conditional parsing."""

    def test_parse_ternary(self):
        """Test parsing ternary expression."""
        ast = parse_expression("x > 0 ? 1 : 0")
        assert isinstance(ast, TernaryNode)
        assert isinstance(ast.condition, BinaryOpNode)
        assert isinstance(ast.true_expr, NumberNode)
        assert isinstance(ast.false_expr, NumberNode)

    def test_parse_nested_ternary(self):
        """Test parsing nested ternary (in false branch)."""
        ast = parse_expression("a > b ? 1 : c > d ? 2 : 3")
        assert isinstance(ast, TernaryNode)
        assert isinstance(ast.false_expr, TernaryNode)


class TestComplexExpressions:
    """Tests for complex Alpha101-style expressions."""

    def test_breakout_factor_expression(self):
        """Test parsing breakout factor expression."""
        expr = "(close > highest_close) * (volume > highest_volume) * (close > sma)"
        ast = parse_expression(expr)
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "*"

    def test_alpha_style_expression(self):
        """Test parsing Alpha101-style expression."""
        expr = "ts_rank(ts_argmax(close, 30), 5)"
        ast = parse_expression(expr)
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "ts_rank"

    def test_complex_arithmetic(self):
        """Test complex arithmetic expression."""
        expr = "(close - ts_mean(close, 20)) / ts_std(close, 20)"
        ast = parse_expression(expr)
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "/"


class TestParserInstance:
    """Tests for ExpressionParser class."""

    def test_parser_reusable(self):
        """Test parser can be reused for multiple expressions."""
        parser = ExpressionParser()
        ast1 = parser.parse("1 + 2")
        ast2 = parser.parse("3 * 4")
        assert isinstance(ast1, BinaryOpNode)
        assert isinstance(ast2, BinaryOpNode)
        assert ast1.operator == "+"
        assert ast2.operator == "*"

    def test_parser_error_handling(self):
        """Test parser raises error for invalid expression."""
        parser = ExpressionParser()
        # "1 + + 2" is valid (+ 2 is unary positive), use truly invalid expressions
        with pytest.raises(Exception):  # lark raises UnexpectedToken
            parser.parse("1 +")  # incomplete expression
        with pytest.raises(Exception):
            parser.parse("((1 + 2)")  # unbalanced parentheses
