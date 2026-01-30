# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Expression Parser for Factor Framework.

This module implements the parser that converts Alpha101-style expression
strings into Abstract Syntax Trees (AST).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args

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


# Path to grammar file
GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


@v_args(inline=True)
class ASTTransformer(Transformer[Any, ASTNode]):
    """Transforms Lark parse tree into AST nodes."""
    
    def number(self, token: Any) -> NumberNode:
        """Transform number token."""
        return NumberNode(float(token))
    
    def string(self, token: Any) -> StringNode:
        """Transform string token (strip quotes)."""
        value = str(token)
        # Remove surrounding quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return StringNode(value)
    
    def variable(self, token: Any) -> VariableNode:
        """Transform identifier to variable reference."""
        return VariableNode(str(token))
    
    def function_call(self, name: Any, *args: Any) -> FunctionCallNode:
        """Transform function call."""
        # Handle case with no arguments
        if len(args) == 0:
            return FunctionCallNode(str(name), ())
        # Handle case with arguments node
        if len(args) == 1 and isinstance(args[0], tuple):
            return FunctionCallNode(str(name), args[0])
        return FunctionCallNode(str(name), tuple(args))
    
    def arguments(self, *args: ASTNode) -> tuple[ASTNode, ...]:
        """Collect function arguments."""
        return tuple(args)
    
    # Binary operations
    def add_expr(self, *args: Any) -> ASTNode:
        """Handle addition/subtraction chain."""
        return self._build_binary_chain(args)
    
    def mul_expr(self, *args: Any) -> ASTNode:
        """Handle multiplication/division chain."""
        return self._build_binary_chain(args)
    
    def pow_expr(self, left: ASTNode, right: ASTNode | None = None) -> ASTNode:
        """Handle power operation."""
        if right is None:
            return left
        return BinaryOpNode("^", left, right)
    
    def comparison(self, left: ASTNode, op: Any = None, right: ASTNode | None = None) -> ASTNode:
        """Handle comparison operation."""
        if op is None:
            return left
        return BinaryOpNode(str(op), left, right)
    
    def or_expr(self, *args: ASTNode) -> ASTNode:
        """Handle logical OR chain."""
        return self._build_binary_chain_with_op(args, "||")
    
    def and_expr(self, *args: ASTNode) -> ASTNode:
        """Handle logical AND chain."""
        return self._build_binary_chain_with_op(args, "&&")
    
    def logical_not(self, operand: ASTNode) -> UnaryOpNode:
        """Handle logical NOT."""
        return UnaryOpNode("!", operand)
    
    # Unary operations
    def neg(self, operand: ASTNode) -> UnaryOpNode:
        """Handle negation."""
        return UnaryOpNode("-", operand)
    
    def pos(self, operand: ASTNode) -> ASTNode:
        """Handle positive (just return operand)."""
        return operand
    
    # Ternary operation
    def ternary(self, condition: ASTNode, true_expr: ASTNode | None = None, 
                false_expr: ASTNode | None = None) -> ASTNode:
        """Handle ternary conditional."""
        if true_expr is None:
            return condition
        return TernaryNode(condition, true_expr, false_expr)  # type: ignore
    
    def _build_binary_chain(self, args: tuple[Any, ...]) -> ASTNode:
        """Build left-associative binary operation chain."""
        if len(args) == 0:
            raise ValueError("Empty args in binary chain")
        if len(args) == 1:
            return args[0]
        
        # Check if args are already just expressions (no operators between them)
        # This happens when lark filters out operators in certain cases
        if len(args) == 2 and isinstance(args[0], ASTNode) and isinstance(args[1], ASTNode):
            # This shouldn't happen normally, but handle gracefully
            return args[0]
        
        # args alternates: expr, op, expr, op, expr, ...
        # But with inline=True, we need to check if we have operators
        result = args[0]
        i = 1
        while i < len(args):
            # Check if current arg is an operator (string/Token) or an ASTNode
            if i + 1 < len(args):
                op = str(args[i])
                right = args[i + 1]
                result = BinaryOpNode(op, result, right)
                i += 2
            else:
                # Odd number of args - shouldn't happen with valid input
                break
        return result
    
    def _build_binary_chain_with_op(self, args: tuple[ASTNode, ...], op: str) -> ASTNode:
        """Build left-associative chain with fixed operator."""
        if len(args) == 1:
            return args[0]
        
        result = args[0]
        for arg in args[1:]:
            result = BinaryOpNode(op, result, arg)
        return result


class ExpressionParser:
    """
    Parser for Alpha101-style factor expressions.
    
    Converts expression strings like "ts_mean(close, 20) / ts_std(close, 20)"
    into an Abstract Syntax Tree for evaluation.
    
    Example:
        ```python
        parser = ExpressionParser()
        ast = parser.parse("ts_mean(close, 20)")
        print(ast)  # Call(ts_mean, [Var(close), Number(20.0)])
        ```
    """
    
    def __init__(self) -> None:
        """Initialize the parser with the grammar."""
        with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
            grammar = f.read()
        
        self._lark = Lark(
            grammar,
            parser="lalr",
            transformer=ASTTransformer(),
        )
    
    def parse(self, expression: str) -> ASTNode:
        """
        Parse an expression string into an AST.
        
        Args:
            expression: The expression string to parse
            
        Returns:
            Root AST node
            
        Raises:
            lark.exceptions.LarkError: If parsing fails
        """
        return self._lark.parse(expression)  # type: ignore


# Module-level singleton for convenience
_parser: ExpressionParser | None = None


def parse_expression(expression: str) -> ASTNode:
    """
    Parse an expression string into an AST.
    
    Uses a module-level singleton parser for efficiency.
    
    Args:
        expression: The expression string to parse
        
    Returns:
        Root AST node
    """
    global _parser
    if _parser is None:
        _parser = ExpressionParser()
    return _parser.parse(expression)
