# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
AST Evaluator for Factor Expressions.

This module implements the evaluator that computes values from
parsed AST nodes using operator implementations and variable contexts.
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


class EvaluationError(Exception):
    """Raised when expression evaluation fails."""
    pass


class EvaluationContext:
    """
    Context for expression evaluation.
    
    Holds variable values and operator functions available during evaluation.
    
    Attributes:
        variables: Dict of variable name -> value (float or np.ndarray)
        operators: Dict of operator name -> callable
        parameters: Dict of parameter name -> value (from config)
    """
    
    def __init__(
        self,
        variables: dict[str, float | np.ndarray] | None = None,
        operators: dict[str, Callable[..., Any]] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.variables = variables or {}
        self.operators = operators or {}
        self.parameters = parameters or {}
    
    def get_variable(self, name: str) -> float | np.ndarray:
        """Get a variable value by name."""
        # First check variables
        if name in self.variables:
            return self.variables[name]
        # Then check parameters
        if name in self.parameters:
            return self.parameters[name]
        raise EvaluationError(f"Unknown variable: '{name}'")
    
    def get_operator(self, name: str) -> Callable[..., Any]:
        """Get an operator function by name."""
        if name not in self.operators:
            raise EvaluationError(f"Unknown operator: '{name}'")
        return self.operators[name]
    
    def set_variable(self, name: str, value: float | np.ndarray) -> None:
        """Set a variable value."""
        self.variables[name] = value
    
    def copy(self) -> EvaluationContext:
        """Create a shallow copy of the context."""
        return EvaluationContext(
            variables=self.variables.copy(),
            operators=self.operators,  # Share operators
            parameters=self.parameters.copy(),
        )


class Evaluator(ASTVisitor):
    """
    Evaluates AST nodes to compute factor values.
    
    Uses the visitor pattern to traverse the AST and compute results.
    Supports scalar and vectorized (numpy array) operations.
    
    Example:
        ```python
        context = EvaluationContext(
            variables={"close": 100.0, "volume": 1000.0},
            operators={"ts_mean": ts_mean_func},
        )
        evaluator = Evaluator(context)
        result = evaluator.evaluate(ast)
        ```
    """
    
    def __init__(self, context: EvaluationContext) -> None:
        """Initialize evaluator with context."""
        self.context = context
    
    def evaluate(self, node: ASTNode) -> float | np.ndarray:
        """
        Evaluate an AST node.
        
        Args:
            node: The AST node to evaluate
            
        Returns:
            Computed value (scalar or array)
        """
        return node.accept(self)
    
    def visit_number(self, node: NumberNode) -> float:
        """Evaluate number literal."""
        return node.value
    
    def visit_string(self, node: StringNode) -> str:
        """Evaluate string literal."""
        return node.value
    
    def visit_variable(self, node: VariableNode) -> float | np.ndarray:
        """Evaluate variable reference."""
        return self.context.get_variable(node.name)
    
    def visit_binary_op(self, node: BinaryOpNode) -> float | np.ndarray:
        """Evaluate binary operation."""
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        op = node.operator
        
        # Arithmetic operations
        if op == "+":
            return left + right  # type: ignore
        elif op == "-":
            return left - right  # type: ignore
        elif op == "*":
            return left * right  # type: ignore
        elif op == "/":
            # Handle division by zero for scalars and arrays
            if isinstance(right, np.ndarray):
                out = np.full_like(right, np.nan, dtype=float)
                mask = right != 0
                np.divide(left, right, out=out, where=mask)
                return out
            if isinstance(right, (int, float)) and right == 0:
                if isinstance(left, np.ndarray):
                    return np.full_like(left, np.nan, dtype=float)
                return float('nan')
            return left / right  # type: ignore
        elif op == "^":
            return np.power(left, right)
        
        # Comparison operations (return 1.0 for true, 0.0 for false)
        elif op == "==":
            return float(left == right) if isinstance(left, (int, float)) else (left == right).astype(float)
        elif op == "!=":
            return float(left != right) if isinstance(left, (int, float)) else (left != right).astype(float)
        elif op == "<":
            return float(left < right) if isinstance(left, (int, float)) else (left < right).astype(float)
        elif op == ">":
            return float(left > right) if isinstance(left, (int, float)) else (left > right).astype(float)
        elif op == "<=":
            return float(left <= right) if isinstance(left, (int, float)) else (left <= right).astype(float)
        elif op == ">=":
            return float(left >= right) if isinstance(left, (int, float)) else (left >= right).astype(float)
        
        # Logical operations
        elif op == "&&":
            left_bool = self._to_bool(left)
            right_bool = self._to_bool(right)
            if isinstance(left_bool, np.ndarray) or isinstance(right_bool, np.ndarray):
                return (np.asarray(left_bool) & np.asarray(right_bool)).astype(float)
            return float(left_bool and right_bool)
        elif op == "||":
            left_bool = self._to_bool(left)
            right_bool = self._to_bool(right)
            if isinstance(left_bool, np.ndarray) or isinstance(right_bool, np.ndarray):
                return (np.asarray(left_bool) | np.asarray(right_bool)).astype(float)
            return float(left_bool or right_bool)
        
        raise EvaluationError(f"Unknown operator: '{op}'")
    
    def visit_unary_op(self, node: UnaryOpNode) -> float | np.ndarray:
        """Evaluate unary operation."""
        operand = self.evaluate(node.operand)
        
        if node.operator == "-":
            return -operand  # type: ignore
        elif node.operator == "!":
            if isinstance(operand, np.ndarray):
                return (~operand.astype(bool)).astype(float)
            return float(not self._to_bool(operand))
        
        raise EvaluationError(f"Unknown unary operator: '{node.operator}'")
    
    def visit_ternary(self, node: TernaryNode) -> float | np.ndarray:
        """Evaluate ternary conditional."""
        condition = self.evaluate(node.condition)
        
        # For numpy arrays, use np.where
        if isinstance(condition, np.ndarray):
            true_val = self.evaluate(node.true_expr)
            false_val = self.evaluate(node.false_expr)
            return np.where(condition.astype(bool), true_val, false_val)
        
        # For scalars, short-circuit evaluation
        if self._to_bool(condition):
            return self.evaluate(node.true_expr)
        else:
            return self.evaluate(node.false_expr)
    
    def visit_function_call(self, node: FunctionCallNode) -> float | np.ndarray:
        """Evaluate function call."""
        # Get the operator function
        func = self.context.get_operator(node.name)
        
        # Evaluate all arguments
        args = [self.evaluate(arg) for arg in node.arguments]
        
        # Call the function
        try:
            return func(*args)
        except Exception as e:
            raise EvaluationError(
                f"Error calling operator '{node.name}': {e}"
            ) from e
    
    def _to_bool(self, value: Any) -> bool | np.ndarray:
        """Convert value to boolean."""
        if isinstance(value, np.ndarray):
            return value.astype(bool)
        return bool(value)


def evaluate_expression(
    node: ASTNode,
    variables: dict[str, float | np.ndarray],
    operators: dict[str, Callable[..., Any]],
    parameters: dict[str, Any] | None = None,
) -> float | np.ndarray:
    """
    Convenience function to evaluate an AST node.
    
    Args:
        node: The AST node to evaluate
        variables: Dict of variable name -> value
        operators: Dict of operator name -> callable
        parameters: Optional config parameters
        
    Returns:
        Computed value
    """
    context = EvaluationContext(
        variables=variables,
        operators=operators,
        parameters=parameters or {},
    )
    evaluator = Evaluator(context)
    return evaluator.evaluate(node)


class VectorizedEvaluator(ASTVisitor):
    """Evaluate AST nodes over full pd.Series time-series.

    Variables are bound to ``pd.Series`` (complete time-series per instrument).
    Function calls dispatch to operator ``compute_vectorized()`` methods.
    Binary operations use element-wise pandas arithmetic.
    """

    def __init__(
        self,
        variables: dict[str, pd.Series],
        ts_operators: dict[str, Any],
        math_operators: dict[str, Callable[..., Any]],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.variables = variables
        self.ts_operators = ts_operators  # name -> operator instance
        self.math_operators = math_operators  # name -> callable (already vectorized)
        self.parameters = parameters or {}

    def evaluate(self, node: ASTNode) -> pd.Series | float:
        return node.accept(self)

    def visit_number(self, node: NumberNode) -> float:
        return node.value

    def visit_string(self, node: StringNode) -> str:
        return node.value

    def visit_variable(self, node: VariableNode) -> pd.Series | float:
        if node.name in self.variables:
            return self.variables[node.name]
        if node.name in self.parameters:
            return self.parameters[node.name]
        raise EvaluationError(f"Unknown variable: '{node.name}'")

    def visit_binary_op(self, node: BinaryOpNode) -> pd.Series | float:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        op = node.operator
        if op == "+":
            return left + right  # type: ignore
        elif op == "-":
            return left - right  # type: ignore
        elif op == "*":
            return left * right  # type: ignore
        elif op == "/":
            if isinstance(right, pd.Series):
                return left.divide(right).where(right != 0, np.nan) if isinstance(left, pd.Series) else pd.Series(
                    np.where(right != 0, left / right, np.nan), index=right.index,
                )
            if isinstance(right, (int, float)) and right == 0:
                if isinstance(left, pd.Series):
                    return pd.Series(np.nan, index=left.index)
                return float("nan")
            return left / right  # type: ignore
        elif op == "^":
            return np.power(left, right)
        elif op == ">":
            return (left > right).astype(float) if isinstance(left, pd.Series) else float(left > right)  # type: ignore
        elif op == "<":
            return (left < right).astype(float) if isinstance(left, pd.Series) else float(left < right)  # type: ignore
        elif op == ">=":
            return (left >= right).astype(float) if isinstance(left, pd.Series) else float(left >= right)  # type: ignore
        elif op == "<=":
            return (left <= right).astype(float) if isinstance(left, pd.Series) else float(left <= right)  # type: ignore
        elif op == "==":
            return (left == right).astype(float) if isinstance(left, pd.Series) else float(left == right)  # type: ignore
        elif op == "!=":
            return (left != right).astype(float) if isinstance(left, pd.Series) else float(left != right)  # type: ignore
        raise EvaluationError(f"Unknown operator: '{op}'")

    def visit_unary_op(self, node: UnaryOpNode) -> pd.Series | float:
        operand = self.evaluate(node.operand)
        if node.operator == "-":
            return -operand  # type: ignore
        elif node.operator == "!":
            if isinstance(operand, pd.Series):
                return (~operand.astype(bool)).astype(float)
            return float(not bool(operand))
        raise EvaluationError(f"Unknown unary operator: '{node.operator}'")

    def visit_ternary(self, node: TernaryNode) -> pd.Series | float:
        condition = self.evaluate(node.condition)
        true_val = self.evaluate(node.true_expr)
        false_val = self.evaluate(node.false_expr)
        if isinstance(condition, pd.Series):
            return pd.Series(
                np.where(condition.astype(bool), true_val, false_val),
                index=condition.index,
            )
        return true_val if bool(condition) else false_val

    def visit_function_call(self, node: FunctionCallNode) -> pd.Series | float:
        func_name = node.name

        # Check time-series operators first
        if func_name in self.ts_operators:
            op_instance = self.ts_operators[func_name]
            args = [self.evaluate(arg) for arg in node.arguments]
            data = args[0]
            extra_kwargs: dict[str, Any] = {}
            if func_name in ("correlation", "covariance") and len(args) > 2:
                # correlation(x, y, window) -> data=x, window=window, data2=y
                extra_kwargs["data2"] = args[1]
                window = int(args[2])
            else:
                window = int(args[1]) if len(args) > 1 else 1
            return op_instance.compute_vectorized(data, window, **extra_kwargs)

        # Check math operators
        if func_name in self.math_operators:
            math_fn = self.math_operators[func_name]
            args = [self.evaluate(arg) for arg in node.arguments]
            result = math_fn(*args)
            # Preserve pd.Series type when input was Series
            if isinstance(result, np.ndarray) and not isinstance(result, pd.Series):
                for a in args:
                    if isinstance(a, pd.Series):
                        return pd.Series(result, index=a.index)
            return result

        raise EvaluationError(f"Unknown operator: '{func_name}'")
