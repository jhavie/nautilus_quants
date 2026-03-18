# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
AST Node Definitions for Factor Expressions.

This module defines the Abstract Syntax Tree nodes produced by parsing
Alpha101-style factor expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ASTNode(ABC):
    """Abstract base class for all AST nodes."""
    
    @abstractmethod
    def accept(self, visitor: ASTVisitor) -> Any:
        """Accept a visitor for tree traversal."""
        pass


class ASTVisitor(ABC):
    """Abstract visitor for AST traversal."""
    
    @abstractmethod
    def visit_number(self, node: NumberNode) -> Any:
        pass
    
    @abstractmethod
    def visit_string(self, node: StringNode) -> Any:
        pass
    
    @abstractmethod
    def visit_variable(self, node: VariableNode) -> Any:
        pass
    
    @abstractmethod
    def visit_binary_op(self, node: BinaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_unary_op(self, node: UnaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_ternary(self, node: TernaryNode) -> Any:
        pass
    
    @abstractmethod
    def visit_function_call(self, node: FunctionCallNode) -> Any:
        pass


@dataclass(frozen=True)
class NumberNode(ASTNode):
    """Numeric literal node."""
    value: float
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_number(self)
    
    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass(frozen=True)
class StringNode(ASTNode):
    """String literal node."""
    value: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string(self)
    
    def __repr__(self) -> str:
        return f"String({self.value!r})"


@dataclass(frozen=True)
class VariableNode(ASTNode):
    """Variable reference node (e.g., 'close', 'volume', 'returns')."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_variable(self)
    
    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    """Binary operation node (e.g., a + b, x * y, p > q)."""
    operator: str
    left: ASTNode
    right: ASTNode
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_op(self)
    
    def __repr__(self) -> str:
        return f"BinOp({self.operator}, {self.left}, {self.right})"


@dataclass(frozen=True)
class UnaryOpNode(ASTNode):
    """Unary operation node (e.g., -x, !condition)."""
    operator: str
    operand: ASTNode
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_op(self)
    
    def __repr__(self) -> str:
        return f"UnaryOp({self.operator}, {self.operand})"


@dataclass(frozen=True)
class TernaryNode(ASTNode):
    """Ternary conditional node (condition ? true_expr : false_expr)."""
    condition: ASTNode
    true_expr: ASTNode
    false_expr: ASTNode
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ternary(self)
    
    def __repr__(self) -> str:
        return f"Ternary({self.condition}, {self.true_expr}, {self.false_expr})"


@dataclass(frozen=True)
class FunctionCallNode(ASTNode):
    """Function call node (e.g., ts_mean(close, 20))."""
    name: str
    arguments: tuple[ASTNode, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_call(self)
    
    def __repr__(self) -> str:
        args_str = ", ".join(repr(arg) for arg in self.arguments)
        return f"Call({self.name}, [{args_str}])"
