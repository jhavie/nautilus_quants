# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Expression engine for parsing and evaluating Alpha101-style expressions.

This module provides the complete expression processing pipeline:
1. Parse expression string -> AST
2. Evaluate AST with context -> computed value

Example:
    ```python
    from nautilus_quants.factors.expression import parse_expression, Evaluator, EvaluationContext
    
    ast = parse_expression("ts_mean(close, 20) / ts_std(close, 20)")
    context = EvaluationContext(variables={"close": ...}, operators={"ts_mean": ...})
    result = Evaluator(context).evaluate(ast)
    ```
"""

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
from nautilus_quants.factors.expression.evaluator import (
    EvaluationContext,
    EvaluationError,
    Evaluator,
    VectorizedEvaluator,
    evaluate_expression,
)
from nautilus_quants.factors.expression.complexity import (
    ComplexityConstraints,
    ComplexityMetrics,
    analyze_complexity,
    check_constraints,
)
from nautilus_quants.factors.expression.parser import (
    ExpressionParser,
    parse_expression,
)

__all__ = [
    # AST nodes
    "ASTNode",
    "ASTVisitor",
    "NumberNode",
    "StringNode",
    "VariableNode",
    "BinaryOpNode",
    "UnaryOpNode",
    "TernaryNode",
    "FunctionCallNode",
    # Parser
    "ExpressionParser",
    "parse_expression",
    # Evaluator
    "Evaluator",
    "VectorizedEvaluator",
    "EvaluationContext",
    "EvaluationError",
    "evaluate_expression",
    # Complexity analysis
    "ComplexityConstraints",
    "ComplexityMetrics",
    "analyze_complexity",
    "check_constraints",
]
