# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Operator implementations for factor expressions.

This module provides all operators available for use in Alpha101-style
factor expressions, including time-series, cross-sectional, and math operators.
"""

from nautilus_quants.factors.operators.base import (
    CrossSectionalOperator,
    MathOperator,
    Operator,
    OperatorType,
    TimeSeriesOperator,
    get_all_operators,
    get_operator,
    list_operators,
    register_operator,
)

__all__ = [
    # Base classes
    "Operator",
    "TimeSeriesOperator",
    "CrossSectionalOperator",
    "MathOperator",
    "OperatorType",
    # Registry functions
    "register_operator",
    "get_operator",
    "get_all_operators",
    "list_operators",
]
