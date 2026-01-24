# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Math Operators.

Basic mathematical operators for factor expressions including
log, sign, abs, sqrt, and conditional operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nautilus_quants.factors.operators.base import (
    MathOperator,
    register_operator,
)


@register_operator
class Log(MathOperator):
    """Natural logarithm."""
    
    name = "log"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Compute natural logarithm."""
        # Handle negative values
        if isinstance(value, np.ndarray):
            result = np.log(np.where(value > 0, value, np.nan))
            return result
        return float(np.log(value)) if value > 0 else float('nan')


@register_operator
class Sign(MathOperator):
    """Sign function (-1, 0, or 1)."""
    
    name = "sign"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return sign of value."""
        return np.sign(value)


@register_operator
class Abs(MathOperator):
    """Absolute value."""
    
    name = "abs"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return absolute value."""
        return np.abs(value)


@register_operator
class Sqrt(MathOperator):
    """Square root."""
    
    name = "sqrt"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Compute square root."""
        if isinstance(value, np.ndarray):
            return np.sqrt(np.where(value >= 0, value, np.nan))
        return float(np.sqrt(value)) if value >= 0 else float('nan')


@register_operator
class Power(MathOperator):
    """Power function: x^y."""
    
    name = "power"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Compute power. Expects exponent as second positional arg."""
        exponent = kwargs.get('exponent', 2)
        return np.power(value, exponent)


@register_operator
class Max(MathOperator):
    """Maximum of two values."""
    
    name = "max"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return maximum of two values."""
        other = kwargs.get('other', 0)
        return np.maximum(value, other)


@register_operator
class Min(MathOperator):
    """Minimum of two values."""
    
    name = "min"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return minimum of two values."""
        other = kwargs.get('other', 0)
        return np.minimum(value, other)


@register_operator
class Exp(MathOperator):
    """Exponential function."""
    
    name = "exp"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Compute exponential."""
        return np.exp(value)


@register_operator
class Floor(MathOperator):
    """Floor function."""
    
    name = "floor"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return floor of value."""
        return np.floor(value)


@register_operator
class Ceil(MathOperator):
    """Ceiling function."""
    
    name = "ceil"
    min_args = 1
    max_args = 1
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return ceiling of value."""
        return np.ceil(value)


@register_operator
class Round(MathOperator):
    """Round to nearest integer."""
    
    name = "round"
    min_args = 1
    max_args = 2
    
    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Round to specified decimals (default 0)."""
        decimals = kwargs.get('decimals', 0)
        return np.round(value, int(decimals))


# Convenience function wrappers
def log(value: float | np.ndarray) -> float | np.ndarray:
    """Natural logarithm."""
    return Log().compute(value)


def sign(value: float | np.ndarray) -> float | np.ndarray:
    """Sign function."""
    return Sign().compute(value)


def abs_(value: float | np.ndarray) -> float | np.ndarray:
    """Absolute value (named abs_ to avoid conflict with builtin)."""
    return Abs().compute(value)


def sqrt(value: float | np.ndarray) -> float | np.ndarray:
    """Square root."""
    return Sqrt().compute(value)


def power(base: float | np.ndarray, exponent: float) -> float | np.ndarray:
    """Power function."""
    return Power().compute(base, exponent=exponent)


def max_(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Maximum of two values."""
    return Max().compute(a, other=b)


def min_(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Minimum of two values."""
    return Min().compute(a, other=b)


def exp(value: float | np.ndarray) -> float | np.ndarray:
    """Exponential function."""
    return Exp().compute(value)


def floor(value: float | np.ndarray) -> float | np.ndarray:
    """Floor function."""
    return Floor().compute(value)


def ceil(value: float | np.ndarray) -> float | np.ndarray:
    """Ceiling function."""
    return Ceil().compute(value)


def round_(value: float | np.ndarray, decimals: int = 0) -> float | np.ndarray:
    """Round function."""
    return Round().compute(value, decimals=decimals)


# Export all function wrappers
MATH_OPERATORS = {
    "log": log,
    "sign": sign,
    "abs": abs_,
    "sqrt": sqrt,
    "power": power,
    "max": max_,
    "min": min_,
    "exp": exp,
    "floor": floor,
    "ceil": ceil,
    "round": round_,
}
