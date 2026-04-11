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
import pandas as pd

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
        if isinstance(value, pd.DataFrame):
            return np.log(value.where(value > 0))
        if isinstance(value, (np.ndarray, pd.Series)):
            return np.log(np.where(value > 0, value, np.nan))
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
        if isinstance(value, pd.DataFrame):
            return np.sqrt(value.where(value >= 0))
        if isinstance(value, (np.ndarray, pd.Series)):
            return np.sqrt(np.where(value >= 0, value, np.nan))
        return float(np.sqrt(value)) if value >= 0 else float('nan')


@register_operator
class Power(MathOperator):
    """Power function: x^y."""
    
    name = "power"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, exponent: float = 2, **kwargs: Any) -> float | np.ndarray:
        """Compute power."""
        return np.power(value, exponent)


@register_operator
class Max(MathOperator):
    """Maximum of two values."""
    
    name = "max"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, other: float | np.ndarray = 0, **kwargs: Any) -> float | np.ndarray:
        """Return maximum of two values."""
        return np.maximum(value, other)


@register_operator
class Min(MathOperator):
    """Minimum of two values."""
    
    name = "min"
    min_args = 2
    max_args = 2
    
    def compute(self, value: float | np.ndarray, other: float | np.ndarray = 0, **kwargs: Any) -> float | np.ndarray:
        """Return minimum of two values."""
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

    def compute(self, value: float | np.ndarray, decimals: int = 0, **kwargs: Any) -> float | np.ndarray:
        """Round to specified decimals (default 0)."""
        return np.round(value, int(decimals))


@register_operator
class SignedPower(MathOperator):
    """Signed power: sign(x) * |x|^y. Used in Alpha101."""

    name = "signed_power"
    min_args = 2
    max_args = 2

    def compute(self, value: float | np.ndarray, exponent: float = 2, **kwargs: Any) -> float | np.ndarray:
        """Compute sign(x) * abs(x)^y."""
        return np.sign(value) * np.power(np.abs(value), exponent)


@register_operator
class IfElse(MathOperator):
    """BRAIN-compatible if_else(condition, true_val, false_val).

    Equivalent to the ternary operator but as a function call.
    Supports scalar, numpy array, Series, and DataFrame conditions.
    """

    name = "if_else"
    min_args = 3
    max_args = 3

    def compute(self, condition: Any, true_val: Any, false_val: Any, **kwargs: Any) -> float | np.ndarray:
        """Return true_val where condition is truthy, else false_val."""
        if isinstance(condition, pd.DataFrame):
            result = np.where(condition.astype(bool).values,
                            true_val.values if isinstance(true_val, pd.DataFrame) else true_val,
                            false_val.values if isinstance(false_val, pd.DataFrame) else false_val)
            return pd.DataFrame(result, index=condition.index, columns=condition.columns)
        if isinstance(condition, np.ndarray):
            return np.where(condition.astype(bool), true_val, false_val)
        if isinstance(condition, pd.Series):
            return pd.Series(
                np.where(condition.astype(bool), true_val, false_val),
                index=condition.index,
            )
        return true_val if bool(condition) else false_val


@register_operator
class IsNan(MathOperator):
    """BRAIN-compatible is_nan(x): returns 1 if NaN, else 0."""

    name = "is_nan"
    min_args = 1
    max_args = 1

    def compute(self, value: float | np.ndarray, **kwargs: Any) -> float | np.ndarray:
        """Return 1.0 where value is NaN, 0.0 otherwise."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.isna().astype(float)
        if isinstance(value, np.ndarray):
            return np.where(np.isnan(value), 1.0, 0.0)
        return 1.0 if (isinstance(value, float) and np.isnan(value)) else 0.0


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


def signed_power(base: float | np.ndarray, exponent: float) -> float | np.ndarray:
    """Signed power function."""
    return SignedPower().compute(base, exponent)


def if_else(condition: Any, true_val: Any, false_val: Any) -> float | np.ndarray:
    """BRAIN-compatible conditional function."""
    return IfElse().compute(condition, true_val, false_val)


def is_nan(value: float | np.ndarray) -> float | np.ndarray:
    """BRAIN-compatible NaN check."""
    return IsNan().compute(value)


@register_operator
class FillNan(MathOperator):
    """Replace NaN values with a constant.

    Usage in expressions: ``fill_nan(x, 0)``
    """

    name = "fill_nan"
    min_args = 2
    max_args = 2

    def compute(self, value: float | np.ndarray, fill_value: float = 0.0, **kwargs: Any) -> float | np.ndarray:
        """Replace NaN with *fill_value*."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.fillna(fill_value)
        if isinstance(value, np.ndarray):
            result = value.copy()
            result[np.isnan(result)] = fill_value
            return result
        return fill_value if (isinstance(value, float) and np.isnan(value)) else value


def fill_nan(value: float | np.ndarray, fill_value: float = 0.0) -> float | np.ndarray:
    """Replace NaN with a constant."""
    return FillNan().compute(value, fill_value)


@register_operator
class ReplaceZero(MathOperator):
    """Replace exact zero values with epsilon.

    Popbo-aligned zero-division protection: only exact zeros are replaced,
    non-zero values are unchanged (unlike ``+ eps`` which shifts all values).

    Usage in expressions: ``replace_zero(close - low, 0.0001)``
    """

    name = "replace_zero"
    min_args = 2
    max_args = 2

    def compute(self, value: float | np.ndarray, eps: float = 0.0001, **kwargs: Any) -> float | np.ndarray:
        """Replace exact zero values with *eps*."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.replace(0, eps)
        if isinstance(value, np.ndarray):
            result = value.copy()
            result[result == 0] = eps
            return result
        return eps if value == 0 else value


def replace_zero(value: float | np.ndarray, eps: float = 0.0001) -> float | np.ndarray:
    """Replace exact zeros with epsilon (popbo-aligned)."""
    return ReplaceZero().compute(value, eps)


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
    "signed_power": signed_power,
    "if_else": if_else,
    "is_nan": is_nan,
    "replace_zero": replace_zero,
    "fill_nan": fill_nan,
}
