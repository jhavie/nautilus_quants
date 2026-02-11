"""Tests for Math operators receiving positional arguments correctly.

These tests verify that Power, Max, Min, Round work when called with
positional arguments (as the evaluator does via func(*args)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.math import (
    Max,
    Min,
    Power,
    Round,
)
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.expression.evaluator import (
    Evaluator,
    EvaluationContext,
    VectorizedEvaluator,
)
from nautilus_quants.factors.operators.math import MATH_OPERATORS


class TestPowerPositionalArgs:
    """Power.compute must accept exponent as second positional arg."""

    def test_scalar_positional(self):
        op = Power()
        result = op.compute(3.0, 2)
        assert result == pytest.approx(9.0)

    def test_scalar_cube(self):
        op = Power()
        result = op.compute(2.0, 3)
        assert result == pytest.approx(8.0)

    def test_array_positional(self):
        op = Power()
        arr = np.array([2.0, 3.0, 4.0])
        result = op.compute(arr, 2)
        np.testing.assert_allclose(result, [4.0, 9.0, 16.0])

    def test_default_exponent(self):
        op = Power()
        result = op.compute(5.0)
        assert result == pytest.approx(25.0)  # default exponent=2


class TestMaxPositionalArgs:
    """Max.compute must accept 'other' as second positional arg."""

    def test_scalar_positional(self):
        op = Max()
        result = op.compute(3.0, 5.0)
        assert result == pytest.approx(5.0)

    def test_scalar_first_larger(self):
        op = Max()
        result = op.compute(10.0, 5.0)
        assert result == pytest.approx(10.0)

    def test_array_positional(self):
        op = Max()
        arr = np.array([-1.0, 5.0, 3.0])
        result = op.compute(arr, 2.0)
        np.testing.assert_allclose(result, [2.0, 5.0, 3.0])

    def test_default_other(self):
        op = Max()
        result = op.compute(-5.0)
        assert result == pytest.approx(0.0)  # default other=0


class TestMinPositionalArgs:
    """Min.compute must accept 'other' as second positional arg."""

    def test_scalar_positional(self):
        op = Min()
        result = op.compute(3.0, 5.0)
        assert result == pytest.approx(3.0)

    def test_array_positional(self):
        op = Min()
        arr = np.array([1.0, 5.0, 3.0])
        result = op.compute(arr, 2.0)
        np.testing.assert_allclose(result, [1.0, 2.0, 2.0])

    def test_default_other(self):
        op = Min()
        result = op.compute(5.0)
        assert result == pytest.approx(0.0)  # default other=0


class TestRoundPositionalArgs:
    """Round.compute must accept decimals as second positional arg."""

    def test_scalar_positional(self):
        op = Round()
        result = op.compute(3.14159, 2)
        assert result == pytest.approx(3.14)

    def test_default_decimals(self):
        op = Round()
        result = op.compute(3.7)
        assert result == pytest.approx(4.0)


class TestMathOperatorsViaEvaluator:
    """Verify math operators work when called through the evaluator (func(*args))."""

    def _make_evaluator(self, variables=None):
        from nautilus_quants.factors.operators.time_series import TIME_SERIES_OPERATORS
        operators = {**TIME_SERIES_OPERATORS, **MATH_OPERATORS}
        return Evaluator(
            EvaluationContext(
                variables=variables or {},
                operators=operators,
            )
        )

    def test_power_via_evaluator(self):
        evaluator = self._make_evaluator({"x": 3.0})
        result = evaluator.evaluate(parse_expression("power(x, 2)"))
        assert result == pytest.approx(9.0)

    def test_max_via_evaluator(self):
        evaluator = self._make_evaluator({"x": -5.0})
        result = evaluator.evaluate(parse_expression("max(x, 0)"))
        assert result == pytest.approx(0.0)

    def test_min_via_evaluator(self):
        evaluator = self._make_evaluator({"x": 5.0})
        result = evaluator.evaluate(parse_expression("min(x, 3)"))
        assert result == pytest.approx(3.0)

    def test_power_array_via_evaluator(self):
        arr = np.array([2.0, 3.0])
        evaluator = self._make_evaluator({"x": arr})
        result = evaluator.evaluate(parse_expression("power(x, 3)"))
        np.testing.assert_allclose(result, [8.0, 27.0])
