"""Tests for division-by-zero handling in Evaluator and VectorizedEvaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.expression.evaluator import (
    Evaluator,
    EvaluationContext,
    VectorizedEvaluator,
)


class TestEvaluatorDivisionByZero:
    """Evaluator (scalar/numpy) must produce NaN on division by zero."""

    def test_scalar_divided_by_zero(self):
        ctx = EvaluationContext(variables={"a": 10.0, "b": 0.0})
        evaluator = Evaluator(ctx)
        result = evaluator.evaluate(parse_expression("a / b"))
        assert np.isnan(result)

    def test_array_divided_by_scalar_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        ctx = EvaluationContext(variables={"a": arr, "b": 0.0})
        evaluator = Evaluator(ctx)
        result = evaluator.evaluate(parse_expression("a / b"))
        assert isinstance(result, np.ndarray)
        assert np.all(np.isnan(result))

    def test_array_divided_by_array_with_zeros(self):
        """CRITICAL: array / array-containing-zeros must produce NaN at zero positions."""
        left = np.array([10.0, 20.0, 30.0, 40.0])
        right = np.array([2.0, 0.0, 5.0, 0.0])
        ctx = EvaluationContext(variables={"a": left, "b": right})
        evaluator = Evaluator(ctx)
        result = evaluator.evaluate(parse_expression("a / b"))
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(5.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(6.0)
        assert np.isnan(result[3])

    def test_scalar_divided_by_nonzero(self):
        ctx = EvaluationContext(variables={"a": 10.0, "b": 2.0})
        evaluator = Evaluator(ctx)
        result = evaluator.evaluate(parse_expression("a / b"))
        assert result == pytest.approx(5.0)

    def test_array_divided_by_nonzero_array(self):
        left = np.array([10.0, 20.0])
        right = np.array([2.0, 5.0])
        ctx = EvaluationContext(variables={"a": left, "b": right})
        evaluator = Evaluator(ctx)
        result = evaluator.evaluate(parse_expression("a / b"))
        np.testing.assert_allclose(result, [5.0, 4.0])


class TestVectorizedEvaluatorDivisionByZero:
    """VectorizedEvaluator (pd.Series) must produce NaN on division by zero."""

    idx = pd.date_range("2024-01-01", periods=4, freq="h")

    def test_series_divided_by_scalar_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.idx)
        evaluator = VectorizedEvaluator(
            variables={"a": s}, ts_operators={}, math_operators={},
        )
        result = evaluator.evaluate(parse_expression("a / 0"))
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_series_divided_by_series_with_zeros(self):
        """CRITICAL: Series / Series-containing-zeros must produce NaN at zero positions."""
        left = pd.Series([10.0, 20.0, 30.0, 40.0], index=self.idx)
        right = pd.Series([2.0, 0.0, 5.0, 0.0], index=self.idx)
        evaluator = VectorizedEvaluator(
            variables={"a": left, "b": right},
            ts_operators={}, math_operators={},
        )
        result = evaluator.evaluate(parse_expression("a / b"))
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == pytest.approx(5.0)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(6.0)
        assert np.isnan(result.iloc[3])

    def test_series_divided_by_nonzero_series(self):
        left = pd.Series([10.0, 20.0, 30.0, 40.0], index=self.idx)
        right = pd.Series([2.0, 4.0, 5.0, 8.0], index=self.idx)
        evaluator = VectorizedEvaluator(
            variables={"a": left, "b": right},
            ts_operators={}, math_operators={},
        )
        result = evaluator.evaluate(parse_expression("a / b"))
        pd.testing.assert_series_equal(
            result, pd.Series([5.0, 5.0, 6.0, 5.0], index=self.idx),
        )
