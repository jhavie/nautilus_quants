# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Evaluator — AST evaluator for panel factor computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.engine.evaluator import Evaluator, EvaluationError
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES, CsRank
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(
    n_timestamps: int = 20,
    n_instruments: int = 5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create a reproducible test panel with OHLCV + returns + vwap."""
    rng = np.random.RandomState(seed)
    instruments = [f"INST_{i}" for i in range(n_instruments)]

    close = pd.DataFrame(
        rng.randn(n_timestamps, n_instruments).cumsum(axis=0) + 100,
        columns=instruments,
    )
    open_ = close.shift(1).bfill() + rng.randn(n_timestamps, n_instruments) * 0.5
    high = pd.DataFrame(
        np.maximum(open_.values, close.values) + np.abs(rng.randn(n_timestamps, n_instruments)),
        columns=instruments,
    )
    low = pd.DataFrame(
        np.minimum(open_.values, close.values) - np.abs(rng.randn(n_timestamps, n_instruments)),
        columns=instruments,
    )
    volume = pd.DataFrame(
        np.abs(rng.randn(n_timestamps, n_instruments)) * 1000 + 500,
        columns=instruments,
    )
    returns = close / close.shift(1) - 1
    vwap = (high + low + close) / 3

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": returns,
        "vwap": vwap,
    }


def _make_evaluator(panel: dict[str, pd.DataFrame]) -> Evaluator:
    return Evaluator(
        panel_fields=panel,
        ts_ops=TS_OPERATOR_INSTANCES,
        cs_ops=CS_OPERATOR_INSTANCES,
        math_ops=MATH_OPERATORS,
    )


def _evaluate(expression: str, panel: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame | float:
    if panel is None:
        panel = _make_panel()
    evaluator = _make_evaluator(panel)
    return evaluator.evaluate(parse_expression(expression))


# ---------------------------------------------------------------------------
# Basic variable access
# ---------------------------------------------------------------------------


class TestVariableAccess:
    def test_close_returns_dataframe(self) -> None:
        panel = _make_panel()
        result = _evaluate("close", panel)
        pd.testing.assert_frame_equal(result, panel["close"])

    def test_scalar_number(self) -> None:
        result = _evaluate("42.5")
        assert result == 42.5

    def test_unknown_variable_raises(self) -> None:
        with pytest.raises(EvaluationError, match="Unknown variable"):
            _evaluate("nonexistent")


# ---------------------------------------------------------------------------
# Arithmetic operations
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_addition(self) -> None:
        panel = _make_panel()
        result = _evaluate("close + open", panel)
        expected = panel["close"] + panel["open"]
        pd.testing.assert_frame_equal(result, expected)

    def test_subtraction(self) -> None:
        panel = _make_panel()
        result = _evaluate("close - open", panel)
        expected = panel["close"] - panel["open"]
        pd.testing.assert_frame_equal(result, expected)

    def test_multiplication_scalar(self) -> None:
        panel = _make_panel()
        result = _evaluate("-1 * close", panel)
        expected = -1 * panel["close"]
        pd.testing.assert_frame_equal(result, expected)

    def test_division_safe(self) -> None:
        panel = _make_panel()
        result = _evaluate("close / volume", panel)
        expected = panel["close"] / panel["volume"]
        pd.testing.assert_frame_equal(result, expected)

    def test_division_by_zero(self) -> None:
        panel = _make_panel()
        result = _evaluate("close / 0", panel)
        assert isinstance(result, pd.DataFrame)
        assert result.isna().all().all()


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_greater_than(self) -> None:
        panel = _make_panel()
        result = _evaluate("close > open", panel)
        assert isinstance(result, pd.DataFrame)
        # Result should be 0.0 or 1.0
        assert set(result.values.flatten()) <= {0.0, 1.0, np.nan}

    def test_less_than(self) -> None:
        result = _evaluate("volume < 1000")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------


class TestTsOperators:
    def test_ts_mean(self) -> None:
        panel = _make_panel()
        result = _evaluate("ts_mean(close, 5)", panel)
        expected = panel["close"].rolling(5).mean()
        pd.testing.assert_frame_equal(result, expected)

    def test_delay(self) -> None:
        panel = _make_panel()
        result = _evaluate("delay(close, 1)", panel)
        expected = panel["close"].shift(1)
        pd.testing.assert_frame_equal(result, expected)

    def test_delta(self) -> None:
        panel = _make_panel()
        result = _evaluate("delta(close, 1)", panel)
        expected = panel["close"].diff(1)
        pd.testing.assert_frame_equal(result, expected)

    def test_stddev(self) -> None:
        panel = _make_panel()
        result = _evaluate("stddev(close, 10)", panel)
        expected = panel["close"].rolling(10).std(ddof=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_correlation(self) -> None:
        panel = _make_panel()
        result = _evaluate("correlation(close, volume, 10)", panel)
        expected = pd.DataFrame(
            {col: panel["close"][col].rolling(10).corr(panel["volume"][col]) for col in panel["close"].columns},
            index=panel["close"].index,
        )
        # NaN propagates naturally; only inf → NaN
        expected = expected.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------


class TestCsOperators:
    def test_rank_is_cross_sectional(self) -> None:
        """rank(x) should rank across instruments at each timestamp."""
        panel = _make_panel(n_timestamps=5, n_instruments=3)
        result = _evaluate("rank(close)", panel)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == panel["close"].shape
        # Values should be in [0, 1]
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_rank_matches_cs_rank_operator(self) -> None:
        """Panel rank should match CsRank.compute_vectorized."""
        panel = _make_panel()
        result = _evaluate("rank(close)", panel)
        expected = CsRank().compute_vectorized(panel["close"])
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# CS → TS nesting (core value of Panel architecture)
# ---------------------------------------------------------------------------


class TestCsTsNesting:
    def test_alpha003_cs_inside_ts(self) -> None:
        """alpha003: -1 * correlation(rank(open), rank(volume), 10)"""
        panel = _make_panel()
        result = _evaluate("-1 * correlation(rank(open), rank(volume), 10)", panel)

        rank_op = CsRank()
        rank_open = rank_op.compute_vectorized(panel["open"])
        rank_vol = rank_op.compute_vectorized(panel["volume"])
        corr = pd.DataFrame(
            {col: rank_open[col].rolling(10).corr(rank_vol[col]) for col in panel["close"].columns},
            index=panel["close"].index,
        )
        # NaN propagates naturally; only inf → NaN
        expected = -1 * corr.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_frame_equal(result, expected)

    def test_alpha013_cs_ts_cs(self) -> None:
        """alpha013: -1 * rank(covariance(rank(close), rank(volume), 5))

        Three layers: CS → TS → CS.
        """
        panel = _make_panel()
        result = _evaluate("-1 * rank(covariance(rank(close), rank(volume), 5))", panel)

        rank_op = CsRank()
        rank_close = rank_op.compute_vectorized(panel["close"])
        rank_vol = rank_op.compute_vectorized(panel["volume"])
        cov_df = pd.DataFrame(
            {col: rank_close[col].rolling(5).cov(rank_vol[col], ddof=1) for col in panel["close"].columns},
            index=panel["close"].index,
        )
        rank_cov = rank_op.compute_vectorized(cov_df)
        expected = -1 * rank_cov

        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Math operators
# ---------------------------------------------------------------------------


class TestMathOperators:
    def test_log(self) -> None:
        panel = _make_panel()
        result = _evaluate("log(volume)", panel)
        expected = np.log(panel["volume"])
        pd.testing.assert_frame_equal(result, expected)

    def test_abs(self) -> None:
        panel = _make_panel()
        result = _evaluate("abs(returns)", panel)
        expected = np.abs(panel["returns"])
        pd.testing.assert_frame_equal(result, expected)

    def test_sign(self) -> None:
        panel = _make_panel()
        result = _evaluate("sign(returns)", panel)
        expected = np.sign(panel["returns"])
        pd.testing.assert_frame_equal(result, expected)

    def test_if_else_with_dataframes(self) -> None:
        panel = _make_panel()
        result = _evaluate("if_else(close > open, 1, -1)", panel)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == panel["close"].shape

    def test_is_nan(self) -> None:
        panel = _make_panel()
        # Returns has NaN in first row
        result = _evaluate("is_nan(returns)", panel)
        assert isinstance(result, pd.DataFrame)
        # First row should be 1.0 (NaN), rest should be 0.0
        assert (result.iloc[0] == 1.0).all()
        assert (result.iloc[-1] == 0.0).all()


# ---------------------------------------------------------------------------
# Ternary operator
# ---------------------------------------------------------------------------


class TestTernary:
    def test_ternary_with_dataframes(self) -> None:
        panel = _make_panel()
        # Note: ternary syntax a ? b : c may be parsed as if_else(a, b, c)
        result = _evaluate("if_else(close > open, close, open)", panel)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == panel["close"].shape


# ---------------------------------------------------------------------------
# BRAIN operator aliases
# ---------------------------------------------------------------------------


class TestBrainAliases:
    def test_ts_corr_alias(self) -> None:
        """ts_corr = correlation."""
        panel = _make_panel()
        r1 = _evaluate("ts_corr(close, volume, 10)", panel)
        r2 = _evaluate("correlation(close, volume, 10)", panel)
        pd.testing.assert_frame_equal(r1, r2)

    def test_ts_delay_alias(self) -> None:
        """ts_delay = delay."""
        panel = _make_panel()
        r1 = _evaluate("ts_delay(close, 1)", panel)
        r2 = _evaluate("delay(close, 1)", panel)
        pd.testing.assert_frame_equal(r1, r2)

    def test_stddev_alias(self) -> None:
        """stddev = ts_std_dev."""
        panel = _make_panel()
        r1 = _evaluate("stddev(close, 10)", panel)
        r2 = _evaluate("ts_std_dev(close, 10)", panel)
        pd.testing.assert_frame_equal(r1, r2)
