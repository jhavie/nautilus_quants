# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for new TS operators: ts_slope, ts_rsquare, ts_residual, ts_percentile, ema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.time_series import (
    Ema,
    TsPercentile,
    TsResidual,
    TsRsquare,
    TsSlope,
)


# ── Fixtures ──


@pytest.fixture
def linear_data() -> np.ndarray:
    """Perfect linear series: [0, 1, 2, 3, 4]."""
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def noisy_data() -> np.ndarray:
    """Non-linear series with known properties."""
    return np.array([10.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0, 18.0])


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Panel DataFrame with 2 instruments, 20 timestamps."""
    idx = pd.date_range("2024-01-01", periods=20, freq="4h")
    return pd.DataFrame(
        {
            "A": np.arange(1, 21, dtype=float),
            "B": np.arange(20, 0, -1, dtype=float),
        },
        index=idx,
    )


# ── TsSlope ──


class TestTsSlope:
    def test_perfect_linear(self, linear_data: np.ndarray) -> None:
        assert TsSlope().compute(linear_data, 5) == pytest.approx(1.0)

    def test_negative_slope(self) -> None:
        data = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        assert TsSlope().compute(data, 5) == pytest.approx(-1.0)

    def test_constant_series(self) -> None:
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert TsSlope().compute(data, 5) == pytest.approx(0.0)

    def test_insufficient_data(self) -> None:
        data = np.array([1.0, 2.0])
        assert np.isnan(TsSlope().compute(data, 5))

    def test_panel(self, panel_df: pd.DataFrame) -> None:
        result = TsSlope().compute_panel(panel_df, 5)
        assert result.shape == panel_df.shape
        # col A is [1..20], slope should be 1.0 everywhere after warmup
        assert result["A"].dropna().iloc[-1] == pytest.approx(1.0)
        # col B is [20..1], slope should be -1.0
        assert result["B"].dropna().iloc[-1] == pytest.approx(-1.0)
        # First 4 rows should be NaN (window=5)
        assert result.iloc[:4].isna().all().all()


# ── TsRsquare ──


class TestTsRsquare:
    def test_perfect_linear(self, linear_data: np.ndarray) -> None:
        assert TsRsquare().compute(linear_data, 5) == pytest.approx(1.0)

    def test_constant_series_nan(self) -> None:
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert np.isnan(TsRsquare().compute(data, 5))

    def test_noisy_data_between_0_and_1(self, noisy_data: np.ndarray) -> None:
        r2 = TsRsquare().compute(noisy_data, 8)
        assert 0.0 < r2 < 1.0

    def test_panel(self, panel_df: pd.DataFrame) -> None:
        result = TsRsquare().compute_panel(panel_df, 5)
        # Perfect linear series → R² = 1.0
        assert result["A"].dropna().iloc[-1] == pytest.approx(1.0)


# ── TsResidual ──


class TestTsResidual:
    def test_perfect_linear_zero_residual(self, linear_data: np.ndarray) -> None:
        assert TsResidual().compute(linear_data, 5) == pytest.approx(0.0)

    def test_positive_residual(self) -> None:
        # [0, 1, 2, 3, 5] — last point above trend
        data = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        resi = TsResidual().compute(data, 5)
        assert resi > 0

    def test_negative_residual(self) -> None:
        # [0, 1, 2, 3, 1] — last point below trend
        data = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
        resi = TsResidual().compute(data, 5)
        assert resi < 0

    def test_panel(self, panel_df: pd.DataFrame) -> None:
        result = TsResidual().compute_panel(panel_df, 5)
        # Perfect linear → residual = 0
        np.testing.assert_allclose(
            result["A"].dropna().values, 0.0, atol=1e-10,
        )


# ── TsPercentile ──


class TestTsPercentile:
    def test_80th_percentile(self) -> None:
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = TsPercentile().compute(data, 5, extra_0=0.8)
        assert result == pytest.approx(3.2)

    def test_20th_percentile(self) -> None:
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = TsPercentile().compute(data, 5, extra_0=0.2)
        assert result == pytest.approx(0.8)

    def test_50th_percentile_is_median(self) -> None:
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = TsPercentile().compute(data, 5, extra_0=0.5)
        assert result == pytest.approx(5.0)

    def test_panel(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        df = pd.DataFrame({"A": range(1, 11), "B": range(10, 0, -1)}, index=idx, dtype=float)
        result = TsPercentile().compute_panel(df, 5, extra_0=0.8)
        assert result.shape == df.shape
        # For A=[6,7,8,9,10], 80th percentile = 9.2
        assert result["A"].iloc[-1] == pytest.approx(9.2)


# ── Ema ──


class TestEma:
    def test_constant_series(self) -> None:
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert Ema().compute(data, 3) == pytest.approx(5.0)

    def test_matches_pandas_ewm(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = pd.Series(data).ewm(span=3, adjust=False).mean().iloc[-1]
        assert Ema().compute(data, 3) == pytest.approx(expected)

    def test_no_warmup_nans(self) -> None:
        """EMA has no NaN warmup — it starts from the first value."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        df = pd.DataFrame({"A": range(1, 11)}, index=idx, dtype=float)
        result = Ema().compute_panel(df, 3)
        assert result.notna().all().all()

    def test_panel_matches_pandas(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        df = pd.DataFrame({"A": range(1, 11), "B": range(10, 0, -1)}, index=idx, dtype=float)
        result = Ema().compute_panel(df, 5)
        expected = df.ewm(span=5, adjust=False).mean()
        pd.testing.assert_frame_equal(result, expected)


# ── Evaluator integration ──


class TestEvaluatorIntegration:
    """Test new operators work through the expression evaluator."""

    @pytest.fixture
    def evaluator(self):
        from nautilus_quants.factors.engine.evaluator import Evaluator
        from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
        from nautilus_quants.factors.operators.math import MATH_OPERATORS
        from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

        idx = pd.date_range("2024-01-01", periods=30, freq="4h")
        panel = {
            "close": pd.DataFrame(
                {"A": np.arange(1, 31, dtype=float), "B": np.arange(30, 0, -1, dtype=float)},
                index=idx,
            ),
        }
        return Evaluator(
            panel_fields=panel,
            ts_ops=dict(TS_OPERATOR_INSTANCES),
            cs_ops=dict(CS_OPERATOR_INSTANCES),
            math_ops=dict(MATH_OPERATORS),
            parameters={},
        )

    @pytest.mark.parametrize("expr", [
        "ts_slope(close, 10)",
        "ts_rsquare(close, 10)",
        "ts_residual(close, 10)",
        "ts_percentile(close, 10, 0.8)",
        "ema(close, 12)",
        "ts_slope(close, 5) / close",
        "ts_percentile(close, 5, 0.8) / close",
    ])
    def test_expression_evaluates(self, evaluator, expr: str) -> None:
        from nautilus_quants.factors.expression import parse_expression

        ast = parse_expression(expr)
        result = evaluator.evaluate(ast)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (30, 2)
        # At least some non-NaN values
        assert result.notna().any().any()
