# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for factor cutting operators (rolling_selmean_*, ts_max_to_min, diff_sign, ts_meanrank)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.cross_sectional import (
    CS_OPERATOR_INSTANCES,
    TsMeanrank,
)
from nautilus_quants.factors.operators.time_series import (
    DiffSign,
    RollingSelmeanBtm,
    RollingSelmeanDiff,
    RollingSelmeanTop,
    TS_OPERATOR_INSTANCES,
    TsMaxToMin,
)


# ── Fixtures ──


@pytest.fixture
def selmean_xy() -> tuple[np.ndarray, np.ndarray]:
    """Paired (x, y) arrays for selmean tests."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
    return x, y


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Panel DataFrame with 3 instruments, 20 timestamps."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=20, freq="4h")
    return pd.DataFrame(
        {
            "A": rng.standard_normal(20).cumsum() + 100,
            "B": rng.standard_normal(20).cumsum() + 200,
            "C": rng.standard_normal(20).cumsum() + 150,
        },
        index=idx,
    )


@pytest.fixture
def panel_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired (x_panel, y_panel) for selmean panel tests."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=20, freq="4h")
    x_df = pd.DataFrame(
        {
            "A": rng.standard_normal(20).cumsum(),
            "B": rng.standard_normal(20).cumsum(),
            "C": rng.standard_normal(20).cumsum(),
        },
        index=idx,
    )
    y_df = pd.DataFrame(
        {
            "A": rng.standard_normal(20).cumsum(),
            "B": rng.standard_normal(20).cumsum(),
            "C": rng.standard_normal(20).cumsum(),
        },
        index=idx,
    )
    return x_df, y_df


# ── RollingSelmeanTop ──


class TestRollingSelmeanTop:
    def test_basic(self, selmean_xy: tuple[np.ndarray, np.ndarray]) -> None:
        """Top-2 by y: y[1]=50->x[1]=2, y[4]=40->x[4]=5 -> mean=3.5."""
        x, y = selmean_xy
        result = RollingSelmeanTop().compute(x, window=5, data2=y, extra_0=2)
        assert result == pytest.approx(3.5)

    def test_insufficient_data_returns_nan(self) -> None:
        """Window larger than data -> NaN."""
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])
        result = RollingSelmeanTop().compute(x, window=5, data2=y, extra_0=2)
        assert np.isnan(result)

    def test_no_data2_returns_nan(self) -> None:
        """Missing y series -> NaN."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = RollingSelmeanTop().compute(x, window=5, data2=None, extra_0=2)
        assert np.isnan(result)


# ── RollingSelmeanBtm ──


class TestRollingSelmeanBtm:
    def test_basic(self, selmean_xy: tuple[np.ndarray, np.ndarray]) -> None:
        """Btm-2 by y: y[0]=10->x[0]=1, y[3]=20->x[3]=4 -> mean=2.5."""
        x, y = selmean_xy
        result = RollingSelmeanBtm().compute(x, window=5, data2=y, extra_0=2)
        assert result == pytest.approx(2.5)


# ── RollingSelmeanDiff ──


class TestRollingSelmeanDiff:
    def test_equals_top_minus_btm(
        self, selmean_xy: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """diff = top(3.5) - btm(2.5) = 1.0."""
        x, y = selmean_xy
        result = RollingSelmeanDiff().compute(x, window=5, data2=y, extra_0=2)
        assert result == pytest.approx(1.0)

    def test_panel_equals_top_minus_btm(
        self, panel_pair: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Panel diff should exactly equal top_panel - btm_panel."""
        x_df, y_df = panel_pair
        window, n = 10, 3

        top_df = RollingSelmeanTop().compute_panel(
            x_df, window, data2=y_df, extra_0=n
        )
        btm_df = RollingSelmeanBtm().compute_panel(
            x_df, window, data2=y_df, extra_0=n
        )
        diff_df = RollingSelmeanDiff().compute_panel(
            x_df, window, data2=y_df, extra_0=n
        )

        expected = top_df - btm_df
        pd.testing.assert_frame_equal(diff_df, expected)


# ── RollingSelmean NaN handling ──


class TestRollingSelmeanNan:
    def test_top_with_nan_in_x(self) -> None:
        """NaN in x should be skipped; valid top-2 still computes."""
        x = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
        # Valid pairs: (2,50), (3,30), (4,20), (5,40)
        # top-2 by y: y=50->x=2, y=40->x=5 -> mean=3.5
        result = RollingSelmeanTop().compute(x, window=5, data2=y, extra_0=2)
        assert result == pytest.approx(3.5)

    def test_top_with_nan_in_y(self) -> None:
        """NaN in y should be skipped."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, np.nan, 30.0, 20.0, 40.0])
        # Valid pairs: (1,10), (3,30), (4,20), (5,40)
        # top-2 by y: y=40->x=5, y=30->x=3 -> mean=4.0
        result = RollingSelmeanTop().compute(x, window=5, data2=y, extra_0=2)
        assert result == pytest.approx(4.0)

    def test_fewer_valid_than_n_returns_nan(self) -> None:
        """If fewer than n valid values exist, result is NaN."""
        x = np.array([np.nan, np.nan, np.nan, 4.0, 5.0])
        y = np.array([10.0, 20.0, np.nan, np.nan, 40.0])
        # Valid pairs: only (5, 40) -> n_valid=1 < n=2 -> NaN
        result = RollingSelmeanTop().compute(x, window=5, data2=y, extra_0=2)
        assert np.isnan(result)


# ── RollingSelmean Panel ──


class TestRollingSelmeanPanel:
    def test_panel_matches_per_column(
        self, panel_pair: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Panel compute should match per-column scalar compute."""
        x_df, y_df = panel_pair
        window, n = 10, 2
        op = RollingSelmeanTop()

        panel_result = op.compute_panel(x_df, window, data2=y_df, extra_0=n)
        assert panel_result.shape == x_df.shape

        # Verify per-column match for column "A"
        col = "A"
        x_col = x_df[col].values
        y_col = y_df[col].values
        for t in range(window - 1, len(x_col)):
            expected = op.compute(
                x_col[: t + 1], window, data2=y_col[: t + 1], extra_0=n
            )
            actual = panel_result[col].iloc[t]
            if np.isnan(expected):
                assert np.isnan(actual), f"Mismatch at t={t}: expected NaN, got {actual}"
            else:
                assert actual == pytest.approx(expected, abs=1e-10), (
                    f"Mismatch at t={t}: expected {expected}, got {actual}"
                )


# ── TsMaxToMin ──


class TestTsMaxToMin:
    def test_basic(self) -> None:
        """max(5) - min(1) = 4.0."""
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = TsMaxToMin().compute(x, window=5)
        assert result == pytest.approx(4.0)

    def test_constant_series(self) -> None:
        """Constant series -> amplitude = 0."""
        x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = TsMaxToMin().compute(x, window=5)
        assert result == pytest.approx(0.0)

    def test_insufficient_data(self) -> None:
        """Window larger than data -> NaN."""
        x = np.array([1.0, 2.0])
        result = TsMaxToMin().compute(x, window=5)
        assert np.isnan(result)

    def test_with_nan(self) -> None:
        """NaN in window -> nanmax/nanmin still computes."""
        x = np.array([1.0, np.nan, 3.0, 5.0, 4.0])
        result = TsMaxToMin().compute(x, window=5)
        # nanmax=5, nanmin=1 -> 4.0
        assert result == pytest.approx(4.0)

    def test_panel(self, panel_df: pd.DataFrame) -> None:
        """Panel compute matches rolling(max) - rolling(min)."""
        window = 5
        result = TsMaxToMin().compute_panel(panel_df, window)
        expected = panel_df.rolling(window).max() - panel_df.rolling(window).min()
        pd.testing.assert_frame_equal(result, expected)


# ── DiffSign ──


class TestDiffSign:
    def test_positive_deviation(self) -> None:
        """Last value above mean -> sign = +1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        # mean = 4.0, last = 10 -> sign(10-4) = 1
        result = DiffSign().compute(x, window=5)
        assert result == pytest.approx(1.0)

    def test_negative_deviation(self) -> None:
        """Last value below mean -> sign = -1."""
        x = np.array([10.0, 2.0, 3.0, 4.0, 1.0])
        # mean = 4.0, last = 1 -> sign(1-4) = -1
        result = DiffSign().compute(x, window=5)
        assert result == pytest.approx(-1.0)

    def test_zero_deviation(self) -> None:
        """Last value equals mean -> sign = 0."""
        x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        # mean = 3.0, last = 3 -> sign(0) = 0
        result = DiffSign().compute(x, window=5)
        assert result == pytest.approx(0.0)

    def test_insufficient_data(self) -> None:
        """Window larger than data -> NaN."""
        x = np.array([1.0, 2.0])
        result = DiffSign().compute(x, window=5)
        assert np.isnan(result)

    def test_panel(self, panel_df: pd.DataFrame) -> None:
        """Panel compute matches sign(data - rolling_mean)."""
        window = 5
        result = DiffSign().compute_panel(panel_df, window)
        expected = np.sign(panel_df - panel_df.rolling(window).mean())
        pd.testing.assert_frame_equal(result, expected)


# ── TsMeanrank ──


class TestTsMeanrank:
    def test_panel_basic(self, panel_df: pd.DataFrame) -> None:
        """Panel: cs_rank then rolling mean."""
        window = 5
        result = TsMeanrank().compute_panel(panel_df, window)

        # Manual: rank row-wise, then rolling mean column-wise
        ranked = panel_df.rank(axis=1, method="min", pct=True)
        expected = ranked.rolling(window, min_periods=1).mean()
        pd.testing.assert_frame_equal(result, expected)

    def test_panel_shape(self, panel_df: pd.DataFrame) -> None:
        """Output shape matches input shape."""
        result = TsMeanrank().compute_panel(panel_df, window=5)
        assert result.shape == panel_df.shape

    def test_panel_values_between_0_and_1(self, panel_df: pd.DataFrame) -> None:
        """Since ranks are in (0, 1], averaged ranks should remain in (0, 1]."""
        result = TsMeanrank().compute_panel(panel_df, window=5)
        valid = result.dropna()
        assert (valid > 0).all().all()
        assert (valid <= 1).all().all()

    def test_scalar_fallback_returns_rank(self) -> None:
        """Scalar mode returns cross-sectional rank (no time-series context)."""
        values = {"A": 10.0, "B": 30.0, "C": 20.0}
        result = TsMeanrank().compute(values)
        # rank(min, pct=True): A=1/3, C=2/3, B=3/3
        assert result["A"] == pytest.approx(1 / 3)
        assert result["C"] == pytest.approx(2 / 3)
        assert result["B"] == pytest.approx(1.0)


# ── DSL Expression Parse + Evaluate ──


class TestDslExpressionParseAndEvaluate:
    """Test that new operators work through the full expression evaluator."""

    @pytest.fixture
    def evaluator(self):
        from nautilus_quants.factors.engine.evaluator import Evaluator
        from nautilus_quants.factors.operators.math import MATH_OPERATORS

        rng = np.random.default_rng(99)
        idx = pd.date_range("2024-01-01", periods=30, freq="4h")
        panel = {
            "close": pd.DataFrame(
                {
                    "A": rng.standard_normal(30).cumsum() + 100,
                    "B": rng.standard_normal(30).cumsum() + 100,
                },
                index=idx,
            ),
            "returns": pd.DataFrame(
                {
                    "A": rng.standard_normal(30) * 0.01,
                    "B": rng.standard_normal(30) * 0.01,
                },
                index=idx,
            ),
            "volume": pd.DataFrame(
                {
                    "A": rng.uniform(100, 1000, 30),
                    "B": rng.uniform(100, 1000, 30),
                },
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

    @pytest.mark.parametrize(
        "expr",
        [
            "rolling_selmean_diff(returns, volume, 10, 3)",
            "rolling_selmean_top(returns, volume, 10, 3)",
            "rolling_selmean_btm(returns, volume, 10, 3)",
            "ts_max_to_min(close, 10)",
            "diff_sign(close, 10)",
            "ts_meanrank(close, 5)",
        ],
    )
    def test_expression_evaluates(self, evaluator, expr: str) -> None:
        from nautilus_quants.factors.expression import parse_expression

        ast = parse_expression(expr)
        result = evaluator.evaluate(ast)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (30, 2)
        # At least some non-NaN values
        assert result.notna().any().any()


# ── Operator Registration ──


class TestOperatorRegistration:
    """Verify new operators are registered in the instance registries."""

    @pytest.mark.parametrize(
        "name",
        [
            "rolling_selmean_top",
            "rolling_selmean_btm",
            "rolling_selmean_diff",
            "ts_max_to_min",
            "diff_sign",
        ],
    )
    def test_ts_operator_registered(self, name: str) -> None:
        assert name in TS_OPERATOR_INSTANCES

    def test_ts_meanrank_registered_in_cs(self) -> None:
        assert "ts_meanrank" in CS_OPERATOR_INSTANCES
