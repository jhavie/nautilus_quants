"""Tests for FactorEvaluator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.analysis.config import AlphaAnalysisConfig
from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
from nautilus_quants.factors.expression import VectorizedEvaluator, parse_expression
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES


def _make_mock_bar(instrument_id: str, ts_event: int, close: float, volume: float = 1000.0):
    """Create a mock Bar for testing."""
    bar = MagicMock()
    bar.ts_event = ts_event
    bar.open = MagicMock(__float__=lambda s: close * 0.99)
    bar.high = MagicMock(__float__=lambda s: close * 1.01)
    bar.low = MagicMock(__float__=lambda s: close * 0.98)
    bar.close = MagicMock(__float__=lambda s: close)
    bar.volume = MagicMock(__float__=lambda s: volume)

    bar_type = MagicMock()
    bar_type.instrument_id = instrument_id
    bar.bar_type = bar_type
    return bar


class TestFactorEvaluator:
    """Test FactorEvaluator core logic."""

    def test_init(self):
        mock_factor_config = MagicMock()
        evaluator = FactorEvaluator(mock_factor_config)
        assert evaluator._ts_engine is not None
        assert evaluator._cs_engine is not None

    def test_evaluate_returns_tuple(self):
        """Test evaluate returns (factor_series_dict, pricing_df)."""
        mock_factor_config = MagicMock()

        with patch.object(FactorEvaluator, "_compute_all_factors") as mock_compute:
            # Mock the internal computation to return test data
            idx = pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-01"), "BTCUSDT.BINANCE"),
                    (pd.Timestamp("2024-01-01"), "ETHUSDT.BINANCE"),
                ],
                names=["date", "asset"],
            )
            mock_compute.return_value = (
                {"volume": pd.Series([1.0, 2.0], index=idx)},
                pd.DataFrame(
                    {"BTCUSDT.BINANCE": [100.0], "ETHUSDT.BINANCE": [50.0]},
                    index=[pd.Timestamp("2024-01-01")],
                ),
            )

            evaluator = FactorEvaluator(mock_factor_config)
            bars_by_instrument = {
                "BTCUSDT.BINANCE": [_make_mock_bar("BTCUSDT.BINANCE", 1000, 100.0)],
                "ETHUSDT.BINANCE": [_make_mock_bar("ETHUSDT.BINANCE", 1000, 50.0)],
            }

            factor_series, pricing = evaluator.evaluate(bars_by_instrument)

            assert isinstance(factor_series, dict)
            assert isinstance(pricing, pd.DataFrame)

    def test_evaluate_factor_series_has_multiindex(self):
        """Test that factor series have MultiIndex(date, asset)."""
        mock_factor_config = MagicMock()

        with patch.object(FactorEvaluator, "_compute_all_factors") as mock_compute:
            idx = pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-01"), "BTCUSDT.BINANCE"),
                    (pd.Timestamp("2024-01-01"), "ETHUSDT.BINANCE"),
                ],
                names=["date", "asset"],
            )
            mock_compute.return_value = (
                {"volume": pd.Series([1.0, 2.0], index=idx)},
                pd.DataFrame(
                    {"BTCUSDT.BINANCE": [100.0], "ETHUSDT.BINANCE": [50.0]},
                    index=[pd.Timestamp("2024-01-01")],
                ),
            )

            evaluator = FactorEvaluator(mock_factor_config)
            bars = {
                "BTCUSDT.BINANCE": [_make_mock_bar("BTCUSDT.BINANCE", 1000, 100.0)],
                "ETHUSDT.BINANCE": [_make_mock_bar("ETHUSDT.BINANCE", 1000, 50.0)],
            }

            factor_series, _ = evaluator.evaluate(bars)

            for name, series in factor_series.items():
                assert isinstance(series.index, pd.MultiIndex)
                assert series.index.names == ["date", "asset"]

    def test_run_alphalens_returns_dict(self):
        """Test run_alphalens returns results dict."""
        mock_factor_config = MagicMock()
        evaluator = FactorEvaluator(mock_factor_config)

        # Create minimal alphalens-compatible data
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        assets = ["A", "B", "C", "D", "E"]

        # Factor series with MultiIndex
        tuples = [(d, a) for d in dates for a in assets]
        idx = pd.MultiIndex.from_tuples(tuples, names=["date", "asset"])
        np.random.seed(42)
        factor_values = np.random.randn(len(tuples))
        factor_series = pd.Series(factor_values, index=idx)

        # Pricing DataFrame
        pricing = pd.DataFrame(
            np.random.uniform(90, 110, (len(dates), len(assets))).cumsum(axis=0),
            index=dates,
            columns=assets,
        )

        config = AlphaAnalysisConfig(
            catalog_path="/test",
            factor_config_path="test.yaml",
            instrument_ids=assets,
            periods=(1, 4),
            quantiles=3,
            max_loss=0.99,
            filter_zscore=None,
        )

        result = evaluator.run_alphalens(factor_series, pricing, config)

        assert isinstance(result, dict)
        assert "factor_data" in result
        assert "ic" in result


class TestCsExpressionParser:
    """Test CS expression vectorized evaluation (Issues 5, H4)."""

    @pytest.fixture()
    def evaluator(self):
        mock_config = MagicMock()
        return FactorEvaluator(mock_config)

    @pytest.fixture()
    def panels(self) -> dict[str, pd.DataFrame]:
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        return {
            "x": pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=idx, dtype=float),
            "y": pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [50, 40, 30, 20, 10]}, index=idx, dtype=float),
        }

    def test_simple_weighted_sum(self, evaluator, panels):
        result = evaluator._evaluate_cs_expression_vectorized("0.6 * x + 0.4 * y", panels)
        expected = panels["x"] * 0.6 + panels["y"] * 0.4
        pd.testing.assert_frame_equal(result, expected)

    def test_negative_leading_weighted_sum(self, evaluator, panels):
        result = evaluator._evaluate_cs_expression_vectorized("- 0.5 * x - 0.5 * y", panels)
        expected = panels["x"] * (-0.5) + panels["y"] * (-0.5)
        pd.testing.assert_frame_equal(result, expected)

    def test_scientific_notation_weights(self, evaluator, panels):
        result = evaluator._evaluate_cs_expression_vectorized("1e-3 * x + 2e-3 * y", panels)
        expected = panels["x"] * 1e-3 + panels["y"] * 2e-3
        pd.testing.assert_frame_equal(result, expected)

    def test_simple_variable_reference(self, evaluator, panels):
        result = evaluator._evaluate_cs_expression_vectorized("x", panels)
        pd.testing.assert_frame_equal(result, panels["x"])

    def test_missing_factor_returns_none_with_warning(self, evaluator, panels, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            result = evaluator._evaluate_cs_expression_vectorized("0.5 * x + 0.5 * z", panels)
        assert result is None

    def test_weighted_sum_fallback_to_function(self, evaluator, panels):
        """When weighted-sum parse fails, should fallback to function path."""
        # "-1 * normalize(...)" looks like weighted sum but 'normalize' is not a panel key
        # Should fall through to function path (which may also return None here,
        # but the fallback mechanism itself should work without error)
        result = evaluator._evaluate_cs_expression_vectorized("-1 * nonexistent_func(x)", panels)
        # Neither weighted-sum nor function parse succeeds → None
        assert result is None

    def test_is_cs_weighted_sum_ignores_scientific_notation(self, evaluator):
        assert evaluator._is_cs_weighted_sum("1e-3 * x") is False  # no binary +/- at depth 0
        assert evaluator._is_cs_weighted_sum("1e-3 * x + 2e-3 * y") is True
        assert evaluator._is_cs_weighted_sum("normalize(x, true, 0)") is False

    def test_is_cs_weighted_sum_leading_minus_not_binary(self, evaluator):
        # Leading '-' at i=0 should not be treated as binary operator
        assert evaluator._is_cs_weighted_sum("-x") is False


class TestVectorizedEvaluator:
    """Test VectorizedEvaluator expression evaluation."""

    def _make_data(self, n: int = 50, seed: int = 42) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.standard_normal(n).cumsum() + 100)

    def test_simple_arithmetic(self):
        close = self._make_data()
        volume = self._make_data(seed=99)
        ev = VectorizedEvaluator(
            variables={"close": close, "volume": volume},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("close + volume")
        result = ev.evaluate(ast)
        pd.testing.assert_series_equal(result, close + volume, check_names=False)

    def test_ts_mean_expression(self):
        close = self._make_data()
        ev = VectorizedEvaluator(
            variables={"close": close},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("ts_mean(close, 10)")
        result = ev.evaluate(ast)
        expected = close.rolling(10).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_composite_expression(self):
        close = self._make_data()
        ev = VectorizedEvaluator(
            variables={"close": close},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("ts_mean(close, 10) / ts_std(close, 10)")
        result = ev.evaluate(ast)
        expected = close.rolling(10).mean() / close.rolling(10).std(ddof=1)
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-10)

    def test_delta_expression(self):
        close = self._make_data()
        ev = VectorizedEvaluator(
            variables={"close": close},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("delta(close, 5)")
        result = ev.evaluate(ast)
        expected = close.diff(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_number_literal(self):
        close = self._make_data()
        ev = VectorizedEvaluator(
            variables={"close": close},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("close * 2.0")
        result = ev.evaluate(ast)
        pd.testing.assert_series_equal(result, close * 2.0, check_names=False)

    def test_math_operator(self):
        close = self._make_data()
        ev = VectorizedEvaluator(
            variables={"close": close},
            ts_operators=TS_OPERATOR_INSTANCES,
            math_operators=MATH_OPERATORS,
        )
        ast = parse_expression("log(close)")
        result = ev.evaluate(ast)
        expected = np.log(close)
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestParseSingleCsArgVec:
    """Test numeric parsing in _parse_single_cs_arg_vec."""

    @pytest.fixture()
    def evaluator(self):
        mock_config = MagicMock()
        return FactorEvaluator(mock_config)

    def test_integer_string(self, evaluator):
        result = evaluator._parse_single_cs_arg_vec("42", {})
        assert result == 42
        assert isinstance(result, int)

    def test_float_string(self, evaluator):
        result = evaluator._parse_single_cs_arg_vec("3.14", {})
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_scientific_notation(self, evaluator):
        result = evaluator._parse_single_cs_arg_vec("1e5", {})
        assert result == pytest.approx(1e5)
        assert isinstance(result, float)

    def test_scientific_notation_negative_exp(self, evaluator):
        result = evaluator._parse_single_cs_arg_vec("2.5e-3", {})
        assert result == pytest.approx(0.0025)
        assert isinstance(result, float)

    def test_bool_true(self, evaluator):
        assert evaluator._parse_single_cs_arg_vec("true", {}) is True

    def test_bool_false(self, evaluator):
        assert evaluator._parse_single_cs_arg_vec("false", {}) is False

    def test_variable_lookup(self, evaluator):
        panel = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = evaluator._parse_single_cs_arg_vec("my_factor", {"my_factor": panel})
        pd.testing.assert_frame_equal(result, panel)
