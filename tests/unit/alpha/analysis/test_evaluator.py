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
        assert evaluator._factor_config is not None

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


class TestPanelEvaluatorIntegration:
    """Test that PanelEvaluator (which replaced CS expression methods) works correctly."""

    def test_weighted_sum_via_panel_evaluator(self):
        """PanelEvaluator evaluates weighted-sum expressions correctly."""
        from nautilus_quants.factors.engine.panel_evaluator import PanelEvaluator
        from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES

        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        panel = {
            "x": pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=idx, dtype=float),
            "y": pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [50, 40, 30, 20, 10]}, index=idx, dtype=float),
        }
        evaluator = PanelEvaluator(
            panel_fields=panel,
            ts_ops=TS_OPERATOR_INSTANCES,
            cs_ops=CS_OPERATOR_INSTANCES,
            math_ops=MATH_OPERATORS,
        )
        result = evaluator.evaluate(parse_expression("0.6 * x + 0.4 * y"))
        expected = panel["x"] * 0.6 + panel["y"] * 0.4
        pd.testing.assert_frame_equal(result, expected)

    def test_simple_variable_reference(self):
        """PanelEvaluator resolves simple variable references."""
        from nautilus_quants.factors.engine.panel_evaluator import PanelEvaluator
        from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES

        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        panel = {
            "x": pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=idx, dtype=float),
        }
        evaluator = PanelEvaluator(
            panel_fields=panel,
            ts_ops=TS_OPERATOR_INSTANCES,
            cs_ops=CS_OPERATOR_INSTANCES,
            math_ops=MATH_OPERATORS,
        )
        result = evaluator.evaluate(parse_expression("x"))
        pd.testing.assert_frame_equal(result, panel["x"])


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


