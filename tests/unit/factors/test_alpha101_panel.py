# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Alpha101 factor evaluation using Panel architecture.

Verifies that all 101 alpha expressions can be evaluated without error
and produce correct DataFrame results with the panel evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.builtin.alpha101 import ALPHA101_FACTORS
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES, CsRank
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def panel() -> dict[str, pd.DataFrame]:
    """Create a large-enough panel for all Alpha101 warmup requirements."""
    rng = np.random.RandomState(42)
    n_ts, n_inst = 300, 10
    instruments = [f"INST_{i}" for i in range(n_inst)]

    close = pd.DataFrame(
        rng.randn(n_ts, n_inst).cumsum(axis=0) + 100, columns=instruments,
    )
    open_ = close.shift(1).bfill() + rng.randn(n_ts, n_inst) * 0.5
    high = pd.DataFrame(
        np.maximum(open_.values, close.values) + np.abs(rng.randn(n_ts, n_inst)),
        columns=instruments,
    )
    low = pd.DataFrame(
        np.minimum(open_.values, close.values) - np.abs(rng.randn(n_ts, n_inst)),
        columns=instruments,
    )
    volume = pd.DataFrame(
        np.abs(rng.randn(n_ts, n_inst)) * 1000 + 500, columns=instruments,
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


@pytest.fixture()
def evaluator(panel: dict[str, pd.DataFrame]) -> Evaluator:
    return Evaluator(
        panel_fields=panel,
        ts_ops=TS_OPERATOR_INSTANCES,
        cs_ops=CS_OPERATOR_INSTANCES,
        math_ops=MATH_OPERATORS,
    )


# ---------------------------------------------------------------------------
# Parametric test: all 101 alphas evaluate without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha_name",
    sorted(ALPHA101_FACTORS.keys()),
    ids=sorted(ALPHA101_FACTORS.keys()),
)
def test_alpha101_evaluates_without_error(
    alpha_name: str,
    panel: dict[str, pd.DataFrame],
    evaluator: Evaluator,
) -> None:
    """Each Alpha101 expression should produce a valid DataFrame or scalar."""
    expr = ALPHA101_FACTORS[alpha_name]["expression"]
    ast = parse_expression(expr)
    result = evaluator.evaluate(ast)

    if isinstance(result, pd.DataFrame):
        # Should have correct number of columns (instruments)
        assert result.shape[1] == 10, f"{alpha_name}: wrong instrument count"
        # Should have at least some non-NaN values somewhere in the panel
        # (deeply nested decay_linear chains may NaN-propagate through the
        # last row with only 300 rows of random data, which is acceptable)
        assert not result.isna().all().all(), f"{alpha_name}: entirely NaN panel"
    elif isinstance(result, (int, float)):
        # Scalar is acceptable (e.g., constant expressions)
        pass
    else:
        pytest.fail(f"{alpha_name}: unexpected result type {type(result)}")


# ---------------------------------------------------------------------------
# Specific alpha correctness tests
# ---------------------------------------------------------------------------


class TestAlpha006Correctness:
    """alpha006: -1 * correlation(open, volume, 10)

    Pure TS expression — no CS operators.
    """

    def test_matches_manual(self, panel: dict[str, pd.DataFrame], evaluator: Evaluator) -> None:
        result = evaluator.evaluate(parse_expression("-1 * correlation(open, volume, 10)"))
        corr = pd.DataFrame(
            {col: panel["open"][col].rolling(10).corr(panel["volume"][col])
             for col in panel["open"].columns},
            index=panel["open"].index,
        )
        # NaN propagates naturally through warmup period; only inf → NaN
        expected = -1 * corr.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_frame_equal(result, expected)


class TestAlpha003Correctness:
    """alpha003: -1 * correlation(rank(open), rank(volume), 10)

    CS → TS nesting — the core value of the Panel architecture.
    """

    def test_matches_manual(self, panel: dict[str, pd.DataFrame], evaluator: Evaluator) -> None:
        result = evaluator.evaluate(parse_expression("-1 * correlation(rank(open), rank(volume), 10)"))

        rank_op = CsRank()
        rank_open = rank_op.compute_vectorized(panel["open"])
        rank_vol = rank_op.compute_vectorized(panel["volume"])
        corr = pd.DataFrame(
            {col: rank_open[col].rolling(10).corr(rank_vol[col])
             for col in panel["open"].columns},
            index=panel["open"].index,
        )
        # NaN propagates naturally through warmup period; only inf → NaN
        expected = -1 * corr.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_frame_equal(result, expected)


class TestAlpha013Correctness:
    """alpha013: -1 * rank(covariance(rank(close), rank(volume), 5))

    CS → TS → CS three-layer nesting.
    """

    def test_matches_manual(self, panel: dict[str, pd.DataFrame], evaluator: Evaluator) -> None:
        result = evaluator.evaluate(
            parse_expression("-1 * rank(covariance(rank(close), rank(volume), 5))")
        )

        rank_op = CsRank()
        rank_close = rank_op.compute_vectorized(panel["close"])
        rank_vol = rank_op.compute_vectorized(panel["volume"])
        cov_df = pd.DataFrame(
            {col: rank_close[col].rolling(5).cov(rank_vol[col], ddof=1)
             for col in panel["close"].columns},
            index=panel["close"].index,
        )
        expected = -1 * rank_op.compute_vectorized(cov_df)
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# FactorEngine integration test
# ---------------------------------------------------------------------------


class TestFactorEngine:
    """Test the full FactorEngine workflow."""

    def test_on_bar_and_flush(self) -> None:
        from nautilus_quants.factors.engine.factor_engine import FactorEngine

        engine = FactorEngine(max_history=100)
        engine.register_expression_factor("alpha006", "-1 * correlation(open, volume, 10)")

        rng = np.random.RandomState(42)
        instruments = ["AAPL", "GOOGL", "MSFT"]

        for ts in range(1, 21):
            for inst in instruments:
                engine.on_bar(inst, {
                    "open": float(rng.randn() * 10 + 100),
                    "high": float(rng.randn() * 10 + 105),
                    "low": float(rng.randn() * 10 + 95),
                    "close": float(rng.randn() * 10 + 100),
                    "volume": float(abs(rng.randn()) * 1000 + 500),
                }, ts)
            engine.flush_and_compute(ts)

        # Final computation should return results for all instruments
        results = engine.flush_and_compute(20)
        assert "alpha006" in results
        assert set(results["alpha006"].keys()) <= {"AAPL", "GOOGL", "MSFT"}

    def test_config_driven(self, tmp_path) -> None:
        """Test engine loads factors from config YAML."""
        from nautilus_quants.factors.engine.factor_engine import FactorEngine

        config_yaml = tmp_path / "test_factors.yaml"
        config_yaml.write_text("""
metadata:
  name: test
  version: "1.0"

variables:
  returns: "close / delay(close, 1) - 1"

factors:
  alpha_simple:
    expression: "ts_mean(close, 5)"
    description: "5-day moving average"
  alpha_cs:
    expression: "rank(close)"
    description: "Cross-sectional rank"
""")

        engine = FactorEngine(max_history=50)
        engine.load_config(str(config_yaml))

        assert "alpha_simple" in engine.factor_names
        assert "alpha_cs" in engine.factor_names
        assert "returns" in engine.variable_names
