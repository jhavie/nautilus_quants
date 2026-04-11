# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for fill_nan operator and composite nan_policy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.operators.math import FillNan, fill_nan
from nautilus_quants.factors.config import (
    FactorDefinition,
    _build_composite_pipeline,
    _NEUTRAL_VALUES,
)
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.expression import parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES


# ---------------------------------------------------------------------------
# fill_nan operator
# ---------------------------------------------------------------------------


class TestFillNanScalar:
    def test_nan_replaced(self) -> None:
        assert fill_nan(float("nan"), 0.0) == 0.0

    def test_nan_replaced_nonzero(self) -> None:
        assert fill_nan(float("nan"), 0.5) == 0.5

    def test_valid_unchanged(self) -> None:
        assert fill_nan(3.14, 0.0) == 3.14

    def test_zero_unchanged(self) -> None:
        assert fill_nan(0.0, 99.0) == 0.0


class TestFillNanArray:
    def test_numpy_array(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, np.nan])
        result = fill_nan(arr, 0.0)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0, 0.0])

    def test_original_not_mutated(self) -> None:
        arr = np.array([1.0, np.nan])
        fill_nan(arr, 0.0)
        assert np.isnan(arr[1])


class TestFillNanDataFrame:
    def test_dataframe(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan], "B": [np.nan, 2.0]})
        result = FillNan().compute(df, 0.0)
        expected = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 2.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_series(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0])
        result = FillNan().compute(s, -1.0)
        expected = pd.Series([1.0, -1.0, 3.0])
        pd.testing.assert_series_equal(result, expected)


class TestFillNanInExpression:
    """fill_nan works when called through the evaluator."""

    def test_evaluator_fill_nan(self) -> None:
        instruments = [f"INST_{i}" for i in range(5)]
        df = pd.DataFrame(
            [[1.0, np.nan, 3.0, np.nan, 5.0]],
            columns=instruments,
        )
        panel: dict[str, Any] = {"x": df}
        evaluator = Evaluator(
            panel_fields=panel,
            ts_ops=TS_OPERATOR_INSTANCES,
            cs_ops=CS_OPERATOR_INSTANCES,
            math_ops=MATH_OPERATORS,
        )
        result = evaluator.evaluate(parse_expression("fill_nan(x, 0)"))
        assert isinstance(result, pd.DataFrame)
        last_row = result.iloc[-1]
        assert last_row.notna().all()
        assert last_row["INST_1"] == 0.0
        assert last_row["INST_0"] == 1.0


# ---------------------------------------------------------------------------
# _build_composite_pipeline with nan_policy
# ---------------------------------------------------------------------------


class TestBuildCompositePipelineNanPolicy:
    """Verify pipeline generation with nan_policy=fill_neutral."""

    def test_strict_default_no_fill(self) -> None:
        """Default nan_policy=strict: no fill_nan in pipeline."""
        raw = {
            "transform": "normalize",
            "weights": {"f1": 0.6, "f2": 0.4},
        }
        pipeline = _build_composite_pipeline(raw)
        names = [f.name for f in pipeline]
        assert "f1_filled" not in names
        assert "f2_filled" not in names

    def test_fill_neutral_normalize(self) -> None:
        """nan_policy=fill_neutral with normalize: fills with 0.0."""
        raw = {
            "transform": "normalize",
            "nan_policy": "fill_neutral",
            "weights": {"f1": 0.6, "f2": 0.4},
        }
        pipeline = _build_composite_pipeline(raw)
        names = [f.name for f in pipeline]
        # Should have: f1_norm, f1_filled, f2_norm, f2_filled, composite
        assert "f1_norm" in names
        assert "f1_filled" in names
        assert "f2_filled" in names

        f1_filled = next(f for f in pipeline if f.name == "f1_filled")
        assert f1_filled.expression == "fill_nan(f1_norm, 0.0)"

    def test_fill_neutral_cs_rank(self) -> None:
        """nan_policy=fill_neutral with cs_rank: fills with 0.5."""
        raw = {
            "transform": "cs_rank",
            "nan_policy": "fill_neutral",
            "weights": {"f1": 0.5, "f2": 0.5},
        }
        pipeline = _build_composite_pipeline(raw)
        f1_filled = next(f for f in pipeline if f.name == "f1_filled")
        assert f1_filled.expression == "fill_nan(f1_ranked, 0.5)"

    def test_fill_neutral_raw(self) -> None:
        """nan_policy=fill_neutral with raw: fills with 0.0."""
        raw = {
            "transform": "raw",
            "nan_policy": "fill_neutral",
            "weights": {"f1": 0.5, "f2": 0.5},
        }
        pipeline = _build_composite_pipeline(raw)
        f1_filled = next(f for f in pipeline if f.name == "f1_filled")
        assert f1_filled.expression == "fill_nan(f1, 0.0)"

    def test_composite_expr_uses_filled(self) -> None:
        """Composite expression references _filled names, not _norm."""
        raw = {
            "transform": "normalize",
            "nan_policy": "fill_neutral",
            "weights": {"f1": 0.6, "f2": 0.4},
        }
        pipeline = _build_composite_pipeline(raw)
        composite = pipeline[-1]
        assert composite.name == "composite"
        assert "f1_filled" in composite.expression
        assert "f2_filled" in composite.expression
        assert "f1_norm" not in composite.expression


# ---------------------------------------------------------------------------
# End-to-end: composite NaN tolerance through evaluator
# ---------------------------------------------------------------------------


class TestCompositeNanTolerance:
    """Verify that fill_neutral actually prevents NaN poisoning."""

    @staticmethod
    def _make_panel_with_nan() -> dict[str, pd.DataFrame]:
        """Panel where INST_0 has NaN for factor f2, others are valid."""
        instruments = [f"INST_{i}" for i in range(6)]
        rng = np.random.RandomState(99)
        n_ts = 3

        f1_data = rng.randn(n_ts, 6)
        f2_data = rng.randn(n_ts, 6)
        # INST_0 has NaN in f2
        f2_data[:, 0] = np.nan

        return {
            "f1": pd.DataFrame(f1_data, columns=instruments),
            "f2": pd.DataFrame(f2_data, columns=instruments),
        }

    @staticmethod
    def _make_evaluator(panel: dict[str, pd.DataFrame]) -> Evaluator:
        """Create evaluator sharing the same panel dict (engine-like injection)."""
        return Evaluator(
            panel_fields=panel,
            ts_ops=TS_OPERATOR_INSTANCES,
            cs_ops=CS_OPERATOR_INSTANCES,
            math_ops=MATH_OPERATORS,
        )

    def test_strict_nan_poisons_composite(self) -> None:
        """Without fill_neutral, INST_0 is NaN in composite."""
        panel = self._make_panel_with_nan()
        evaluator = self._make_evaluator(panel)
        # Simulate strict pipeline: normalize then add (inject back into panel)
        panel["f1_norm"] = evaluator.evaluate(
            parse_expression("normalize(f1, true, 0)"),
        )
        panel["f2_norm"] = evaluator.evaluate(
            parse_expression("normalize(f2, true, 0)"),
        )
        result = evaluator.evaluate(
            parse_expression("0.6 * f1_norm + 0.4 * f2_norm"),
        )
        last_row = result.iloc[-1]
        # INST_0 should be NaN (poisoned by f2)
        assert math.isnan(last_row["INST_0"])
        # Others should be valid
        assert last_row.iloc[1:].notna().all()

    def test_fill_neutral_prevents_poisoning(self) -> None:
        """With fill_neutral, INST_0 gets neutral fill and stays valid."""
        panel = self._make_panel_with_nan()
        evaluator = self._make_evaluator(panel)
        # Simulate fill_neutral pipeline (inject back into panel)
        panel["f1_norm"] = evaluator.evaluate(
            parse_expression("normalize(f1, true, 0)"),
        )
        panel["f2_norm"] = evaluator.evaluate(
            parse_expression("normalize(f2, true, 0)"),
        )
        panel["f1_filled"] = evaluator.evaluate(
            parse_expression("fill_nan(f1_norm, 0)"),
        )
        panel["f2_filled"] = evaluator.evaluate(
            parse_expression("fill_nan(f2_norm, 0)"),
        )
        result = evaluator.evaluate(
            parse_expression("0.6 * f1_filled + 0.4 * f2_filled"),
        )
        last_row = result.iloc[-1]
        # ALL instruments should be valid now
        assert last_row.notna().all(), f"NaN found: {last_row[last_row.isna()].index.tolist()}"
        # INST_0's composite = 0.6 * f1_norm_value + 0.4 * 0 (neutral)
        # So it should be purely driven by f1
        inst0_f1 = panel["f1_norm"].iloc[-1]["INST_0"]
        assert last_row["INST_0"] == pytest.approx(0.6 * inst0_f1, abs=1e-10)
