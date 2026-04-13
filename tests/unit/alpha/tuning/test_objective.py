# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for objective.py — CV folds + vectorised IC + helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.alpha.tuning.config import CVConfig
from nautilus_quants.alpha.tuning.objective import (
    CVFold,
    CVSchedule,
    _fold_icir,
    _spearman_ic_vectorized,
    build_cv_folds,
    compute_forward_returns_panel,
    evaluate_expression_panel,
)

# ── build_cv_folds ──────────────────────────────────────────────────────────


class TestBuildCVFolds:
    def test_default_schedule_has_three_expanding_folds(self) -> None:
        schedule = build_cv_folds(1000, CVConfig())
        assert len(schedule.folds) == 3
        assert all(f.train_start == 0 for f in schedule.folds)
        # Expanding window: train_end is strictly increasing.
        train_ends = [f.train_end for f in schedule.folds]
        assert train_ends == sorted(train_ends)

    def test_holdout_is_after_last_fold(self) -> None:
        schedule = build_cv_folds(1000, CVConfig())
        last_fold = schedule.folds[-1]
        assert schedule.holdout_start >= last_fold.test_end
        assert schedule.holdout_end <= 1000

    def test_rolling_window_uses_bounded_train(self) -> None:
        schedule = build_cv_folds(1000, CVConfig(method="rolling", n_folds=3, test_ratio=0.1))
        # Non-expanding: train_start should shift with test window.
        starts = [f.train_start for f in schedule.folds]
        assert starts[0] <= starts[1] <= starts[2]
        assert starts[-1] > 0  # rolling is strictly bounded

    def test_gap_bars_shrinks_train_end(self) -> None:
        schedule_no_gap = build_cv_folds(1000, CVConfig(gap_bars=0))
        schedule_gap = build_cv_folds(1000, CVConfig(gap_bars=10))
        for a, b in zip(schedule_no_gap.folds, schedule_gap.folds):
            assert b.train_end == a.train_end - 10

    def test_too_little_data_yields_empty_schedule(self) -> None:
        # 3 bars can't support meaningful fold splits even at default ratios.
        schedule = build_cv_folds(3, CVConfig())
        assert not schedule.is_usable()

    def test_empty_input_returns_empty_schedule(self) -> None:
        schedule = build_cv_folds(0, CVConfig())
        assert schedule.folds == ()
        assert schedule.total_timestamps == 0


# ── IC computation ─────────────────────────────────────────────────────────


class TestSpearmanIcVectorized:
    def test_perfect_positive_correlation(self) -> None:
        T, N = 50, 10
        idx = pd.date_range("2025-01-01", periods=T, freq="4h")
        cols = [f"I{i}" for i in range(N)]
        factor = pd.DataFrame(
            np.arange(T * N).reshape(T, N),
            index=idx,
            columns=cols,
            dtype=float,
        )
        # Returns = factor exactly -> rank correlation should be 1.
        ic = _spearman_ic_vectorized(factor, factor)
        assert (ic - 1.0).abs().max() < 1e-9

    def test_perfect_negative_correlation(self) -> None:
        T, N = 30, 8
        idx = pd.date_range("2025-01-01", periods=T, freq="4h")
        cols = [f"I{i}" for i in range(N)]
        factor = pd.DataFrame(
            np.random.RandomState(1).randn(T, N),
            index=idx,
            columns=cols,
        )
        ic = _spearman_ic_vectorized(factor, -factor)
        assert (ic + 1.0).abs().max() < 1e-9

    def test_unaligned_indices_are_intersected(self) -> None:
        idx_a = pd.date_range("2025-01-01", periods=20, freq="4h")
        idx_b = pd.date_range("2025-01-01 04:00", periods=20, freq="4h")
        factor = pd.DataFrame(
            np.random.RandomState(2).randn(20, 5),
            index=idx_a,
            columns=[f"I{i}" for i in range(5)],
        )
        returns = pd.DataFrame(
            np.random.RandomState(3).randn(20, 5),
            index=idx_b,
            columns=factor.columns,
        )
        ic = _spearman_ic_vectorized(factor, returns)
        # 19 shared timestamps.
        assert len(ic) <= 19
        assert len(ic) > 0

    def test_too_few_columns_yields_empty_series(self) -> None:
        # ``min_assets`` default is 5; if only 3 instruments are available
        # we expect an empty result.
        idx = pd.date_range("2025-01-01", periods=10, freq="4h")
        cols = ["A", "B", "C"]
        factor = pd.DataFrame(
            np.random.RandomState(4).randn(10, 3),
            index=idx,
            columns=cols,
        )
        returns = pd.DataFrame(
            np.random.RandomState(5).randn(10, 3),
            index=idx,
            columns=cols,
        )
        ic = _spearman_ic_vectorized(factor, returns)
        assert ic.empty


class TestFoldIcir:
    def test_mean_over_std(self) -> None:
        s = pd.Series([0.1, 0.2, 0.15, 0.05, 0.12] * 5)
        mean = s.mean()
        std = s.std(ddof=1)
        assert abs(_fold_icir(s) - mean / std) < 1e-9

    def test_zero_std_returns_zero(self) -> None:
        s = pd.Series([0.1] * 15)
        assert _fold_icir(s) == 0.0

    def test_insufficient_data_returns_nan(self) -> None:
        s = pd.Series([0.1, 0.2])
        assert np.isnan(_fold_icir(s))


# ── Forward returns ────────────────────────────────────────────────────────


class TestComputeForwardReturns:
    def test_one_period_shift(self) -> None:
        idx = pd.date_range("2025-01-01", periods=5, freq="4h")
        pricing = pd.DataFrame(
            {"I0": [100, 101, 102, 103, 104]},
            index=idx,
            dtype=float,
        )
        fwd = compute_forward_returns_panel(pricing, 1)
        # At t=0: (pricing[t+1] - pricing[t]) / pricing[t]
        expected = 1.0 / 100.0
        assert abs(fwd.iloc[0, 0] - expected) < 1e-9
        # Last row has no t+1 sample.
        assert np.isnan(fwd.iloc[-1, 0])

    def test_empty_panel(self) -> None:
        pricing = pd.DataFrame()
        result = compute_forward_returns_panel(pricing, 1)
        assert result.empty


# ── evaluate_expression_panel ──────────────────────────────────────────────


class TestEvaluateExpressionPanel:
    def test_simple_rolling_mean(self) -> None:
        T, N = 30, 5
        idx = pd.date_range("2025-01-01", periods=T, freq="4h")
        cols = [f"I{i}" for i in range(N)]
        close = pd.DataFrame(np.arange(T * N, dtype=float).reshape(T, N), index=idx, columns=cols)
        panel = {"close": close}
        result = evaluate_expression_panel("ts_mean(close, 3)", panel)
        assert isinstance(result, pd.DataFrame)
        # The rolling mean of the last window at t=T-1 should be the mean of
        # the last three close rows.
        expected = close.iloc[-3:].mean()
        pd.testing.assert_series_equal(
            result.iloc[-1],
            expected,
            check_names=False,
        )

    def test_parameters_are_respected(self) -> None:
        T, N = 20, 3
        idx = pd.date_range("2025-01-01", periods=T, freq="4h")
        cols = ["I0", "I1", "I2"]
        close = pd.DataFrame(np.random.RandomState(9).randn(T, N), index=idx, columns=cols)
        panel = {"close": close, "w": 5.0}
        result = evaluate_expression_panel("ts_mean(close, w)", panel, parameters={"w": 5})
        assert isinstance(result, pd.DataFrame)
