# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for incremental operators (TDD RED phase).

These tests verify that incremental O(1) implementations match
the batch O(window) implementations at every time step.

Written FIRST — will FAIL until IncrementalMean/Std/Delay/Corr are implemented.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nautilus_quants.factors.operators.time_series import (
    Correlation,
    Delta,
    Delay,
    IncrementalCorr,
    IncrementalDelay,
    IncrementalDelta,
    IncrementalMean,
    IncrementalStd,
    TsMean,
    TsStd,
)


# ---------------------------------------------------------------------------
# IncrementalMean
# ---------------------------------------------------------------------------


class TestIncrementalMean:
    """Tests for IncrementalMean incremental rolling mean."""

    def test_warmup_returns_nan(self):
        """Returns NaN until window is filled."""
        inc = IncrementalMean(3)
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))
        assert not math.isnan(inc.push(3.0))

    def test_basic_mean(self):
        """Mean of first window."""
        inc = IncrementalMean(3)
        inc.push(1.0)
        inc.push(2.0)
        result = inc.push(3.0)
        assert result == pytest.approx(2.0)

    def test_sliding_window(self):
        """Window slides correctly after warmup."""
        inc = IncrementalMean(3)
        for v in [1.0, 2.0, 3.0]:
            inc.push(v)
        result = inc.push(4.0)
        assert result == pytest.approx(3.0)  # mean(2, 3, 4)

    def test_window_1(self):
        """Window=1 returns the last pushed value."""
        inc = IncrementalMean(1)
        assert inc.push(42.0) == pytest.approx(42.0)
        assert inc.push(99.0) == pytest.approx(99.0)

    def test_reset_clears_state(self):
        """After reset, warmup starts again."""
        inc = IncrementalMean(3)
        for v in [1.0, 2.0, 3.0]:
            inc.push(v)
        inc.reset()
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))
        result = inc.push(3.0)
        assert result == pytest.approx(2.0)

    @pytest.mark.parametrize("window", [3, 10, 24, 96])
    @pytest.mark.parametrize("n_bars", [50, 200, 500])
    def test_matches_batch_at_every_step(self, window: int, n_bars: int):
        """IncrementalMean.push() matches TsMean.compute() at every time step."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(n_bars)
        inc = IncrementalMean(window)
        batch_op = TsMean()

        for i in range(len(data)):
            inc_result = inc.push(float(data[i]))
            batch_result = float(batch_op.compute(data[: i + 1], window))

            if math.isnan(batch_result):
                assert math.isnan(inc_result), (
                    f"window={window}, step={i}: expected nan, got {inc_result}"
                )
            else:
                assert inc_result == pytest.approx(batch_result, abs=1e-10), (
                    f"window={window}, step={i}: inc={inc_result}, batch={batch_result}"
                )


# ---------------------------------------------------------------------------
# IncrementalStd
# ---------------------------------------------------------------------------


class TestIncrementalStd:
    """Tests for IncrementalStd incremental rolling standard deviation (ddof=1)."""

    def test_warmup_returns_nan(self):
        """Returns NaN until window is filled."""
        inc = IncrementalStd(3)
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))
        assert not math.isnan(inc.push(3.0))

    def test_basic_std(self):
        """Std of [1, 2, 3] with ddof=1."""
        inc = IncrementalStd(3)
        inc.push(1.0)
        inc.push(2.0)
        result = inc.push(3.0)
        expected = float(np.std([1.0, 2.0, 3.0], ddof=1))
        assert result == pytest.approx(expected)

    def test_constant_returns_zero(self):
        """Std of constant window is zero."""
        inc = IncrementalStd(4)
        for _ in range(5):
            result = inc.push(5.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_reset_clears_state(self):
        """After reset, warmup starts again."""
        inc = IncrementalStd(3)
        for v in [1.0, 2.0, 3.0]:
            inc.push(v)
        inc.reset()
        assert math.isnan(inc.push(1.0))

    @pytest.mark.parametrize("window", [3, 10, 24])
    @pytest.mark.parametrize("n_bars", [50, 200, 500])
    def test_matches_batch_at_every_step(self, window: int, n_bars: int):
        """IncrementalStd.push() matches TsStd.compute() at every time step."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(n_bars)
        inc = IncrementalStd(window)
        batch_op = TsStd()

        for i in range(len(data)):
            inc_result = inc.push(float(data[i]))
            batch_result = float(batch_op.compute(data[: i + 1], window))

            if math.isnan(batch_result):
                assert math.isnan(inc_result), (
                    f"window={window}, step={i}: expected nan, got {inc_result}"
                )
            else:
                assert inc_result == pytest.approx(batch_result, abs=1e-8), (
                    f"window={window}, step={i}: inc={inc_result:.10f}, batch={batch_result:.10f}"
                )


# ---------------------------------------------------------------------------
# IncrementalDelay
# ---------------------------------------------------------------------------


class TestIncrementalDelay:
    """Tests for IncrementalDelay incremental lagged value."""

    def test_warmup_returns_nan(self):
        """Returns NaN until lag+1 values have been seen."""
        inc = IncrementalDelay(2)
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))
        result = inc.push(3.0)
        assert not math.isnan(result)

    def test_basic_delay(self):
        """Lag=2: third push returns first pushed value."""
        inc = IncrementalDelay(2)
        inc.push(10.0)
        inc.push(20.0)
        result = inc.push(30.0)
        assert result == pytest.approx(10.0)

    def test_delay_1(self):
        """Lag=1: second push returns first pushed value."""
        inc = IncrementalDelay(1)
        inc.push(10.0)
        result = inc.push(20.0)
        assert result == pytest.approx(10.0)

    def test_sliding(self):
        """Sliding lag returns correct historical value."""
        inc = IncrementalDelay(2)
        for v in [10.0, 20.0, 30.0]:
            inc.push(v)
        result = inc.push(40.0)
        assert result == pytest.approx(20.0)  # value from 2 steps ago

    def test_reset_clears_state(self):
        """After reset, warmup starts again."""
        inc = IncrementalDelay(2)
        for v in [1.0, 2.0, 3.0]:
            inc.push(v)
        inc.reset()
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))

    @pytest.mark.parametrize("lag", [1, 3, 24])
    @pytest.mark.parametrize("n_bars", [50, 200])
    def test_matches_batch_at_every_step(self, lag: int, n_bars: int):
        """IncrementalDelay.push() matches Delay.compute() at every time step."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(n_bars)
        inc = IncrementalDelay(lag)
        batch_op = Delay()

        for i in range(len(data)):
            inc_result = inc.push(float(data[i]))
            batch_result = float(batch_op.compute(data[: i + 1], lag))

            if math.isnan(batch_result):
                assert math.isnan(inc_result), (
                    f"lag={lag}, step={i}: expected nan, got {inc_result}"
                )
            else:
                assert inc_result == pytest.approx(batch_result, abs=1e-10), (
                    f"lag={lag}, step={i}: inc={inc_result}, batch={batch_result}"
                )


# ---------------------------------------------------------------------------
# IncrementalDelta
# ---------------------------------------------------------------------------


class TestIncrementalDelta:
    """Tests for IncrementalDelta incremental x[t] - x[t-lag]."""

    def test_warmup_returns_nan(self):
        """Returns NaN until lag+1 values have been seen."""
        inc = IncrementalDelta(2)
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))
        result = inc.push(3.0)
        assert not math.isnan(result)

    def test_basic_delta(self):
        """lag=2: delta(3) = 3 - 1 = 2."""
        inc = IncrementalDelta(2)
        inc.push(1.0)
        inc.push(2.0)
        result = inc.push(3.0)
        assert result == pytest.approx(2.0)

    def test_delta_1(self):
        """lag=1: each result is x[t] - x[t-1]."""
        inc = IncrementalDelta(1)
        inc.push(10.0)
        result = inc.push(15.0)
        assert result == pytest.approx(5.0)

    def test_reset_clears_state(self):
        """After reset, warmup starts again."""
        inc = IncrementalDelta(2)
        for v in [1.0, 2.0, 3.0]:
            inc.push(v)
        inc.reset()
        assert math.isnan(inc.push(1.0))
        assert math.isnan(inc.push(2.0))

    @pytest.mark.parametrize("lag", [1, 3, 24])
    @pytest.mark.parametrize("n_bars", [50, 200])
    def test_matches_batch_at_every_step(self, lag: int, n_bars: int):
        """IncrementalDelta.push() matches Delta.compute() at every time step."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(n_bars)
        inc = IncrementalDelta(lag)
        batch_op = Delta()

        for i in range(len(data)):
            inc_result = inc.push(float(data[i]))
            batch_result = float(batch_op.compute(data[: i + 1], lag))

            if math.isnan(batch_result):
                assert math.isnan(inc_result), (
                    f"lag={lag}, step={i}: expected nan, got {inc_result}"
                )
            else:
                assert inc_result == pytest.approx(batch_result, abs=1e-10), (
                    f"lag={lag}, step={i}: inc={inc_result}, batch={batch_result}"
                )


# ---------------------------------------------------------------------------
# IncrementalCorr
# ---------------------------------------------------------------------------


class TestIncrementalCorr:
    """Tests for IncrementalCorr incremental Pearson correlation."""

    def test_warmup_returns_nan(self):
        """Returns NaN until window is filled."""
        inc = IncrementalCorr(3)
        assert math.isnan(inc.push(1.0, 1.0))
        assert math.isnan(inc.push(2.0, 2.0))
        result = inc.push(3.0, 3.0)
        assert not math.isnan(result)

    def test_constant_x_returns_nan(self):
        """Constant x series → correlation is undefined (nan)."""
        inc = IncrementalCorr(3)
        for _ in range(4):
            result = inc.push(5.0, float(_ + 1))
        assert math.isnan(result)

    def test_constant_y_returns_nan(self):
        """Constant y series → correlation is undefined (nan)."""
        inc = IncrementalCorr(3)
        for i in range(4):
            result = inc.push(float(i + 1), 5.0)
        assert math.isnan(result)

    def test_perfect_positive_correlation(self):
        """y = 2x + 1 → r == 1.0."""
        inc = IncrementalCorr(4)
        result = float('nan')
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            result = inc.push(x, 2.0 * x + 1.0)
        assert result == pytest.approx(1.0, abs=1e-8)

    def test_perfect_negative_correlation(self):
        """y = -x → r == -1.0."""
        inc = IncrementalCorr(4)
        result = float('nan')
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            result = inc.push(x, -x)
        assert result == pytest.approx(-1.0, abs=1e-8)

    def test_reset_clears_state(self):
        """After reset, warmup starts again."""
        inc = IncrementalCorr(3)
        for x, y in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
            inc.push(x, y)
        inc.reset()
        assert math.isnan(inc.push(1.0, 2.0))
        assert math.isnan(inc.push(2.0, 4.0))

    @pytest.mark.parametrize("window", [3, 10, 24, 96])
    @pytest.mark.parametrize("n_bars", [150, 300])
    def test_matches_batch_at_every_step(self, window: int, n_bars: int):
        """IncrementalCorr.push() matches Correlation.compute() at every time step."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n_bars)
        y = rng.standard_normal(n_bars)
        inc = IncrementalCorr(window)
        batch_op = Correlation()

        for i in range(len(x)):
            inc_result = inc.push(float(x[i]), float(y[i]))
            batch_result = float(
                batch_op.compute(x[: i + 1], window, data2=y[: i + 1])
            )

            if math.isnan(batch_result):
                assert math.isnan(inc_result), (
                    f"window={window}, step={i}: expected nan, got {inc_result}"
                )
            else:
                assert inc_result == pytest.approx(batch_result, abs=1e-8), (
                    f"window={window}, step={i}: "
                    f"inc={inc_result:.8f}, batch={batch_result:.8f}"
                )
