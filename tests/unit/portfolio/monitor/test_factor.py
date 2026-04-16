# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for portfolio/monitor/factor.py — factor health + realized IC."""

from __future__ import annotations

import pytest

from nautilus_quants.portfolio.monitor.factor import (
    compute_factor_health,
    compute_factor_ic,
)


# ---------------------------------------------------------------------------
# compute_factor_health
# ---------------------------------------------------------------------------


class TestComputeFactorHealth:
    """Tests for compute_factor_health pure function."""

    def test_normal_values(self) -> None:
        """All values valid → anomaly_count=0, metrics computed."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": 0.5,
                "ETHUSDT.OKX": -0.3,
                "SOLUSDT.OKX": 0.1,
            },
        }
        result = compute_factor_health(current, previous=None)

        alpha = result["alpha001"]
        assert alpha["anomaly_count"] == 0
        assert alpha["anomaly_rate"] == 0.0
        assert alpha["anomaly_instruments"] == []
        assert alpha["instrument_count"] == 3
        assert alpha["dispersion"] > 0
        assert isinstance(alpha["kurtosis"], float)
        # No previous → no stuck_count / turnover
        assert "stuck_count" not in alpha
        assert "turnover" not in alpha

    def test_inf_detected_as_anomaly(self) -> None:
        """inf values are counted as anomalies."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": 0.5,
                "ETHUSDT.OKX": float("inf"),
                "SOLUSDT.OKX": 0.1,
            },
        }
        result = compute_factor_health(current, previous=None)

        alpha = result["alpha001"]
        assert alpha["anomaly_count"] == 1
        assert alpha["anomaly_instruments"] == ["ETHUSDT.OKX"]
        assert alpha["instrument_count"] == 2  # only valid ones

    def test_neg_inf_detected_as_anomaly(self) -> None:
        """-inf values are also anomalies."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": float("-inf"),
                "ETHUSDT.OKX": 0.3,
            },
        }
        result = compute_factor_health(current, previous=None)

        alpha = result["alpha001"]
        assert alpha["anomaly_count"] == 1
        assert "BTCUSDT.OKX" in alpha["anomaly_instruments"]

    def test_missing_instruments_detected(self) -> None:
        """Instruments present in other factors but missing → anomaly."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": 0.5,
                "ETHUSDT.OKX": 0.3,
                "SOLUSDT.OKX": 0.1,
            },
            "alpha002": {
                "BTCUSDT.OKX": 0.2,
                # ETHUSDT and SOLUSDT missing → anomaly for alpha002
            },
        }
        result = compute_factor_health(current, previous=None)

        assert result["alpha001"]["anomaly_count"] == 0
        assert result["alpha002"]["anomaly_count"] == 2
        assert sorted(result["alpha002"]["anomaly_instruments"]) == [
            "ETHUSDT.OKX",
            "SOLUSDT.OKX",
        ]

    def test_with_previous_computes_stuck_and_turnover(self) -> None:
        """When previous is provided, stuck_count and turnover are computed."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": 0.5,
                "ETHUSDT.OKX": -0.3,
                "SOLUSDT.OKX": 0.1,
            },
        }
        previous = {
            "alpha001": {
                "BTCUSDT.OKX": 0.5,  # stuck
                "ETHUSDT.OKX": 0.2,  # changed
                "SOLUSDT.OKX": 0.1,  # stuck
            },
        }
        result = compute_factor_health(current, previous)

        alpha = result["alpha001"]
        assert alpha["stuck_count"] == 2  # BTC and SOL unchanged
        assert "turnover" in alpha
        assert 0.0 <= alpha["turnover"] <= 2.0  # valid range

    def test_empty_factors(self) -> None:
        """Empty factors dict → empty result."""
        result = compute_factor_health({}, previous=None)
        assert result == {}

    def test_single_instrument(self) -> None:
        """Single instrument → dispersion=0.0."""
        current = {"alpha001": {"BTCUSDT.OKX": 0.5}}
        result = compute_factor_health(current, previous=None)

        assert result["alpha001"]["dispersion"] == 0.0
        assert result["alpha001"]["instrument_count"] == 1

    def test_all_anomalies(self) -> None:
        """All values are inf → anomaly_rate=1.0, instrument_count=0."""
        current = {
            "alpha001": {
                "BTCUSDT.OKX": float("inf"),
                "ETHUSDT.OKX": float("-inf"),
            },
        }
        result = compute_factor_health(current, previous=None)

        alpha = result["alpha001"]
        assert alpha["anomaly_count"] == 2
        assert alpha["anomaly_rate"] == 1.0
        assert alpha["instrument_count"] == 0
        assert alpha["dispersion"] == 0.0

    def test_empty_factor_values(self) -> None:
        """Factor with no instrument values → counted with anomalies from all_instruments."""
        current = {
            "alpha001": {"BTCUSDT.OKX": 0.5},
            "alpha002": {},  # empty
        }
        result = compute_factor_health(current, previous=None)

        assert result["alpha002"]["anomaly_count"] == 1  # BTCUSDT missing
        assert result["alpha002"]["instrument_count"] == 0

    def test_previous_factor_missing_in_current(self) -> None:
        """Factor exists in previous but not in current → not compared."""
        current = {"alpha001": {"BTCUSDT.OKX": 0.5}}
        previous = {
            "alpha001": {"BTCUSDT.OKX": 0.5},
            "alpha002": {"BTCUSDT.OKX": 0.3},  # not in current
        }
        result = compute_factor_health(current, previous)

        assert "alpha001" in result
        assert "alpha002" not in result


# ---------------------------------------------------------------------------
# compute_factor_ic
# ---------------------------------------------------------------------------


class TestComputeFactorIc:
    """Tests for compute_factor_ic pure function."""

    def test_perfect_positive_ic(self) -> None:
        """Factors perfectly predict returns → IC close to 1."""
        prev_factors = {
            "alpha001": {
                "A": 1.0,
                "B": 2.0,
                "C": 3.0,
                "D": 4.0,
                "E": 5.0,
            },
        }
        # Returns perfectly correlated with factor
        close_prev = {"A": 100, "B": 100, "C": 100, "D": 100, "E": 100}
        close_cur = {"A": 101, "B": 102, "C": 103, "D": 104, "E": 105}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "alpha001" in result
        assert result["alpha001"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_ic(self) -> None:
        """Factors inversely predict returns → IC close to -1."""
        prev_factors = {
            "alpha001": {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0},
        }
        close_prev = {"A": 100, "B": 100, "C": 100, "D": 100, "E": 100}
        close_cur = {"A": 101, "B": 102, "C": 103, "D": 104, "E": 105}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert result["alpha001"] == pytest.approx(-1.0, abs=0.01)

    def test_insufficient_instruments_skipped(self) -> None:
        """Less than 3 common instruments → factor skipped."""
        prev_factors = {"alpha001": {"A": 1.0, "B": 2.0}}
        close_prev = {"A": 100, "B": 100}
        close_cur = {"A": 101, "B": 102}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert result == {}

    def test_zero_close_price_skipped(self) -> None:
        """Instruments with zero previous close → excluded from return calc."""
        prev_factors = {
            "alpha001": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
        }
        close_prev = {"A": 100, "B": 0, "C": 100, "D": 100}  # B has zero
        close_cur = {"A": 101, "B": 50, "C": 103, "D": 104}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "alpha001" in result  # 3 valid instruments (A, C, D)

    def test_inf_factor_values_excluded(self) -> None:
        """Instruments with inf factor values → excluded."""
        prev_factors = {
            "alpha001": {
                "A": 1.0,
                "B": float("inf"),
                "C": 3.0,
                "D": 4.0,
            },
        }
        close_prev = {"A": 100, "B": 100, "C": 100, "D": 100}
        close_cur = {"A": 101, "B": 102, "C": 103, "D": 104}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "alpha001" in result  # 3 valid (A, C, D)

    def test_empty_prev_factors(self) -> None:
        """Empty prev_factors → empty result."""
        result = compute_factor_ic({}, {"A": 101}, {"A": 100})
        assert result == {}

    def test_no_common_instruments(self) -> None:
        """No overlap between factors and close → empty result."""
        prev_factors = {"alpha001": {"X": 1.0, "Y": 2.0, "Z": 3.0}}
        close_prev = {"A": 100, "B": 100, "C": 100}
        close_cur = {"A": 101, "B": 102, "C": 103}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert result == {}

    def test_multiple_factors(self) -> None:
        """Multiple factors → IC computed for each independently."""
        prev_factors = {
            "alpha001": {"A": 1.0, "B": 2.0, "C": 3.0},
            "alpha002": {"A": 3.0, "B": 2.0, "C": 1.0},
        }
        close_prev = {"A": 100, "B": 100, "C": 100}
        close_cur = {"A": 101, "B": 102, "C": 103}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "alpha001" in result
        assert "alpha002" in result
        # alpha001 should have positive IC, alpha002 negative
        assert result["alpha001"] > 0
        assert result["alpha002"] < 0

    def test_ic_in_valid_range(self) -> None:
        """IC values always in [-1, 1]."""
        prev_factors = {
            "alpha001": {"A": 0.5, "B": -0.3, "C": 0.8, "D": -0.1},
        }
        close_prev = {"A": 100, "B": 200, "C": 150, "D": 300}
        close_cur = {"A": 105, "B": 198, "C": 155, "D": 295}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        if "alpha001" in result:
            assert -1.0 <= result["alpha001"] <= 1.0

    def test_stuck_factor_returns_no_ic(self) -> None:
        """A factor with constant values produces no IC entry (NaN rank corr)."""
        prev_factors = {"stuck": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}}
        close_prev = {"A": 100, "B": 100, "C": 100, "D": 100}
        close_cur = {"A": 101, "B": 102, "C": 103, "D": 104}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "stuck" not in result

    def test_tied_values_handled_correctly(self) -> None:
        """Tied factor values use average ranks, not ordinal ranks."""
        # [1,1,2] vs [1,2,3]: ordinal gives 1.0, average ranks give ~0.866
        prev_factors = {"alpha001": {"A": 1.0, "B": 1.0, "C": 2.0}}
        close_prev = {"A": 100, "B": 100, "C": 100}
        close_cur = {"A": 101, "B": 102, "C": 103}

        result = compute_factor_ic(prev_factors, close_cur, close_prev)

        assert "alpha001" in result
        # With average ranks: should be ~0.866, NOT 1.0
        assert result["alpha001"] < 1.0
        assert result["alpha001"] == pytest.approx(0.866, abs=0.01)
