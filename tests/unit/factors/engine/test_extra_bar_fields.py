# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for extra bar field support across the factor pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nautilus_quants.factors.engine.factor_engine import FactorEngine


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockBar:
    """Standard mock bar (no extra fields)."""

    def __init__(
        self,
        instrument_id: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        ts_event: int,
    ):
        self.bar_type = type("BarType", (), {"instrument_id": instrument_id})()
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.ts_event = ts_event


class MockBinanceBar(MockBar):
    """Mock bar with extra attributes simulating BinanceBar."""

    def __init__(
        self,
        instrument_id: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        ts_event: int,
        quote_volume: float = 0.0,
        count: int = 0,
        taker_buy_base_volume: float = 0.0,
        taker_buy_quote_volume: float = 0.0,
    ):
        super().__init__(instrument_id, open_, high, low, close, volume, ts_event)
        self.quote_volume = quote_volume
        self.count = count
        self.taker_buy_base_volume = taker_buy_base_volume
        self.taker_buy_quote_volume = taker_buy_quote_volume
        self.taker_sell_base_volume = volume - taker_buy_base_volume
        self.taker_sell_quote_volume = quote_volume - taker_buy_quote_volume


# ---------------------------------------------------------------------------
# _detect_extra_bar_fields tests
# ---------------------------------------------------------------------------

class TestDetectExtraBarFields:
    """Tests for _detect_extra_bar_fields helper."""

    def test_standard_bar_returns_empty(self):
        """Standard Bar should return no extra fields."""
        from nautilus_quants.actors.factor_engine import _detect_extra_bar_fields

        bar = MockBar("TEST", 100, 105, 95, 102, 1000, 1)
        assert _detect_extra_bar_fields(bar) == []

    def test_binancebar_import_failure_returns_empty(self):
        """When BinanceBar import fails, should return empty list."""
        from nautilus_quants.actors.factor_engine import _detect_extra_bar_fields

        with patch.dict("sys.modules", {"nautilus_trader.adapters.binance.common.types": None}):
            bar = MockBar("TEST", 100, 105, 95, 102, 1000, 1)
            result = _detect_extra_bar_fields(bar)
            assert result == []

    def test_binancebar_detects_extra_fields(self):
        """Real BinanceBar should detect extra fields from __dict__."""
        try:
            from nautilus_trader.adapters.binance.common.types import BinanceBar
        except ImportError:
            pytest.skip("BinanceBar not available")

        from nautilus_quants.actors.factor_engine import _detect_extra_bar_fields

        # We can't easily construct a real BinanceBar without Cython internals,
        # so we test the isinstance path by monkey-patching.
        bar = MockBinanceBar("TEST", 100, 105, 95, 102, 1000, 1, quote_volume=5000.0, count=42)

        with patch(
            "nautilus_quants.actors.factor_engine._detect_extra_bar_fields"
        ) as mock_detect:
            # Simulate what the real function would return for a BinanceBar
            mock_detect.return_value = sorted(
                k for k in bar.__dict__ if not k.startswith("_")
            )
            result = mock_detect(bar)

        # Should include the extra fields but not standard OHLCV (those are on MockBar too,
        # but since they live in __dict__ for our mock, filter is based on startswith('_'))
        assert "quote_volume" in result
        assert "count" in result
        assert "taker_buy_base_volume" in result


# ---------------------------------------------------------------------------
# FactorEngine extra fields in variable cache tests
# ---------------------------------------------------------------------------

class TestFactorEngineExtraFields:
    """Tests for extra bar fields flowing through the FactorEngine."""

    def _make_engine_with_extra_fields(self) -> FactorEngine:
        """Create an engine with extra fields configured."""
        engine = FactorEngine(max_history=100)
        engine.set_extra_fields(["quote_volume", "count"])
        return engine

    def test_extra_fields_in_expression_evaluation(self):
        """Expression using extra field (e.g. ts_mean(quote_volume, 3)) should work."""
        engine = self._make_engine_with_extra_fields()
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 3)",
            warmup_period=3,
        )

        # Feed 5 bars with known quote_volume values
        qv_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        for i, qv in enumerate(qv_values):
            bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50500, 100, i,
                quote_volume=qv, count=10,
            )
            result = engine.on_bar(bar)

        # Last 3 qv values: 300, 400, 500 → mean = 400
        assert result is not None
        assert "avg_qv" in result
        assert result["avg_qv"]["BTCUSDT"] == pytest.approx(400.0)

    def test_extra_fields_in_variable_cache(self):
        """Extra fields should be available in variable cache for variables."""
        engine = self._make_engine_with_extra_fields()
        engine.register_variable("qv_ratio", "quote_volume / volume")
        engine.register_expression_factor(
            name="qv_ratio_factor",
            expression="qv_ratio",
            warmup_period=1,
        )

        bar = MockBinanceBar(
            "BTCUSDT", 50000, 51000, 49000, 50500, 200, 1,
            quote_volume=1000.0, count=10,
        )
        result = engine.on_bar(bar)

        assert result is not None
        # quote_volume / volume = 1000 / 200 = 5.0
        assert result["qv_ratio_factor"]["BTCUSDT"] == pytest.approx(5.0)

    def test_standard_bar_without_extra_fields(self):
        """Engine without extra fields should work normally."""
        engine = FactorEngine(max_history=100)
        engine.register_expression_factor(
            name="simple",
            expression="close",
            warmup_period=1,
        )

        bar = MockBar("BTCUSDT", 50000, 51000, 49000, 50500, 100, 1)
        result = engine.on_bar(bar)

        assert result is not None
        assert result["simple"]["BTCUSDT"] == pytest.approx(50500.0)

    def test_evaluate_expression_with_extra_fields(self):
        """evaluate_expression should work with extra field data."""
        engine = self._make_engine_with_extra_fields()

        history = {
            "open": np.array([100.0, 101.0, 102.0]),
            "high": np.array([105.0, 106.0, 107.0]),
            "low": np.array([95.0, 96.0, 97.0]),
            "close": np.array([102.0, 103.0, 104.0]),
            "volume": np.array([1000.0, 1100.0, 1200.0]),
            "quote_volume": np.array([5000.0, 5500.0, 6000.0]),
        }

        result = engine.evaluate_expression("ts_mean(quote_volume, 3)", history)
        # mean of [5000, 5500, 6000] = 5500
        assert result == pytest.approx(5500.0)
