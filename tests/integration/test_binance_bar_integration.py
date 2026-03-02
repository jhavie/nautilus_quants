# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Integration tests for BinanceBar extra field support.

Verifies that:
- Factor YAML referencing extra fields (quote_volume, count, etc.) evaluates correctly
- Standard OHLCV-only factors + standard Bar remain unaffected (backward compatibility)
- Breakout strategy (no FactorEngine) + BinanceBar produces consistent close/volume behavior
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nautilus_quants.factors.config import (
    FactorConfig,
    FactorDefinition,
    PerformanceConfig,
    load_factor_config,
)
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.factor_values import FactorValues
from nautilus_quants.strategies.breakout.signal import PriceVolumeBreakoutSignal


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


class MockBar:
    """Standard 5-field bar (open/high/low/close/volume)."""

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


def make_binance_bars(
    instrument_id: str,
    count: int,
    *,
    base_close: float = 100.0,
    close_step: float = 1.0,
    base_volume: float = 1000.0,
    base_qv: float = 5000.0,
    qv_step: float = 100.0,
    trade_count: int = 50,
    buy_ratio: float = 0.6,
) -> list[MockBinanceBar]:
    """Generate a sequence of MockBinanceBars with predictable values."""
    bars: list[MockBinanceBar] = []
    for i in range(count):
        close = base_close + i * close_step
        volume = base_volume + i * 10.0
        qv = base_qv + i * qv_step
        bars.append(
            MockBinanceBar(
                instrument_id=instrument_id,
                open_=close - 1.0,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=volume,
                ts_event=i + 1,
                quote_volume=qv,
                count=trade_count + i,
                taker_buy_base_volume=volume * buy_ratio,
                taker_buy_quote_volume=qv * buy_ratio,
            )
        )
    return bars


def make_standard_bars(
    instrument_id: str,
    count: int,
    *,
    base_close: float = 100.0,
    close_step: float = 1.0,
    base_volume: float = 1000.0,
) -> list[MockBar]:
    """Generate a sequence of standard MockBars with matching OHLCV values."""
    bars: list[MockBar] = []
    for i in range(count):
        close = base_close + i * close_step
        volume = base_volume + i * 10.0
        bars.append(
            MockBar(
                instrument_id=instrument_id,
                open_=close - 1.0,
                high=close + 2.0,
                low=close - 2.0,
                close=close,
                volume=volume,
                ts_event=i + 1,
            )
        )
    return bars


def make_engine_with_extra_fields(
    fields: list[str] | None = None,
    max_history: int = 100,
) -> FactorEngine:
    """Create a FactorEngine with extra fields configured."""
    engine = FactorEngine(max_history=max_history)
    if fields:
        engine.set_extra_fields(fields)
    return engine


# ===========================================================================
# 1. TestBinanceBarFactorEngineIntegration — Engine core pipeline (6 tests)
# ===========================================================================


class TestBinanceBarFactorEngineIntegration:
    """Test extra bar fields flow through the full FactorEngine pipeline."""

    def test_extra_field_ts_mean_computation(self):
        """ts_mean(quote_volume, 5) returns correct mean of last 5 qv values."""
        engine = make_engine_with_extra_fields(["quote_volume", "count"])
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 5)",
            warmup_period=5,
        )

        qv_values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
        result = None
        for i, qv in enumerate(qv_values):
            bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50500, 100, i,
                quote_volume=qv, count=10,
            )
            result = engine.on_bar(bar)

        assert result is not None
        # Last 5 qv values: 300, 400, 500, 600, 700 -> mean = 500
        assert result.get("avg_qv", "BTCUSDT") == pytest.approx(500.0)

    def test_extra_field_ts_std_computation(self):
        """ts_std(quote_volume, 5) returns non-NaN positive value."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="std_qv",
            expression="ts_std(quote_volume, 5)",
            warmup_period=5,
        )

        bars = make_binance_bars("BTCUSDT", 10)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("std_qv", "BTCUSDT")
        assert val is not None
        assert not math.isnan(val)
        assert val > 0

    def test_extra_field_in_variable_then_factor(self):
        """Variable buy_ratio = taker_buy_base_volume / volume -> ts_mean(buy_ratio, 5) ~ 0.6."""
        engine = make_engine_with_extra_fields(
            ["quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume"]
        )
        engine.register_variable("buy_ratio", "taker_buy_base_volume / volume")
        engine.register_expression_factor(
            name="avg_buy_ratio",
            expression="ts_mean(buy_ratio, 5)",
            warmup_period=5,
        )

        bars = make_binance_bars("BTCUSDT", 10, buy_ratio=0.6)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("avg_buy_ratio", "BTCUSDT")
        assert val == pytest.approx(0.6, abs=0.01)

    def test_extra_field_arithmetic_expression(self):
        """quote_volume - taker_buy_quote_volume ~ taker_sell_quote_volume."""
        engine = make_engine_with_extra_fields(
            ["quote_volume", "taker_buy_quote_volume"]
        )
        engine.register_expression_factor(
            name="sell_qv",
            expression="quote_volume - taker_buy_quote_volume",
            warmup_period=1,
        )

        bar = MockBinanceBar(
            "BTCUSDT", 50000, 51000, 49000, 50500, 100, 1,
            quote_volume=10000.0, count=50,
            taker_buy_base_volume=60.0,
            taker_buy_quote_volume=6000.0,
        )
        result = engine.on_bar(bar)

        assert result is not None
        # 10000 - 6000 = 4000
        val = result.get("sell_qv", "BTCUSDT")
        assert val == pytest.approx(4000.0)

    def test_multiple_extra_fields_in_one_expression(self):
        """(taker_buy_base_volume / volume) * (quote_volume / count) is non-NaN."""
        engine = make_engine_with_extra_fields(
            ["quote_volume", "count", "taker_buy_base_volume"]
        )
        engine.register_expression_factor(
            name="composite",
            expression="(taker_buy_base_volume / volume) * (quote_volume / count)",
            warmup_period=1,
        )

        bar = MockBinanceBar(
            "BTCUSDT", 50000, 51000, 49000, 50500, 200, 1,
            quote_volume=10000.0, count=50,
            taker_buy_base_volume=120.0,
            taker_buy_quote_volume=6000.0,
        )
        result = engine.on_bar(bar)

        assert result is not None
        val = result.get("composite", "BTCUSDT")
        assert val is not None
        assert not math.isnan(val)
        # (120/200) * (10000/50) = 0.6 * 200 = 120.0
        assert val == pytest.approx(120.0)

    def test_extra_field_delta_operator(self):
        """delta(quote_volume, 1) = qv[i] - qv[i-1]."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="dqv",
            expression="delta(quote_volume, 1)",
            warmup_period=2,
        )

        qv_values = [1000.0, 1500.0, 1300.0, 1800.0]
        result = None
        for i, qv in enumerate(qv_values):
            bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50500, 100, i,
                quote_volume=qv, count=10,
            )
            result = engine.on_bar(bar)

        assert result is not None
        # delta = 1800 - 1300 = 500
        val = result.get("dqv", "BTCUSDT")
        assert val == pytest.approx(500.0)


# ===========================================================================
# 2. TestStandardBarBackwardCompatibility — Backward compatibility (3 tests)
# ===========================================================================


class TestStandardBarBackwardCompatibility:
    """Standard OHLCV-only factors + standard Bar remain unaffected."""

    def test_ohlcv_only_factor_with_standard_bar(self):
        """ts_mean(close, 5) + MockBar -> correct SMA."""
        engine = FactorEngine(max_history=100)
        engine.register_expression_factor(
            name="sma5",
            expression="ts_mean(close, 5)",
            warmup_period=5,
        )

        close_values = [100.0, 102.0, 104.0, 103.0, 105.0, 107.0]
        result = None
        for i, c in enumerate(close_values):
            bar = MockBar("BTCUSDT", c - 1, c + 1, c - 2, c, 1000.0, i)
            result = engine.on_bar(bar)

        assert result is not None
        # Last 5: 102, 104, 103, 105, 107 -> mean = 104.2
        assert result.get("sma5", "BTCUSDT") == pytest.approx(104.2)

    def test_ohlcv_factor_still_works_when_extra_fields_configured(self):
        """Engine has extra_fields set but factor only uses close -> result unaffected."""
        engine = make_engine_with_extra_fields(["quote_volume", "count"])
        engine.register_expression_factor(
            name="sma3",
            expression="ts_mean(close, 3)",
            warmup_period=3,
        )

        close_values = [10.0, 20.0, 30.0, 40.0]
        result = None
        for i, c in enumerate(close_values):
            bar = MockBar("BTCUSDT", c, c + 1, c - 1, c, 100.0, i)
            result = engine.on_bar(bar)

        assert result is not None
        # Last 3: 20, 30, 40 -> mean = 30.0
        assert result.get("sma3", "BTCUSDT") == pytest.approx(30.0)

    def test_standard_bar_produces_same_ohlcv_result_as_binance_bar(self):
        """Same OHLCV factor, MockBar vs MockBinanceBar -> identical result."""
        close_vals = [100.0, 101.0, 102.0, 103.0, 104.0]

        # Engine A: standard bars, no extra fields
        engine_a = FactorEngine(max_history=100)
        engine_a.register_expression_factor(
            name="sma5", expression="ts_mean(close, 5)", warmup_period=5,
        )

        # Engine B: binance bars, extra fields configured
        engine_b = make_engine_with_extra_fields(["quote_volume", "count"])
        engine_b.register_expression_factor(
            name="sma5", expression="ts_mean(close, 5)", warmup_period=5,
        )

        result_a = result_b = None
        for i, c in enumerate(close_vals):
            bar_a = MockBar("BTCUSDT", c - 1, c + 1, c - 2, c, 500.0, i)
            bar_b = MockBinanceBar(
                "BTCUSDT", c - 1, c + 1, c - 2, c, 500.0, i,
                quote_volume=9999.0, count=42,
            )
            result_a = engine_a.on_bar(bar_a)
            result_b = engine_b.on_bar(bar_b)

        assert result_a is not None and result_b is not None
        assert result_a.get("sma5", "BTCUSDT") == pytest.approx(
            result_b.get("sma5", "BTCUSDT")
        )


# ===========================================================================
# 3. TestMixedFactors — Mixed factors (3 tests)
# ===========================================================================


class TestMixedFactors:
    """OHLCV and extra-field factors coexist correctly."""

    def test_ohlcv_and_extra_field_factors_coexist(self):
        """ts_mean(close, 5) + ts_mean(quote_volume, 5) in same engine -> each correct."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="sma_close", expression="ts_mean(close, 5)", warmup_period=5,
        )
        engine.register_expression_factor(
            name="sma_qv", expression="ts_mean(quote_volume, 5)", warmup_period=5,
        )

        bars = make_binance_bars("BTCUSDT", 7, base_close=100.0, close_step=2.0, base_qv=1000.0, qv_step=200.0)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        # Last 5 close: 104, 106, 108, 110, 112 -> mean = 108.0
        assert result.get("sma_close", "BTCUSDT") == pytest.approx(108.0)
        # Last 5 qv: 1400, 1600, 1800, 2000, 2200 -> mean = 1800.0 (wait, let me recalculate)
        # qv values: 1000, 1200, 1400, 1600, 1800, 2000, 2200
        # Last 5: 1400, 1600, 1800, 2000, 2200 -> mean = 1800.0
        assert result.get("sma_qv", "BTCUSDT") == pytest.approx(1800.0)

    def test_mixed_variables_and_factors(self):
        """Variable uses OHLCV + extra field -> factor composes them correctly."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_variable("price_per_qv", "close / quote_volume")
        engine.register_expression_factor(
            name="avg_price_per_qv",
            expression="ts_mean(price_per_qv, 3)",
            warmup_period=3,
        )

        # close=100, qv=1000 -> ratio = 0.1
        # close=200, qv=2000 -> ratio = 0.1
        # close=300, qv=3000 -> ratio = 0.1
        result = None
        for i in range(3):
            bar = MockBinanceBar(
                "BTCUSDT", 99, 301, 99, (i + 1) * 100.0, 500, i,
                quote_volume=(i + 1) * 1000.0, count=10,
            )
            result = engine.on_bar(bar)

        assert result is not None
        assert result.get("avg_price_per_qv", "BTCUSDT") == pytest.approx(0.1, abs=0.001)

    def test_three_factors_different_field_types(self):
        """close SMA + qv SMA + raw count -> all three in result.factors."""
        engine = make_engine_with_extra_fields(["quote_volume", "count"])
        engine.register_expression_factor(
            name="sma_close", expression="ts_mean(close, 3)", warmup_period=3,
        )
        engine.register_expression_factor(
            name="sma_qv", expression="ts_mean(quote_volume, 3)", warmup_period=3,
        )
        engine.register_expression_factor(
            name="raw_count", expression="count", warmup_period=1,
        )

        bars = make_binance_bars("BTCUSDT", 5, base_close=100.0, base_qv=500.0, trade_count=100)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        factors = result.factors
        assert "sma_close" in factors
        assert "sma_qv" in factors
        assert "raw_count" in factors

        # raw_count should be the last bar's count value
        assert result.get("raw_count", "BTCUSDT") == pytest.approx(104.0)


# ===========================================================================
# 4. TestGracefulDegradation — Degradation behavior (3 tests)
# ===========================================================================


class TestGracefulDegradation:
    """Engine degrades gracefully when extra fields are missing from bar."""

    def test_extra_field_factor_with_standard_bar_uses_zero(self):
        """Extra fields configured but fed MockBar -> getattr returns 0 -> ts_mean = 0.0."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 3)",
            warmup_period=3,
        )

        bars = make_standard_bars("BTCUSDT", 5)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("avg_qv", "BTCUSDT")
        # Standard bars have no quote_volume -> getattr(bar, "quote_volume", 0) = 0
        assert val == pytest.approx(0.0)

    def test_division_by_extra_field_with_standard_bar(self):
        """close / quote_volume with standard bar -> inf or NaN, no crash."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="close_over_qv",
            expression="close / quote_volume",
            warmup_period=1,
        )

        bar = MockBar("BTCUSDT", 99, 101, 98, 100.0, 500.0, 1)
        result = engine.on_bar(bar)

        assert result is not None
        val = result.get("close_over_qv", "BTCUSDT")
        assert val is not None
        # Division by zero -> inf or NaN; either is acceptable, no crash
        assert math.isinf(val) or math.isnan(val)

    def test_no_crash_on_missing_extra_field(self):
        """ts_mean(nonexistent_field, 5) with MockBar -> output 0.0, no exception."""
        engine = make_engine_with_extra_fields(["nonexistent_field"])
        engine.register_expression_factor(
            name="avg_missing",
            expression="ts_mean(nonexistent_field, 5)",
            warmup_period=5,
        )

        bars = make_standard_bars("BTCUSDT", 7)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("avg_missing", "BTCUSDT")
        assert val == pytest.approx(0.0)


# ===========================================================================
# 5. TestBreakoutStrategyBinanceBarCompatibility — Breakout signal (4 tests)
# ===========================================================================


class TestBreakoutStrategyBinanceBarCompatibility:
    """Breakout signal produces identical results regardless of bar type."""

    def setup_method(self):
        self.signal = PriceVolumeBreakoutSignal(breakout_period=5, sma_period=5)

    def test_binance_bar_close_volume_extraction(self):
        """float(MockBinanceBar.close) and float(MockBinanceBar.volume) match MockBar."""
        bar_std = MockBar("BTCUSDT", 99, 101, 98, 100.0, 500.0, 1)
        bar_bin = MockBinanceBar(
            "BTCUSDT", 99, 101, 98, 100.0, 500.0, 1,
            quote_volume=9999.0, count=42,
        )

        assert float(bar_std.close) == float(bar_bin.close)
        assert float(bar_std.volume) == float(bar_bin.volume)

    def test_check_long_identical_both_bar_types(self):
        """Same close/volume data -> check_long result identical for both bar types."""
        recent_closes = [95.0, 96.0, 97.0, 98.0, 99.0]
        recent_volumes = [400.0, 410.0, 420.0, 430.0, 440.0]
        current_close = 100.0
        current_volume = 500.0
        sma_value = 95.0

        result = self.signal.check_long(
            current_close, current_volume,
            recent_closes, recent_volumes, sma_value,
        )

        # Should be True: close > max(recent), volume > max(recent), close > sma
        assert result is True

        # Same call with values extracted from BinanceBar
        bar = MockBinanceBar(
            "BTCUSDT", 99, 101, 98, current_close, current_volume, 1,
            quote_volume=50000.0, count=999,
        )
        result_bin = self.signal.check_long(
            float(bar.close), float(bar.volume),
            recent_closes, recent_volumes, sma_value,
        )
        assert result == result_bin

    def test_check_short_identical_both_bar_types(self):
        """Same close/volume data -> check_short result identical for both bar types."""
        recent_closes = [105.0, 106.0, 107.0, 108.0, 109.0]
        recent_volumes = [400.0, 410.0, 420.0, 430.0, 440.0]
        current_close = 100.0
        current_volume = 500.0
        sma_value = 110.0

        result = self.signal.check_short(
            current_close, current_volume,
            recent_closes, recent_volumes, sma_value,
        )

        # close < min(recent), volume > max(recent), close < sma
        assert result is True

        bar = MockBinanceBar(
            "BTCUSDT", 99, 101, 98, current_close, current_volume, 1,
            quote_volume=50000.0, count=999,
        )
        result_bin = self.signal.check_short(
            float(bar.close), float(bar.volume),
            recent_closes, recent_volumes, sma_value,
        )
        assert result == result_bin

    def test_signal_ignores_extra_fields(self):
        """Two BinanceBars with same close/volume but wildly different extras -> same signal."""
        recent_closes = [95.0, 96.0, 97.0, 98.0, 99.0]
        recent_volumes = [400.0, 410.0, 420.0, 430.0, 440.0]
        sma_value = 95.0

        bar_a = MockBinanceBar(
            "BTCUSDT", 99, 101, 98, 100.0, 500.0, 1,
            quote_volume=1.0, count=1,
            taker_buy_base_volume=0.1, taker_buy_quote_volume=0.1,
        )
        bar_b = MockBinanceBar(
            "BTCUSDT", 99, 101, 98, 100.0, 500.0, 1,
            quote_volume=999999.0, count=999999,
            taker_buy_base_volume=499999.0, taker_buy_quote_volume=499999.0,
        )

        result_a = self.signal.check_long(
            float(bar_a.close), float(bar_a.volume),
            recent_closes, recent_volumes, sma_value,
        )
        result_b = self.signal.check_long(
            float(bar_b.close), float(bar_b.volume),
            recent_closes, recent_volumes, sma_value,
        )
        assert result_a == result_b


# ===========================================================================
# 6. TestFullPipelineExtraFields — End-to-end pipeline (6 tests)
# ===========================================================================


class TestFullPipelineExtraFields:
    """End-to-end: bars -> engine -> FactorValues -> serialization roundtrips."""

    def _build_result(self) -> FactorValues:
        """Feed 20 binance bars through engine and return last FactorValues."""
        engine = make_engine_with_extra_fields(["quote_volume", "count"])
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 5)",
            warmup_period=5,
        )

        bars = make_binance_bars("BTCUSDT", 20, base_qv=1000.0, qv_step=100.0)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)
        assert result is not None
        return result

    def test_pipeline_factor_values_get(self):
        """20 bars -> result.get('avg_qv', 'BTCUSDT') returns correct float."""
        result = self._build_result()
        val = result.get("avg_qv", "BTCUSDT")
        assert val is not None
        assert isinstance(val, float)
        # qv values: 1000, 1100, ..., 2900
        # Last 5: 2500, 2600, 2700, 2800, 2900 -> mean = 2700.0
        assert val == pytest.approx(2700.0)

    def test_pipeline_json_roundtrip(self):
        """to_json() -> from_json() preserves extra-field derived values."""
        original = self._build_result()
        json_str = original.to_json()
        restored = FactorValues.from_json(json_str)

        assert restored.get("avg_qv", "BTCUSDT") == pytest.approx(
            original.get("avg_qv", "BTCUSDT"),
        )

    def test_pipeline_bytes_roundtrip(self):
        """to_bytes() -> from_bytes() preserves extra-field derived values."""
        original = self._build_result()
        raw = original.to_bytes()
        restored = FactorValues.from_bytes(raw)

        assert restored.get("avg_qv", "BTCUSDT") == pytest.approx(
            original.get("avg_qv", "BTCUSDT"),
        )

    def test_pipeline_custom_data_roundtrip(self):
        """to_custom_data() -> from_custom_data() preserves values."""
        original = self._build_result()
        try:
            custom = original.to_custom_data()
            restored = FactorValues.from_custom_data(custom)
            assert restored.get("avg_qv", "BTCUSDT") == pytest.approx(
                original.get("avg_qv", "BTCUSDT"),
            )
        except ImportError:
            pytest.skip("nautilus_trader pyo3 CustomData not available")

    def test_pipeline_multi_instrument(self):
        """BTCUSDT + ETHUSDT with different quote_volume -> each correct via FactorValues."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 3)",
            warmup_period=3,
        )

        # Feed interleaved bars for two instruments
        btc_result = None
        eth_result = None
        for i in range(5):
            btc_bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50000, 100, i * 2,
                quote_volume=1000.0 * (i + 1), count=10,
            )
            eth_bar = MockBinanceBar(
                "ETHUSDT", 3000, 3100, 2900, 3000, 50, i * 2 + 1,
                quote_volume=100.0 * (i + 1), count=5,
            )
            btc_result = engine.on_bar(btc_bar)
            eth_result = engine.on_bar(eth_bar)

        # BTC last 3 qv: 3000, 4000, 5000 -> mean = 4000
        assert btc_result is not None
        assert btc_result.get("avg_qv", "BTCUSDT") == pytest.approx(4000.0)

        # ETH last 3 qv: 300, 400, 500 -> mean = 400
        assert eth_result is not None
        assert eth_result.get("avg_qv", "ETHUSDT") == pytest.approx(400.0)

    def test_pipeline_reset_preserves_config(self):
        """engine.reset() clears data but extra_fields still effective."""
        engine = make_engine_with_extra_fields(["quote_volume"])
        engine.register_expression_factor(
            name="avg_qv",
            expression="ts_mean(quote_volume, 3)",
            warmup_period=3,
        )

        # Feed initial bars
        bars = make_binance_bars("BTCUSDT", 5, base_qv=1000.0, qv_step=100.0)
        for bar in bars:
            engine.on_bar(bar)

        # Reset
        engine.reset()

        # Feed new bars after reset
        new_bars = make_binance_bars("BTCUSDT", 5, base_qv=2000.0, qv_step=200.0)
        result = None
        for bar in new_bars:
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("avg_qv", "BTCUSDT")
        # qv: 2000, 2200, 2400, 2600, 2800 -> last 3: 2400, 2600, 2800 -> mean = 2600
        assert val == pytest.approx(2600.0)


# ===========================================================================
# 7. TestYamlConfigExtraFieldIntegration — YAML config loading (4 tests)
# ===========================================================================


class TestYamlConfigExtraFieldIntegration:
    """YAML config loads correctly and works with extra fields."""

    def test_inline_config_with_extra_field_variable(self):
        """FactorConfig with variables referencing extra field -> engine computes correctly."""
        config = FactorConfig(
            name="test_extra",
            variables={"avg_qv": "ts_mean(quote_volume, 3)"},
            factors=[
                FactorDefinition(
                    name="qv_factor",
                    expression="avg_qv",
                    description="Average quote volume via variable",
                ),
            ],
        )
        engine = FactorEngine(config=config, max_history=100)
        engine.set_extra_fields(["quote_volume"])

        qv_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = None
        for i, qv in enumerate(qv_values):
            bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50500, 100, i,
                quote_volume=qv, count=10,
            )
            result = engine.on_bar(bar)

        assert result is not None
        # warmup auto-estimated from ts_mean(..., 3) -> 4
        # After 5 bars: last 3 qv = 300, 400, 500 -> mean = 400
        val = result.get("qv_factor", "BTCUSDT")
        assert val == pytest.approx(400.0)

    def test_yaml_file_config_with_extra_fields(self, tmp_path: Path):
        """Write YAML to tmp_path -> load_factor_config() -> engine computes correctly."""
        yaml_content = """\
metadata:
  name: test_extra_yaml
  version: "1.0"

parameters: {}

variables:
  avg_qv: "ts_mean(quote_volume, 3)"

factors:
  qv_factor:
    expression: "avg_qv"
    description: "Average quote volume"
    category: "extra"

performance:
  max_compute_time_ms: 1.0
  enable_timing: false
  warning_threshold_ms: 0.5
"""
        yaml_path = tmp_path / "extra_factors.yaml"
        yaml_path.write_text(yaml_content)

        config = load_factor_config(str(yaml_path))
        engine = FactorEngine(config=config, max_history=100)
        engine.set_extra_fields(["quote_volume"])

        qv_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = None
        for i, qv in enumerate(qv_values):
            bar = MockBinanceBar(
                "BTCUSDT", 50000, 51000, 49000, 50500, 100, i,
                quote_volume=qv, count=10,
            )
            result = engine.on_bar(bar)

        assert result is not None
        val = result.get("qv_factor", "BTCUSDT")
        assert val == pytest.approx(400.0)

    def test_existing_factors_yaml_with_binance_bars(self):
        """Load config/factors.yaml (OHLCV only) -> feed BinanceBar -> alpha_breakout unaffected."""
        config_path = Path(__file__).parents[2] / "config" / "factors.yaml"
        if not config_path.exists():
            pytest.skip("config/factors.yaml not found")

        config = load_factor_config(str(config_path))
        engine = FactorEngine(config=config, max_history=500)
        engine.set_extra_fields(["quote_volume", "count"])

        # Feed enough bars for the breakout factor (needs sma_period=200 warmup)
        # Use 250 bars to be safe
        bars = make_binance_bars(
            "BTCUSDT", 250,
            base_close=100.0, close_step=0.1,
            base_volume=1000.0, base_qv=5000.0,
        )
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        # alpha_breakout should exist and be a valid number (not crash)
        val = result.get("alpha_breakout", "BTCUSDT")
        assert val is not None
        assert not math.isnan(val)

    def test_config_mixed_standard_and_extra_fields(self):
        """Config has both close factor and quote_volume factor -> each correct."""
        config = FactorConfig(
            name="mixed",
            factors=[
                FactorDefinition(name="sma_close", expression="ts_mean(close, 3)"),
                FactorDefinition(name="sma_qv", expression="ts_mean(quote_volume, 3)"),
            ],
        )
        engine = FactorEngine(config=config, max_history=100)
        engine.set_extra_fields(["quote_volume"])

        bars = make_binance_bars("BTCUSDT", 5, base_close=100.0, close_step=10.0, base_qv=500.0, qv_step=100.0)
        result = None
        for bar in bars:
            result = engine.on_bar(bar)

        assert result is not None
        # close values: 100, 110, 120, 130, 140 -> last 3: 120, 130, 140 -> mean = 130.0
        assert result.get("sma_close", "BTCUSDT") == pytest.approx(130.0)
        # qv values: 500, 600, 700, 800, 900 -> last 3: 700, 800, 900 -> mean = 800.0
        assert result.get("sma_qv", "BTCUSDT") == pytest.approx(800.0)
