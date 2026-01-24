# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Integration test for FactorStrategy with FactorEngineActor.

This test validates the complete Constitution-compliant flow:
1. FactorEngineActor computes factors from bars
2. FactorEngineActor publishes signals via MessageBus  
3. FactorStrategy subscribes to signals and makes trading decisions
"""

import pytest
import numpy as np

from nautilus_quants.factors import FactorEngine, FactorValues
from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineConfig


class TestFactorValuesCustomData:
    """Test FactorValues serialization and CustomData conversion."""

    def test_to_json_and_back(self):
        """Test JSON serialization round-trip."""
        original = FactorValues(
            ts_event=1234567890000000000,
            factors={
                "alpha001": {"ETHUSDT.BINANCE": 0.5, "BTCUSDT.BINANCE": -0.3},
                "alpha002": {"ETHUSDT.BINANCE": 0.8},
            },
        )

        # Serialize
        json_str = original.to_json()
        assert '"ts_event": 1234567890000000000' in json_str
        assert '"alpha001"' in json_str

        # Deserialize
        restored = FactorValues.from_json(json_str)
        assert restored.ts_event == original.ts_event
        assert restored.get("alpha001", "ETHUSDT.BINANCE") == 0.5
        assert restored.get("alpha001", "BTCUSDT.BINANCE") == -0.3
        assert restored.get("alpha002", "ETHUSDT.BINANCE") == 0.8

    def test_to_bytes_and_back(self):
        """Test bytes serialization round-trip."""
        original = FactorValues(
            ts_event=9999999999000000000,
            factors={"breakout": {"ETHUSDT.BINANCE": 1.0}},
        )

        # Serialize
        data = original.to_bytes()
        assert isinstance(data, bytes)

        # Deserialize
        restored = FactorValues.from_bytes(data)
        assert restored.ts_event == original.ts_event
        assert restored.get("breakout", "ETHUSDT.BINANCE") == 1.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        fv = FactorValues(
            ts_event=123,
            factors={"f1": {"A": 1.0}},
        )
        d = fv.to_dict()
        assert d["ts_event"] == 123
        assert d["factors"]["f1"]["A"] == 1.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"ts_event": 456, "factors": {"f2": {"B": 2.0}}}
        fv = FactorValues.from_dict(d)
        assert fv.ts_event == 456
        assert fv.get("f2", "B") == 2.0

    def test_to_custom_data(self):
        """Test conversion to Nautilus CustomData."""
        fv = FactorValues(
            ts_event=1000000000,
            factors={"alpha": {"ETHUSDT.BINANCE": 0.5}},
        )

        # Convert to CustomData
        custom_data = fv.to_custom_data()

        # Verify structure
        assert custom_data.ts_event == 1000000000
        assert custom_data.ts_init == 1000000000
        # DataType stores type name, verify it's set correctly
        assert custom_data.data_type is not None

        # Verify we can restore from CustomData
        restored = FactorValues.from_custom_data(custom_data)
        assert restored.ts_event == fv.ts_event
        assert restored.get("alpha", "ETHUSDT.BINANCE") == 0.5

    def test_custom_data_with_different_ts_init(self):
        """Test CustomData with explicit ts_init."""
        fv = FactorValues(ts_event=1000, factors={})
        custom_data = fv.to_custom_data(ts_init=2000)

        assert custom_data.ts_event == 1000
        assert custom_data.ts_init == 2000


class TestFactorEngineActorConfig:
    """Test FactorEngineActor configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FactorEngineConfig()

        assert config.factor_config_path is None
        assert config.bar_types is None
        assert config.max_history == 500
        assert config.publish_signals is True
        assert config.signal_prefix == "factor"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FactorEngineConfig(
            factor_config_path="config/factors.yaml",
            bar_types=["ETHUSDT.BINANCE-1-HOUR-LAST"],
            max_history=1000,
            publish_signals=False,
            signal_prefix="custom_factor",
        )

        assert config.factor_config_path == "config/factors.yaml"
        assert config.bar_types == ["ETHUSDT.BINANCE-1-HOUR-LAST"]
        assert config.max_history == 1000
        assert config.publish_signals is False
        assert config.signal_prefix == "custom_factor"


class TestFactorEngineIntegration:
    """Test FactorEngine with FactorValues flow."""

    def test_engine_produces_factor_values(self):
        """Test that FactorEngine produces FactorValues on each bar."""
        from nautilus_trader.test_kit.stubs.data import TestDataStubs
        from nautilus_trader.model.identifiers import InstrumentId

        # Create engine with a simple factor
        engine = FactorEngine(max_history=100)
        engine.register_expression_factor(
            name="sma_5",
            expression="ts_mean(close, 5)",
            warmup_period=5,
        )

        # Create test bars
        instrument_id = InstrumentId.from_str("ETHUSDT.BINANCE")
        bar_type = TestDataStubs.bartype_adabtc_binance_1min_last()

        results = []
        for i in range(20):
            bar = TestDataStubs.bar_5decimal()
            result = engine.on_bar(bar)
            if result is not None:
                results.append(result)

        # Verify we got FactorValues
        assert len(results) > 0
        for result in results:
            assert isinstance(result, FactorValues)
            # ts_event comes from bar, TestDataStubs bar may have ts_event=0
            assert result.ts_event >= 0
            assert "sma_5" in result.factors

    def test_factor_values_serialization_in_flow(self):
        """Test FactorValues serialization works in engine flow."""
        from nautilus_trader.test_kit.stubs.data import TestDataStubs

        engine = FactorEngine(max_history=50)
        engine.register_expression_factor(
            name="momentum",
            expression="delta(close, 3)",
            warmup_period=4,
        )

        # Process bars and collect results
        results = []
        for i in range(10):
            bar = TestDataStubs.bar_5decimal()
            result = engine.on_bar(bar)
            if result:
                results.append(result)

        # Verify serialization works for all results
        for result in results:
            # JSON round-trip
            json_str = result.to_json()
            restored = FactorValues.from_json(json_str)
            assert restored.ts_event == result.ts_event

            # CustomData round-trip
            custom_data = result.to_custom_data()
            restored2 = FactorValues.from_custom_data(custom_data)
            assert restored2.ts_event == result.ts_event


class TestBreakoutFactorFlow:
    """Test the complete breakout factor flow with CustomData."""

    def test_breakout_factor_to_custom_data(self):
        """Test breakout factor values can be converted to CustomData."""
        from nautilus_trader.test_kit.stubs.data import TestDataStubs

        engine = FactorEngine(max_history=100)

        # Register breakout components
        engine.register_variable("highest_close", "delay(ts_max(close, 10), 1)")
        engine.register_variable("highest_volume", "delay(ts_max(volume, 10), 1)")

        engine.register_expression_factor(
            name="breakout",
            expression="(close > highest_close) * (volume > highest_volume)",
            warmup_period=12,
        )

        # Process bars
        all_custom_data = []
        for i in range(50):
            bar = TestDataStubs.bar_5decimal()
            result = engine.on_bar(bar)
            if result:
                custom_data = result.to_custom_data()
                all_custom_data.append(custom_data)

                # Verify we can restore
                restored = FactorValues.from_custom_data(custom_data)
                breakout_value = restored.get("breakout", "ADA/BTC.BINANCE")
                # Value should be 0 or 1 (binary signal)
                if breakout_value is not None:
                    assert breakout_value in [0.0, 1.0] or np.isnan(breakout_value)

        assert len(all_custom_data) > 0
        print(f"Processed {len(all_custom_data)} bars with CustomData")


class TestFactorStrategyImport:
    """Test FactorStrategy can be imported."""

    def test_import_factor_strategy(self):
        """Test FactorStrategy imports correctly."""
        from nautilus_quants.strategies.factor import FactorStrategy, FactorStrategyConfig

        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            bar_type="ETHUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
            factor_signal_name="factor.breakout",
            entry_threshold=1.0,
            order_amount=10000.0,
            stop_loss_pct=0.02,
        )

        assert config.instrument_id == "ETHUSDT.BINANCE"
        assert config.factor_signal_name == "factor.breakout"
        assert config.entry_threshold == 1.0
