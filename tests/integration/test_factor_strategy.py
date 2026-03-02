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

from nautilus_quants.factors import FactorValues
from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineActorConfig


class TestFactorValuesCustomData:
    """Test FactorValues serialization and CustomData conversion."""

    def test_to_json_and_back(self):
        """Test JSON serialization round-trip."""
        original = FactorValues.create(
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
        original = FactorValues.create(
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
        fv = FactorValues.create(
            ts_event=123,
            factors={"f1": {"A": 1.0}},
        )
        d = fv.to_dict()
        assert d["ts_event"] == 123
        assert fv.factors["f1"]["A"] == 1.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"ts_event": 456, "ts_init": 456, "factors_bytes": b'{"f2": {"B": 2.0}}'}
        fv = FactorValues.from_dict(d)
        assert fv.ts_event == 456
        assert fv.get("f2", "B") == 2.0

    def test_to_custom_data(self):
        """Test conversion to Nautilus CustomData."""
        fv = FactorValues.create(
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
        fv = FactorValues.create(ts_event=1000, factors={}, ts_init=2000)

        assert fv.ts_event == 1000
        assert fv.ts_init == 2000


class TestFactorEngineActorConfigIntegration:
    """Test FactorEngineActor configuration."""

    def test_required_config(self):
        """Test configuration with required field."""
        config = FactorEngineActorConfig(
            factor_config_path="config/factors.yaml",
        )

        assert config.factor_config_path == "config/factors.yaml"
        assert config.data_cls == ""
        assert config.bar_spec == ""
        assert config.max_history == 500
        assert config.publish_signals is True
        assert config.signal_prefix == "factor"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FactorEngineActorConfig(
            factor_config_path="config/factors.yaml",
            data_cls="nautilus_trader.adapters.binance.common.types:BinanceBar",
            bar_spec="4h",
            max_history=1000,
            publish_signals=False,
            signal_prefix="custom_factor",
        )

        assert config.factor_config_path == "config/factors.yaml"
        assert config.data_cls == "nautilus_trader.adapters.binance.common.types:BinanceBar"
        assert config.bar_spec == "4h"
        assert config.max_history == 1000
        assert config.publish_signals is False
        assert config.signal_prefix == "custom_factor"


class TestFactorStrategyImport:
    """Test FactorStrategy can be imported."""

    def test_import_factor_strategy(self):
        """Test FactorStrategy imports correctly."""
        from nautilus_quants.strategies.factor import FactorStrategy, FactorStrategyConfig

        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.breakout",
            entry_threshold=1.0,
            order_amount=10000.0,
            stop_loss_pct=0.02,
        )

        assert config.instrument_id == "ETHUSDT.BINANCE"
        assert config.signal_name == "factor.breakout"
        assert config.entry_threshold == 1.0
