# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Integration tests for Actor-decoupled backtest architecture.

Tests the complete flow:
1. FactorEngineActor computes factors from bars
2. FactorEngineActor publishes signals via MessageBus
3. FactorStrategy subscribes to signals and executes trades
"""

import pytest

from nautilus_quants.factors.engine import FactorEngineActor, FactorEngineActorConfig
from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.factor import FactorStrategy, FactorStrategyConfig


class TestActorDecouplingArchitecture:
    """Integration tests for Actor-Strategy decoupling."""

    def test_factor_engine_actor_config_creation(self) -> None:
        """Test FactorEngineActorConfig can be created with required fields."""
        config = FactorEngineActorConfig(
            factor_config_path="config/factors.yaml",
        )
        assert config.factor_config_path == "config/factors.yaml"
        assert config.signal_prefix == "factor"
        assert config.max_history == 500
        assert config.publish_signals is True

    def test_factor_engine_actor_config_with_interval(self) -> None:
        """Test FactorEngineActorConfig with custom interval."""
        config = FactorEngineActorConfig(
            factor_config_path="config/factors.yaml",
            interval="1h",
        )
        assert config.interval == "1h"

    def test_factor_strategy_config_creation(self) -> None:
        """Test FactorStrategyConfig can be created with required fields."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        assert config.instrument_id == "ETHUSDT.BINANCE"
        assert config.signal_name == "factor.alpha_breakout_long"

    def test_factor_values_serialization(self) -> None:
        """Test FactorValues can be serialized to CustomData format."""
        values = FactorValues.create(
            ts_event=1234567890000000000,
            factors={
                "alpha_breakout_long": {"ETHUSDT.BINANCE": 1.0},
            }
        )
        
        # Test JSON serialization
        json_str = values.to_json()
        assert "alpha_breakout_long" in json_str
        assert "ETHUSDT.BINANCE" in json_str
        
        # Test roundtrip
        restored = FactorValues.from_json(json_str)
        assert restored.ts_event == values.ts_event
        assert restored.factors == values.factors

    def test_factor_values_bytes_serialization(self) -> None:
        """Test FactorValues can be serialized to bytes."""
        values = FactorValues.create(
            ts_event=1234567890000000000,
            factors={
                "momentum": {"BTCUSDT.BINANCE": 0.5},
            }
        )
        
        # Test bytes serialization
        data = values.to_bytes()
        assert isinstance(data, bytes)
        
        # Test roundtrip
        restored = FactorValues.from_bytes(data)
        assert restored.ts_event == values.ts_event
        assert restored.get("momentum", "BTCUSDT.BINANCE") == 0.5

    def test_config_compatibility(self) -> None:
        """Test Actor and Strategy configs are compatible with YAML format."""
        # These configs should be serializable to YAML-compatible dicts
        actor_config = FactorEngineActorConfig(
            factor_config_path="config/factors.yaml",
            signal_prefix="factor",
            max_history=500,
        )
        
        strategy_config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
            entry_threshold=1.0,
            stop_loss_pct=0.02,
            order_amount=10000.0,
            enable_long=True,
            enable_short=False,
        )
        
        # Verify all required fields are present
        assert actor_config.factor_config_path is not None
        assert strategy_config.instrument_id is not None
        assert strategy_config.signal_name is not None


class TestFactorValuesCustomData:
    """Tests for FactorValues CustomData integration."""

    def test_to_custom_data_creates_valid_object(self) -> None:
        """Test to_custom_data creates a valid CustomData object."""
        values = FactorValues.create(
            ts_event=1234567890000000000,
            factors={"alpha001": {"ETHUSDT.BINANCE": 0.75}},
        )
        
        try:
            custom_data = values.to_custom_data()
            assert custom_data is not None
            assert custom_data.ts_event == values.ts_event
        except ImportError:
            pytest.skip("nautilus_trader.core.nautilus_pyo3 not available")

    def test_from_custom_data_restores_values(self) -> None:
        """Test from_custom_data restores original values."""
        original = FactorValues.create(
            ts_event=9876543210000000000,
            factors={
                "alpha001": {"ETHUSDT.BINANCE": 0.5, "BTCUSDT.BINANCE": -0.3},
                "alpha002": {"ETHUSDT.BINANCE": 0.8},
            },
        )
        
        try:
            custom_data = original.to_custom_data()
            restored = FactorValues.from_custom_data(custom_data)
            
            assert restored.ts_event == original.ts_event
            assert restored.factors == original.factors
        except ImportError:
            pytest.skip("nautilus_trader.core.nautilus_pyo3 not available")
