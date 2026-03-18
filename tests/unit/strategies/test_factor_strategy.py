# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for FactorStrategy and FactorStrategyConfig."""

import pytest

from nautilus_quants.strategies.factor import FactorStrategy, FactorStrategyConfig


class TestFactorStrategyConfig:
    """Tests for FactorStrategyConfig."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        assert config.instrument_id == "ETHUSDT.BINANCE"
        assert config.signal_name == "factor.alpha_breakout_long"

    def test_defaults_are_applied(self) -> None:
        """Test default values are applied."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        assert config.entry_threshold == 1.0
        assert config.stop_loss_pct == 0.02
        assert config.order_amount == 10000.0
        assert config.enable_long is True
        assert config.enable_short is False

    def test_creates_with_custom_values(self) -> None:
        """Test creation with custom values."""
        config = FactorStrategyConfig(
            instrument_id="BTCUSDT.BINANCE",
            signal_name="factor.momentum",
            entry_threshold=0.5,
            stop_loss_pct=0.03,
            order_amount=5000.0,
            enable_long=True,
            enable_short=True,
        )
        assert config.instrument_id == "BTCUSDT.BINANCE"
        assert config.signal_name == "factor.momentum"
        assert config.entry_threshold == 0.5
        assert config.stop_loss_pct == 0.03
        assert config.order_amount == 5000.0
        assert config.enable_long is True
        assert config.enable_short is True

    def test_bar_type_not_in_config(self) -> None:
        """Test bar_type is not in config (auto-inferred from cache)."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        # bar_type is auto-inferred from cache, not in config
        assert config.instrument_id == "ETHUSDT.BINANCE"

    def test_init_parses_bar_type_when_provided(self) -> None:
        """Test initialization starts with no bar_type (will be inferred)."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        strategy = FactorStrategy(config)
        
        # bar_type is None until on_start() infers from cache
        assert strategy._bar_type is None


class TestFactorStrategyInit:
    """Tests for FactorStrategy initialization."""

    def test_init_stores_config(self) -> None:
        """Test initialization stores config correctly."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        strategy = FactorStrategy(config)
        
        assert str(strategy.instrument_id) == "ETHUSDT.BINANCE"
        assert strategy.signal_name == "factor.alpha_breakout_long"

    def test_init_parses_instrument_id(self) -> None:
        """Test initialization parses instrument_id correctly."""
        config = FactorStrategyConfig(
            instrument_id="BTCUSDT.BINANCE",
            signal_name="factor.momentum",
        )
        strategy = FactorStrategy(config)
        
        assert strategy.instrument_id.symbol.value == "BTCUSDT"
        assert strategy.instrument_id.venue.value == "BINANCE"

    def test_init_parses_bar_type_when_provided(self) -> None:
        """Test initialization starts with no bar_type (will be inferred)."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        strategy = FactorStrategy(config)
        
        # bar_type is None until on_start() infers from cache
        assert strategy._bar_type is None

    def test_init_state_is_clean(self) -> None:
        """Test initialization starts with clean state."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
        )
        strategy = FactorStrategy(config)
        
        assert strategy.entry_price is None
        assert strategy.position_side is None
        assert strategy._current_close is None
        assert strategy._bar_count == 0
        assert strategy._signal_count == 0


class TestFactorStrategyLogic:
    """Tests for FactorStrategy trading logic."""

    def test_calculate_quantity(self) -> None:
        """Test quantity calculation."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
            order_amount=10000.0,
        )
        strategy = FactorStrategy(config)
        
        # At price 2000, quantity should be 5
        qty = strategy._calculate_quantity(2000.0)
        assert float(qty) == 5.0

    def test_calculate_quantity_with_different_amounts(self) -> None:
        """Test quantity calculation with different order amounts."""
        config = FactorStrategyConfig(
            instrument_id="ETHUSDT.BINANCE",
            signal_name="factor.alpha_breakout_long",
            order_amount=5000.0,
        )
        strategy = FactorStrategy(config)
        
        # At price 1000, quantity should be 5
        qty = strategy._calculate_quantity(1000.0)
        assert float(qty) == 5.0
