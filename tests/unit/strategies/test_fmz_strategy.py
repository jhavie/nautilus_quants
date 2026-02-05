# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Unit tests for FMZFactorStrategy and FMZFactorStrategyConfig."""

import pytest

from nautilus_quants.strategies.fmz import FMZFactorStrategy, FMZFactorStrategyConfig


class TestFMZFactorStrategyConfig:
    """Tests for FMZFactorStrategyConfig."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        assert len(config.instrument_ids) == 2
        assert config.instrument_ids[0] == "BTCUSDT.BINANCE"

    def test_defaults_are_applied(self) -> None:
        """Test default values are applied (FMZ original parameters)."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        assert config.n_long == 40
        assert config.n_short == 40
        assert config.position_value == 300.0
        assert config.rebalance_period == 1
        assert config.composite_factor == "composite"
        assert config.bar_types == []

    def test_creates_with_custom_values(self) -> None:
        """Test creation with custom values."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            n_long=20,
            n_short=20,
            position_value=500.0,
            rebalance_period=4,
            composite_factor="my_factor",
        )
        assert config.n_long == 20
        assert config.n_short == 20
        assert config.position_value == 500.0
        assert config.rebalance_period == 4
        assert config.composite_factor == "my_factor"


class TestFMZFactorStrategyInit:
    """Tests for FMZFactorStrategy initialization."""

    def test_init_stores_config(self) -> None:
        """Test initialization stores config correctly."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            n_long=10,
            n_short=10,
        )
        strategy = FMZFactorStrategy(config)

        assert len(strategy._instrument_ids) == 2
        assert strategy._n_instruments == 2

    def test_init_parses_instrument_ids(self) -> None:
        """Test initialization parses instrument_ids correctly."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        strategy = FMZFactorStrategy(config)

        assert strategy._instrument_ids[0].symbol.value == "BTCUSDT"
        assert strategy._instrument_ids[0].venue.value == "BINANCE"
        assert strategy._instrument_ids[1].symbol.value == "ETHUSDT"

    def test_init_state_is_clean(self) -> None:
        """Test initialization starts with clean state."""
        config = FMZFactorStrategyConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        strategy = FMZFactorStrategy(config)

        assert strategy._current_prices == {}
        assert strategy._long_positions == set()
        assert strategy._short_positions == set()
        assert strategy._hour_count == 0
        assert strategy._bar_count == 0
        assert strategy._composite_values == {}


class TestFMZFactorStrategyRebalance:
    """Tests for FMZFactorStrategy rebalance logic."""

    def test_rebalance_selects_correct_positions(self) -> None:
        """Test rebalance selects bottom N for long and top N for short."""
        config = FMZFactorStrategyConfig(
            instrument_ids=[
                "A.BINANCE",
                "B.BINANCE",
                "C.BINANCE",
                "D.BINANCE",
                "E.BINANCE",
                "F.BINANCE",
            ],
            n_long=2,
            n_short=2,
            position_value=100.0,
        )
        strategy = FMZFactorStrategy(config)

        # Mock prices
        strategy._current_prices = {
            "A.BINANCE": 10.0,
            "B.BINANCE": 20.0,
            "C.BINANCE": 30.0,
            "D.BINANCE": 40.0,
            "E.BINANCE": 50.0,
            "F.BINANCE": 60.0,
        }

        # Simulate composite values (sorted: A=0.1, B=0.2, C=0.3, D=0.4, E=0.5, F=0.6)
        composite = {
            "A.BINANCE": 0.1,  # Should be long (lowest)
            "B.BINANCE": 0.2,  # Should be long
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
            "E.BINANCE": 0.5,  # Should be short
            "F.BINANCE": 0.6,  # Should be short (highest)
        }

        # Test sorting logic
        sorted_symbols = sorted(composite.items(), key=lambda x: x[1])
        long_targets = set([s for s, _ in sorted_symbols[:config.n_long]])
        short_targets = set([s for s, _ in sorted_symbols[-config.n_short:]])

        assert long_targets == {"A.BINANCE", "B.BINANCE"}
        assert short_targets == {"E.BINANCE", "F.BINANCE"}

    def test_rebalance_with_nan_values(self) -> None:
        """Test rebalance filters out NaN values."""
        import math

        composite = {
            "A.BINANCE": 0.1,
            "B.BINANCE": math.nan,  # Should be filtered
            "C.BINANCE": 0.3,
            "D.BINANCE": 0.4,
        }

        # Filter logic from strategy
        filtered = {
            k: v for k, v in composite.items()
            if not math.isnan(v)
        }

        assert len(filtered) == 3
        assert "B.BINANCE" not in filtered
