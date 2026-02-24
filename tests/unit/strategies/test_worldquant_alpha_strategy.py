# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Unit tests for WorldQuantAlphaStrategy and WorldQuantAlphaConfig.

TDD-first: tests are written before implementation.
These tests cover the 7-step BRAIN portfolio construction pipeline.
"""

import math

import pytest

from nautilus_quants.strategies.worldquant.strategy import (
    WorldQuantAlphaConfig,
    WorldQuantAlphaStrategy,
)


class TestWorldQuantAlphaConfig:
    """Tests for WorldQuantAlphaConfig defaults and construction."""

    def test_creates_with_required_fields(self) -> None:
        """Test creation with required fields only."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        assert len(config.instrument_ids) == 2
        assert config.instrument_ids[0] == "BTCUSDT.BINANCE"

    def test_defaults_match_brain_spec(self) -> None:
        """Test default values match BRAIN specification."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        assert config.factor_name == "alpha_101"
        assert config.delay == 1
        assert config.decay == 0
        assert config.neutralization == "MARKET"
        assert config.truncation == 0.0
        assert config.rebalance_period == 24
        assert config.capital == 100_000.0
        assert config.enable_long is True
        assert config.enable_short is True
        assert config.bar_types == []

    def test_creates_with_custom_values(self) -> None:
        """Test creation with custom configuration values."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            factor_name="my_alpha",
            delay=0,
            decay=5,
            neutralization="NONE",
            truncation=0.1,
            rebalance_period=4,
            capital=500_000.0,
            enable_long=True,
            enable_short=False,
        )
        assert config.factor_name == "my_alpha"
        assert config.delay == 0
        assert config.decay == 5
        assert config.neutralization == "NONE"
        assert config.truncation == 0.1
        assert config.rebalance_period == 4
        assert config.capital == 500_000.0
        assert config.enable_short is False


class TestWorldQuantAlphaStrategyInit:
    """Tests for WorldQuantAlphaStrategy initialization."""

    def test_init_stores_instrument_ids(self) -> None:
        """Test initialization stores instrument_ids correctly."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
        )
        strategy = WorldQuantAlphaStrategy(config)
        assert len(strategy._instrument_ids) == 2

    def test_init_state_is_clean(self) -> None:
        """Test initialization starts with clean state."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["BTCUSDT.BINANCE"],
        )
        strategy = WorldQuantAlphaStrategy(config)
        assert strategy._current_prices == {}
        assert strategy._long_positions == set()
        assert strategy._short_positions == set()
        assert strategy._prev_alpha is None
        assert strategy._alpha_history == []
        assert strategy._signal_count == 0


class TestNeutralize:
    """Step 4: Market neutralization - subtract mean so sum(alpha) = 0."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_neutralize_makes_sum_zero(self) -> None:
        """After neutralization, sum of values should be zero."""
        alpha = {"A": 0.6, "B": 0.4, "C": 0.2}
        result = self.strategy._neutralize(alpha)
        assert abs(sum(result.values())) < 1e-10

    def test_neutralize_preserves_relative_order(self) -> None:
        """Neutralization should preserve relative rankings."""
        alpha = {"A": 1.0, "B": 0.5, "C": 0.0}
        result = self.strategy._neutralize(alpha)
        assert result["A"] > result["B"] > result["C"]

    def test_neutralize_already_neutral(self) -> None:
        """Already neutral alpha should remain near zero sum."""
        alpha = {"A": 1.0, "B": 0.0, "C": -1.0}
        result = self.strategy._neutralize(alpha)
        assert abs(sum(result.values())) < 1e-10

    def test_neutralize_single_instrument(self) -> None:
        """Single instrument should result in zero after neutralization."""
        alpha = {"A": 0.5}
        result = self.strategy._neutralize(alpha)
        assert abs(result["A"]) < 1e-10

    def test_neutralize_with_negative_values(self) -> None:
        """Handles mixed positive/negative values."""
        alpha = {"A": 0.8, "B": -0.3, "C": -0.1}
        result = self.strategy._neutralize(alpha)
        assert abs(sum(result.values())) < 1e-10


class TestScale:
    """Step 5: Normalization - divide by sum(|alpha|) so sum(|alpha|) = 1."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_scale_makes_abs_sum_one(self) -> None:
        """After scaling, sum of absolute values should be 1."""
        alpha = {"A": 0.6, "B": 0.4, "C": -0.2}
        result = self.strategy._scale(alpha)
        total_abs = sum(abs(v) for v in result.values())
        assert abs(total_abs - 1.0) < 1e-10

    def test_scale_preserves_signs(self) -> None:
        """Scaling should not change the sign of values."""
        alpha = {"A": 0.6, "B": -0.4, "C": 0.2}
        result = self.strategy._scale(alpha)
        assert result["A"] > 0
        assert result["B"] < 0
        assert result["C"] > 0

    def test_scale_preserves_relative_ratios(self) -> None:
        """Scaling should preserve relative proportions."""
        alpha = {"A": 2.0, "B": 1.0, "C": 1.0}
        result = self.strategy._scale(alpha)
        assert abs(result["A"] / result["B"] - 2.0) < 1e-10

    def test_scale_zero_sum_returns_unchanged(self) -> None:
        """If all values are zero, return unchanged (avoid division by zero)."""
        alpha = {"A": 0.0, "B": 0.0}
        result = self.strategy._scale(alpha)
        assert result == alpha


class TestTruncate:
    """Step 6: Truncation - cap individual weights at threshold."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            truncation=0.5,
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_truncate_caps_max_weight(self) -> None:
        """No weight should exceed the truncation threshold."""
        alpha = {"A": 0.9, "B": 0.05, "C": 0.05}
        result = self.strategy._truncate(alpha)
        assert max(abs(v) for v in result.values()) <= 0.5 + 1e-10

    def test_truncate_caps_negative_weight(self) -> None:
        """Negative weights should also be capped."""
        alpha = {"A": -0.9, "B": 0.05, "C": 0.05}
        result = self.strategy._truncate(alpha)
        assert max(abs(v) for v in result.values()) <= 0.5 + 1e-10

    def test_truncate_within_threshold_unchanged(self) -> None:
        """Values within threshold should remain unchanged."""
        alpha = {"A": 0.3, "B": 0.2, "C": 0.1}
        result = self.strategy._truncate(alpha)
        assert result == alpha

    def test_truncate_uses_config_threshold(self) -> None:
        """Truncation should use the configured threshold."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            truncation=0.1,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.5, "B": -0.3, "C": 0.2}
        result = strategy._truncate(alpha)
        assert max(abs(v) for v in result.values()) <= 0.1 + 1e-10


class TestApplyDelay:
    """Step 2: Delay - use previous period's data (delay=1)."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE"],
            delay=1,
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_delay_1_returns_none_on_first_call(self) -> None:
        """With delay=1, first call returns None (warmup period)."""
        alpha = {"A": 0.5, "B": -0.5}
        result = self.strategy._apply_delay(alpha)
        assert result is None

    def test_delay_1_returns_prev_on_second_call(self) -> None:
        """With delay=1, second call returns first period's data."""
        alpha1 = {"A": 0.5, "B": -0.5}
        alpha2 = {"A": 0.3, "B": -0.3}
        self.strategy._apply_delay(alpha1)  # warmup → None
        result = self.strategy._apply_delay(alpha2)
        assert result == alpha1

    def test_delay_1_slides_correctly(self) -> None:
        """With delay=1, each call returns the previous period's data."""
        alpha1 = {"A": 0.1}
        alpha2 = {"A": 0.2}
        alpha3 = {"A": 0.3}

        self.strategy._apply_delay(alpha1)  # → None
        r2 = self.strategy._apply_delay(alpha2)  # → alpha1
        r3 = self.strategy._apply_delay(alpha3)  # → alpha2

        assert r2 == alpha1
        assert r3 == alpha2

    def test_delay_0_returns_current(self) -> None:
        """With delay=0, current period's data is used directly."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.5, "B": -0.5}
        result = strategy._apply_delay(alpha)
        assert result == alpha


class TestApplyDecay:
    """Step 3: Decay - linear decay weighted average."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
            decay=0,
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_decay_0_returns_current_unchanged(self) -> None:
        """With decay=0, no decay is applied; current value returned."""
        alpha = {"A": 0.5, "B": -0.5}
        result = self.strategy._apply_decay(alpha)
        assert result == alpha

    def test_decay_2_weighted_average(self) -> None:
        """With decay=2, weights [1, 2] for [oldest, newest]."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
            decay=2,
        )
        strategy = WorldQuantAlphaStrategy(config)

        # First period only: weight = [1]
        strategy._apply_decay({"A": 1.0})

        # Second period: weights [1, 2] for [first, second]
        result = strategy._apply_decay({"A": 3.0})
        # Expected: (1 * 1.0 + 2 * 3.0) / (1 + 2) = 7/3
        expected = (1 * 1.0 + 2 * 3.0) / 3
        assert abs(result["A"] - expected) < 1e-10

    def test_decay_accumulates_up_to_window(self) -> None:
        """History window should not exceed decay+1 periods."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
            decay=3,
        )
        strategy = WorldQuantAlphaStrategy(config)
        for _ in range(10):
            strategy._apply_decay({"A": 1.0})
        # Window should not exceed decay+1 = 4
        assert len(strategy._alpha_history) <= 4

    def test_decay_most_recent_has_highest_weight(self) -> None:
        """More recent values should have higher weight."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
            decay=2,
        )
        strategy = WorldQuantAlphaStrategy(config)
        strategy._apply_decay({"A": 0.0})   # old value
        result = strategy._apply_decay({"A": 1.0})  # recent value
        # With weights [1, 2], result should be closer to 1.0 than to 0.0
        assert result["A"] > 0.5


class TestProcessAlpha:
    """Integration test: full 7-step pipeline."""

    def test_market_neutral_result_sum_near_zero(self) -> None:
        """Market neutral strategy: sum of weights should be near zero."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            delay=0,
            decay=0,
            neutralization="MARKET",
            truncation=0.0,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.6, "B": 0.4, "C": 0.2, "D": -0.1}
        result = strategy._process_alpha(alpha)
        assert abs(sum(result.values())) < 1e-10

    def test_scaled_result_abs_sum_equals_one(self) -> None:
        """After scale, sum of absolute values should be 1."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            delay=0,
            decay=0,
            neutralization="MARKET",
            truncation=0.0,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.6, "B": 0.2, "C": -0.2}
        result = strategy._process_alpha(alpha)
        total_abs = sum(abs(v) for v in result.values())
        assert abs(total_abs - 1.0) < 1e-10

    def test_truncation_caps_weights(self) -> None:
        """After truncation + rescale, max |weight| <= truncation."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE", "D.BINANCE"],
            delay=0,
            decay=0,
            neutralization="MARKET",
            truncation=0.3,
        )
        strategy = WorldQuantAlphaStrategy(config)
        # Create a skewed alpha where one value dominates
        alpha = {"A": 10.0, "B": 0.1, "C": -0.1, "D": 0.0}
        result = strategy._process_alpha(alpha)
        assert max(abs(v) for v in result.values()) <= 0.3 + 1e-10

    def test_none_neutralization_no_mean_subtraction(self) -> None:
        """With NONE neutralization, mean is not subtracted."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            delay=0,
            decay=0,
            neutralization="NONE",
            truncation=0.0,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.6, "B": 0.3, "C": 0.1}  # all positive
        result = strategy._process_alpha(alpha)
        # With NONE neutralization, all values should remain positive
        assert all(v > 0 for v in result.values())


class TestNoSignalDuringWarmup:
    """Test that warmup period works correctly for delay buffer."""

    def test_delay_1_warmup_period(self) -> None:
        """With delay=1, first period is warmup; _apply_delay returns None."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=1,
        )
        strategy = WorldQuantAlphaStrategy(config)

        # Simulate receiving first FactorValues - should be warmup
        raw = {"A": 0.5}
        result = strategy._apply_delay(raw)
        assert result is None

    def test_delay_0_no_warmup(self) -> None:
        """With delay=0, there is no warmup period."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
        )
        strategy = WorldQuantAlphaStrategy(config)

        raw = {"A": 0.5}
        result = strategy._apply_delay(raw)
        assert result is not None
        assert result == raw

    def test_rebalance_period_check(self) -> None:
        """Strategy should only rebalance when signal_count % rebalance_period == 0."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE"],
            delay=0,
            rebalance_period=4,
        )
        strategy = WorldQuantAlphaStrategy(config)

        # Signal count 0: should rebalance (0 % 4 == 0)
        assert strategy._should_rebalance() is True

        # After incrementing to 1
        strategy._signal_count = 1
        assert strategy._should_rebalance() is False

        # After incrementing to 4
        strategy._signal_count = 4
        assert strategy._should_rebalance() is True


class TestNanHandling:
    """Tests for NaN handling in alpha vectors."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_neutralize_filters_nan(self) -> None:
        """NaN values should be filtered before neutralization."""
        alpha = {"A": 0.5, "B": float("nan"), "C": -0.5}
        result = self.strategy._neutralize(alpha)
        assert "B" not in result or math.isnan(result.get("B", 0))
        assert abs(sum(v for v in result.values() if not math.isnan(v))) < 1e-10

    def test_process_alpha_handles_all_nan(self) -> None:
        """If all values are NaN, returns empty dict."""
        alpha = {"A": float("nan"), "B": float("nan")}
        result = self.strategy._process_alpha(alpha)
        assert result == {}


class TestProcessAlphaIntermediates:
    """Tests that _process_alpha captures intermediate pipeline values."""

    def setup_method(self) -> None:
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            delay=0,
            decay=0,
            neutralization="MARKET",
            truncation=0.0,
        )
        self.strategy = WorldQuantAlphaStrategy(config)

    def test_last_neutralized_is_set_after_process(self) -> None:
        """_last_neutralized should be populated after _process_alpha."""
        alpha = {"A": 0.6, "B": 0.4, "C": 0.2}
        self.strategy._process_alpha(alpha)
        assert self.strategy._last_neutralized != {}
        assert abs(sum(self.strategy._last_neutralized.values())) < 1e-10

    def test_last_scaled_abs_sum_one(self) -> None:
        """_last_scaled should have sum(|w|) = 1."""
        alpha = {"A": 0.6, "B": 0.4, "C": 0.2}
        self.strategy._process_alpha(alpha)
        total = sum(abs(v) for v in self.strategy._last_scaled.values())
        assert abs(total - 1.0) < 1e-10

    def test_last_decayed_equals_scaled_when_decay_zero(self) -> None:
        """With decay=0, _last_decayed should equal _last_scaled."""
        alpha = {"A": 0.6, "B": 0.2, "C": -0.2}
        self.strategy._process_alpha(alpha)
        assert self.strategy._last_decayed == self.strategy._last_scaled

    def test_last_neutralized_is_raw_alpha_when_neutralization_none(self) -> None:
        """With NONE neutralization, _last_neutralized equals the input alpha."""
        config = WorldQuantAlphaConfig(
            instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
            delay=0,
            decay=0,
            neutralization="NONE",
            truncation=0.0,
        )
        strategy = WorldQuantAlphaStrategy(config)
        alpha = {"A": 0.6, "B": 0.3, "C": 0.1}
        strategy._process_alpha(alpha)
        assert strategy._last_neutralized == alpha

    def test_intermediates_are_distinct_stages(self) -> None:
        """The three intermediates represent distinct pipeline stages."""
        alpha = {"A": 0.6, "B": 0.2, "C": -0.2}
        self.strategy._process_alpha(alpha)
        # neutralized has sum≈0, scaled has sum(|w|)=1, decayed=scaled here
        assert abs(sum(self.strategy._last_neutralized.values())) < 1e-10
        assert abs(sum(abs(v) for v in self.strategy._last_scaled.values()) - 1.0) < 1e-10


class TestMetadataProviderIntermediates:
    """Tests for intermediate value storage in WorldQuantMetadataProvider."""

    def test_record_open_stores_intermediates(self) -> None:
        """record_open should store neutralized/scaled/decayed entry values."""
        from nautilus_quants.strategies.worldquant.metadata import WorldQuantMetadataProvider
        provider = WorldQuantMetadataProvider()
        provider.record_open(
            instrument_id="BTCUSDT.BINANCE",
            side="LONG",
            alpha101=0.7,
            weight=0.05,
            ts_event=1000,
            neutralized=0.2,
            scaled=0.03,
            decayed=0.05,
        )
        meta = provider.get_all_metadata()["BTCUSDT.BINANCE"]
        assert meta["entry_neutralized"] == 0.2
        assert meta["entry_scaled"] == 0.03
        assert meta["entry_decayed"] == 0.05

    def test_record_open_defaults_intermediates_to_none(self) -> None:
        """record_open should default intermediate values to None when not provided."""
        from nautilus_quants.strategies.worldquant.metadata import WorldQuantMetadataProvider
        provider = WorldQuantMetadataProvider()
        provider.record_open(
            instrument_id="ETHUSDT.BINANCE",
            side="SHORT",
            alpha101=0.3,
            weight=-0.04,
            ts_event=2000,
        )
        meta = provider.get_all_metadata()["ETHUSDT.BINANCE"]
        assert meta["entry_neutralized"] is None
        assert meta["entry_scaled"] is None
        assert meta["entry_decayed"] is None

    def test_update_history_stores_intermediates(self) -> None:
        """update_alpha101_history should store neutralized/scaled/decayed per entry."""
        from nautilus_quants.strategies.worldquant.metadata import WorldQuantMetadataProvider
        provider = WorldQuantMetadataProvider()
        provider.record_open(
            instrument_id="SOLUSDT.BINANCE",
            side="LONG",
            alpha101=0.8,
            weight=0.06,
            ts_event=1000,
        )
        provider.update_alpha101_history(
            instrument_ids=["SOLUSDT.BINANCE"],
            alpha101_lookup={"SOLUSDT.BINANCE": 0.5},
            weight_lookup={"SOLUSDT.BINANCE": 0.04},
            ts_event=2000,
            neutralized_lookup={"SOLUSDT.BINANCE": -0.1},
            scaled_lookup={"SOLUSDT.BINANCE": 0.02},
            decayed_lookup={"SOLUSDT.BINANCE": 0.04},
        )
        history = provider.get_all_metadata()["SOLUSDT.BINANCE"]["alpha101_history"]
        assert len(history) == 1
        assert history[0]["neutralized"] == -0.1
        assert history[0]["scaled"] == 0.02
        assert history[0]["decayed"] == 0.04


class TestMetadataRendererIntermediates:
    """Tests for intermediate column rendering in WorldQuantMetadataRenderer."""

    def setup_method(self) -> None:
        from nautilus_quants.strategies.worldquant.metadata import WorldQuantMetadataRenderer
        self.renderer = WorldQuantMetadataRenderer()

    def test_column_config_has_nine_columns(self) -> None:
        """Column config should include all 9 columns."""
        cols = self.renderer.get_column_config()
        keys = [c["key"] for c in cols]
        assert "neutralized_display" in keys
        assert "scaled_display" in keys
        assert "decayed_display" in keys
        assert len(cols) == 9

    def test_render_position_includes_intermediate_displays(self) -> None:
        """render_position should return neutralized/scaled/decayed display strings."""
        metadata = {
            "entry_alpha101": 0.7,
            "entry_neutralized": 0.268,
            "entry_scaled": 0.0046,
            "entry_decayed": 0.0084,
            "entry_weight": 0.009,
            "alpha101_history": [],
        }
        result = self.renderer.render_position(
            symbol="SOLUSDT",
            position_info={"side": "LONG", "ts_opened": 1000, "value": 500.0},
            metadata=metadata,
            timestamp_ns=999,
        )
        assert "neutralized_display" in result
        assert "scaled_display" in result
        assert "decayed_display" in result
        assert "0.268" in result["neutralized_display"]
        assert "0.0046" in result["scaled_display"]
        assert "0.0084" in result["decayed_display"]

    def test_render_position_shows_dash_when_intermediates_none(self) -> None:
        """render_position should show '-' when intermediate values are None."""
        metadata = {
            "entry_alpha101": 0.5,
            "entry_neutralized": None,
            "entry_scaled": None,
            "entry_decayed": None,
            "entry_weight": 0.04,
            "alpha101_history": [],
        }
        result = self.renderer.render_position(
            symbol="BTCUSDT",
            position_info={"side": "LONG", "ts_opened": 1000, "value": 1000.0},
            metadata=metadata,
            timestamp_ns=999,
        )
        assert result["neutralized_display"] == "-"
        assert result["scaled_display"] == "-"
        assert result["decayed_display"] == "-"

    def test_render_position_updates_current_from_history(self) -> None:
        """Current intermediate values should update from alpha101_history."""
        metadata = {
            "entry_alpha101": 0.7,
            "entry_neutralized": 0.2,
            "entry_scaled": 0.03,
            "entry_decayed": 0.05,
            "entry_weight": 0.05,
            "alpha101_history": [
                {
                    "ts_event": 500,
                    "alpha101": 0.5,
                    "neutralized": -0.1,
                    "scaled": 0.02,
                    "decayed": 0.04,
                    "weight": 0.04,
                }
            ],
        }
        result = self.renderer.render_position(
            symbol="ETHUSDT",
            position_info={"side": "LONG", "ts_opened": 100, "value": 800.0},
            metadata=metadata,
            timestamp_ns=600,
        )
        assert result["current_neutralized"] == -0.1
        assert result["current_scaled"] == 0.02
        assert result["current_decayed"] == 0.04
