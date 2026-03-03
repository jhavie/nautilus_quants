# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Unit tests for WorldQuantAlphaStrategy and WorldQuantAlphaConfig.

TDD-first: tests are written before implementation.
These tests cover the 7-step BRAIN portfolio construction pipeline.
"""

import math

import pytest

from nautilus_quants.factors.operators.cross_sectional import CsRank
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
        # Window should not exceed decay = 3 (BRAIN: N periods, not N+1)
        assert len(strategy._alpha_history) <= 3

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


def _make_strategy(
    neutralization: str = "MARKET",
    decay: int = 0,
    truncation: float = 0.0,
    delay: int = 0,
) -> WorldQuantAlphaStrategy:
    """Helper: create a strategy with given pipeline settings."""
    config = WorldQuantAlphaConfig(
        instrument_ids=["A.BINANCE", "B.BINANCE", "C.BINANCE"],
        delay=delay,
        decay=decay,
        neutralization=neutralization,
        truncation=truncation,
    )
    return WorldQuantAlphaStrategy(config)


class TestBrainDocAlignment:
    """
    BRAIN platform documentation alignment tests.

    Based on the 8-stock rank(-returns) example from BRAIN docs
    (fig_03–fig_06, fig_10, simulation_settings.md).

    Stock ranks: S0=0/7, S1=1/7, ..., S7=7/7.
    """

    def test_neutralize_8stock_rank_returns_exact_values(self) -> None:
        """
        Verify neutralize() matches BRAIN doc F-column exactly.

        Input (D col): rank(-returns) = i/7 for i in 0..7
        Expected (F col): D - mean(D) = i/7 - 0.5 = (2i-7)/14
        """
        strategy = _make_strategy()
        alpha = {f"s{i}": i / 7 for i in range(8)}
        result = strategy._neutralize(alpha)

        # Market neutral: sum must equal zero
        assert abs(sum(result.values())) < 1e-10

        # Exact BRAIN F-column values: (2i - 7) / 14
        assert result["s0"] == pytest.approx(-7 / 14)   # -0.5000
        assert result["s7"] == pytest.approx(7 / 14)    # +0.5000
        assert result["s3"] == pytest.approx(-1 / 14)   # -0.0714
        assert result["s4"] == pytest.approx(1 / 14)    # +0.0714
        assert result["s1"] == pytest.approx(-5 / 14)   # -0.3571
        assert result["s6"] == pytest.approx(5 / 14)    # +0.3571

    def test_scale_abs_sum_matches_brain_doc_16_over_7(self) -> None:
        """
        Verify scale() matches BRAIN doc G/H columns exactly.

        BRAIN doc G-col: sum(|neutralized|) = 16/7 ≈ 2.286 (shown as "2.3")
        BRAIN doc H-col: scaled = neutralized / (16/7) = (2i-7)/32
        """
        strategy = _make_strategy()
        neutralized = {f"s{i}": i / 7 - 0.5 for i in range(8)}

        # Verify the pre-scale abs sum matches the BRAIN doc value exactly
        abs_before = sum(abs(v) for v in neutralized.values())
        assert abs_before == pytest.approx(16 / 7)

        result = strategy._scale(neutralized)

        # After scaling, sum(|w|) == 1
        assert sum(abs(v) for v in result.values()) == pytest.approx(1.0)

        # Exact BRAIN H-column values: (2i - 7) / 32
        assert result["s0"] == pytest.approx(-7 / 32)   # ≈ −0.2188 (doc "−0.22")
        assert result["s7"] == pytest.approx(7 / 32)    # ≈ +0.2188 (doc "+0.22")
        assert result["s3"] == pytest.approx(-1 / 32)   # ≈ −0.0313 (doc "−0.03")
        assert result["s4"] == pytest.approx(1 / 32)    # ≈ +0.0313 (doc "+0.03")
        assert result["s2"] == pytest.approx(-3 / 32)   # ≈ −0.0938 (doc "−0.09")
        assert result["s5"] == pytest.approx(3 / 32)    # ≈ +0.0938 (doc "+0.09")

    def test_apply_decay_steady_state_uses_exactly_n_periods(self) -> None:
        """
        Verify decay=2 in steady state retains exactly 2 periods (not 3).

        BRAIN formula: Decay_linear(x,N) uses N periods with weights 1..N.
        After the 3rd call, oldest entry (v1) must be evicted.
        """
        strategy = _make_strategy(decay=2)
        v1 = {"A": 0.6, "B": 0.4}
        v2 = {"A": 0.4, "B": 0.6}
        v3 = {"A": 0.5, "B": 0.5}

        strategy._apply_decay(v1)   # history = [v1]
        strategy._apply_decay(v2)   # history = [v1, v2] — window full
        result = strategy._apply_decay(v3)  # v1 evicted → history = [v2, v3]

        # History must contain exactly N=2 periods
        assert len(strategy._alpha_history) == 2

        # weights=[1,2], total=3; oldest=v2, newest=v3
        # A: (0.4×1 + 0.5×2) / 3 = 1.4/3
        # B: (0.6×1 + 0.5×2) / 3 = 1.6/3
        assert result["A"] == pytest.approx(1.4 / 3)
        assert result["B"] == pytest.approx(1.6 / 3)

    def test_apply_decay_3_periods_linear_formula(self) -> None:
        """
        Verify decay=3 matches BRAIN formula exactly (fig_10 + simulation_settings.md).

        BRAIN: Decay_linear(x,3) = (x_t×3 + x_{t-1}×2 + x_{t-2}×1) / 6
        """
        strategy = _make_strategy(decay=3)
        p1 = {"A": 0.2, "B": 0.5, "C": 0.3}   # t-2, weight=1
        p2 = {"A": 0.4, "B": 0.3, "C": 0.3}   # t-1, weight=2
        p3 = {"A": 0.6, "B": 0.2, "C": 0.2}   # t,   weight=3

        strategy._apply_decay(p1)
        strategy._apply_decay(p2)
        result = strategy._apply_decay(p3)

        # A: (0.2×1 + 0.4×2 + 0.6×3) / 6 = (0.2 + 0.8 + 1.8) / 6 = 2.8/6
        # B: (0.5×1 + 0.3×2 + 0.2×3) / 6 = (0.5 + 0.6 + 0.6) / 6 = 1.7/6
        # C: (0.3×1 + 0.3×2 + 0.2×3) / 6 = (0.3 + 0.6 + 0.6) / 6 = 1.5/6
        assert result["A"] == pytest.approx(2.8 / 6)
        assert result["B"] == pytest.approx(1.7 / 6)
        assert result["C"] == pytest.approx(1.5 / 6)

        # Sum conservation: each input sums to 1.0
        assert sum(result.values()) == pytest.approx(1.0)

    def test_full_pipeline_8stock_brain_example(self) -> None:
        """
        End-to-end: neutralize → scale → decay=0 → verify BRAIN doc H-column.

        Input: rank(-returns) for 8 stocks = i/7
        Expected output: (2i-7)/32 (BRAIN H-column exact fractions)
        """
        strategy = _make_strategy(neutralization="MARKET", decay=0, truncation=0.0)
        alpha = {f"s{i}": i / 7 for i in range(8)}

        result = strategy._process_alpha(alpha)

        # Portfolio constraints
        assert sum(abs(v) for v in result.values()) == pytest.approx(1.0)
        assert abs(sum(result.values())) < 1e-10

        # Exact BRAIN H-column values
        assert result["s0"] == pytest.approx(-7 / 32)   # doc "−0.22"
        assert result["s7"] == pytest.approx(7 / 32)    # doc "+0.22"
        assert result["s3"] == pytest.approx(-1 / 32)   # doc "−0.03"
        assert result["s4"] == pytest.approx(1 / 32)    # doc "+0.03"

    def test_capital_allocation_20m_brain_example(self) -> None:
        """
        Verify $20M capital allocation matches BRAIN documentation.

        BRAIN doc: positions = weight × $20M
        Exact fractions: (2i-7)/32 × $20M = (2i-7) × $625,000
        Min position: ±$625,000 (doc "±0.6M")
        Max position: ±$4,375,000 (doc "±4.4M")
        """
        CAPITAL = 20_000_000

        strategy = _make_strategy(neutralization="MARKET", decay=0, truncation=0.0)
        alpha = {f"s{i}": i / 7 for i in range(8)}
        weights = strategy._process_alpha(alpha)

        positions = {k: v * CAPITAL for k, v in weights.items()}

        # Exact position values: (2i-7) × $625,000
        assert positions["s0"] == pytest.approx(-7 * 625_000)   # −$4,375,000 (doc "−4.4M")
        assert positions["s7"] == pytest.approx(7 * 625_000)    # +$4,375,000 (doc "+4.4M")
        assert positions["s3"] == pytest.approx(-1 * 625_000)   # −$625,000 (doc "−0.6M")
        assert positions["s4"] == pytest.approx(1 * 625_000)    # +$625,000 (doc "+0.6M")

        # Long/short totals each equal $10M (dollar-neutral)
        long_total = sum(v for v in positions.values() if v > 0)
        short_total = sum(v for v in positions.values() if v < 0)
        assert long_total == pytest.approx(10_000_000)
        assert short_total == pytest.approx(-10_000_000)


class TestBrainRawReturnsToPositions:
    """
    E2E 测试：原始收益率 → CsRank（因子引擎 CS 层）→ 策略 pipeline → 资本分配。

    假设收益率数据：8 只股票，等差 [+7%, +5%, ..., −7%]
    设计目的：rank(-returns) 精确还原 BRAIN 文档 fig_03 D列 [0/7, ..., 7/7]。

    均值回归逻辑：
      - s0（最佳 +7%）：做空 → −$4,375,000
      - s7（最差 −7%）：做多 → +$4,375,000
      - 最小头寸：±$625,000（文档"±0.6M"）
    """

    # 假设收益率：等差列，确保产生均匀排名
    RETURNS = {f"s{i}": 0.07 - 0.02 * i for i in range(8)}
    # s0=+0.07, s1=+0.05, s2=+0.03, s3=+0.01
    # s4=−0.01, s5=−0.03, s6=−0.05, s7=−0.07

    def test_rank_neg_returns_matches_brain_doc_d_column(self) -> None:
        """验证 CsRank 直接调用：rank(-returns) 产生 [1/n, ..., 1] (popbo-aligned).

        CsRank 使用 rank(method='min', pct=True)，值域 [1/n, 1]。
        这与 pandas/popbo 约定一致（83 个 alpha101 精确匹配测试验证）。
        BRAIN 文档 D列用 [0/(n-1), 1] 简化展示，但实际 pipeline 结果相同。
        """
        neg_returns = {k: -v for k, v in self.RETURNS.items()}
        rank_result = CsRank().compute(neg_returns)

        # [1/8, 2/8, ..., 8/8] = [0.125, 0.25, ..., 1.0]
        for i in range(8):
            assert rank_result[f"s{i}"] == pytest.approx((i + 1) / 8)

    def test_full_chain_raw_returns_to_625k_positions(self) -> None:
        """完整链路：原始收益率 → rank(-returns) → pipeline → $20M 头寸
        逐列验证 BRAIN 文档 fig_03：
          D列: rank(-returns) = [(i+1)/8] (popbo-aligned [1/n, 1])
          H列: 缩放后权重     = (2i-7)/32
          头寸: × $20M        = ±k × $625,000
        """
        CAPITAL = 20_000_000

        # Step1: rank(-returns) via CsRank（因子引擎 CS 层）
        neg_returns = {k: -v for k, v in self.RETURNS.items()}
        rank_values = CsRank().compute(neg_returns)

        # 验证 D列 — [1/8, 2/8, ..., 8/8] (popbo-aligned)
        for i in range(8):
            assert rank_values[f"s{i}"] == pytest.approx((i + 1) / 8)

        # Step2-4: 策略 pipeline（neutralize → scale → positions）
        strategy = _make_strategy(neutralization="MARKET", decay=0, truncation=0.0)
        weights = strategy._process_alpha(rank_values)
        positions = {k: v * CAPITAL for k, v in weights.items()}

        # 验证 H列（权重精确分数）
        assert weights["s0"] == pytest.approx(-7 / 32)
        assert weights["s7"] == pytest.approx(+7 / 32)

        # 验证头寸（与文档四舍五入标注对应）
        assert positions["s0"] == pytest.approx(-7 * 625_000)   # 文档 "−4.4M"（做空赢家）
        assert positions["s7"] == pytest.approx(+7 * 625_000)   # 文档 "+4.4M"（做多输家）
        assert positions["s3"] == pytest.approx(-1 * 625_000)   # 文档 "−0.6M"
        assert positions["s4"] == pytest.approx(+1 * 625_000)   # 文档 "+0.6M" ← 最小头寸

        # 多空平衡（均值回归策略）
        long_total = sum(v for v in positions.values() if v > 0)
        short_total = sum(v for v in positions.values() if v < 0)
        assert long_total == pytest.approx(10_000_000)
        assert short_total == pytest.approx(-10_000_000)
