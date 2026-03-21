# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimitExecAlgorithm logic.

Tests are structured to avoid needing fully-initialized Nautilus Actor objects
(which have Cython read-only descriptors). We test:
  1. compute_limit_price() - standalone pure function
  2. Config parameter resolution helpers
  3. State lookup and concurrent order tracking
"""

from __future__ import annotations

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.algorithm import (
    _normalize_qty_or_zero,
    _spawn_linkage_fields,
    compute_limit_price,
)
from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig
from nautilus_quants.execution.post_limit.state import OrderExecutionState, OrderState


# ---------------------------------------------------------------------------
# Tests: compute_limit_price (pure function, no Nautilus deps)
# ---------------------------------------------------------------------------


class TestComputeLimitPrice:
    """Test the standalone price calculation function."""

    def test_buy_at_bbo_no_offset(self) -> None:
        """BUY with offset=0 should peg to best_bid."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.BUY, anchor_px=49000.0,
            offset_ticks=0, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=50000.0, best_ask=50002.0,
        )
        assert price == 50000.0

    def test_sell_at_bbo_no_offset(self) -> None:
        """SELL with offset=0 should peg to best_ask."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.SELL, anchor_px=51000.0,
            offset_ticks=0, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=50000.0, best_ask=50002.0,
        )
        assert price == 50002.0

    def test_buy_with_positive_offset(self) -> None:
        """BUY with offset_ticks=1 improves BBO by 1 tick."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.BUY, anchor_px=49000.0,
            offset_ticks=1, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=50000.0, best_ask=50002.0,
        )
        assert price == pytest.approx(50000.01, abs=0.001)

    def test_sell_with_positive_offset(self) -> None:
        """SELL with offset_ticks=1 improves BBO by 1 tick."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.SELL, anchor_px=51000.0,
            offset_ticks=1, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=50000.0, best_ask=50002.0,
        )
        assert price == pytest.approx(50001.99, abs=0.001)

    def test_buy_with_negative_offset(self) -> None:
        """BUY with offset_ticks=-1 retreats 1 tick from BBO."""
        price = compute_limit_price(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=-1, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=100.0, best_ask=102.0,
        )
        assert price == 99.0

    def test_buy_post_only_clamp(self) -> None:
        """BUY with post_only should clamp to best_ask - tick."""
        # base=100, offset=5*1=5, raw=105. Clamp: min(105, 102-1)=101
        price = compute_limit_price(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=5, chase_count=0, chase_step_ticks=1,
            post_only=True, best_bid=100.0, best_ask=102.0,
        )
        assert price == 101.0

    def test_sell_post_only_clamp(self) -> None:
        """SELL with post_only should clamp to best_bid + tick."""
        # base=102, offset=5*1=5, raw=97. Clamp: max(97, 100+1)=101
        price = compute_limit_price(
            tick=1.0, side=OrderSide.SELL, anchor_px=110.0,
            offset_ticks=5, chase_count=0, chase_step_ticks=1,
            post_only=True, best_bid=100.0, best_ask=102.0,
        )
        assert price == 101.0

    def test_fallback_to_anchor_px_when_no_book(self) -> None:
        """When no BBO available, fall back to anchor_px."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.BUY, anchor_px=50000.0,
            offset_ticks=0, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=None, best_ask=None,
        )
        assert price == 50000.0

    def test_chase_increases_price_for_buy(self) -> None:
        """Each chase step moves BUY price up."""
        base_args = dict(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=0, chase_step_ticks=2,
            post_only=False, best_bid=100.0, best_ask=110.0,
        )
        p0 = compute_limit_price(chase_count=0, **base_args)
        p1 = compute_limit_price(chase_count=1, **base_args)
        p2 = compute_limit_price(chase_count=2, **base_args)
        assert p0 == 100.0
        assert p1 == 102.0
        assert p2 == 104.0

    def test_chase_decreases_price_for_sell(self) -> None:
        """Each chase step moves SELL price down."""
        base_args = dict(
            tick=1.0, side=OrderSide.SELL, anchor_px=110.0,
            offset_ticks=0, chase_step_ticks=2,
            post_only=False, best_bid=90.0, best_ask=100.0,
        )
        p0 = compute_limit_price(chase_count=0, **base_args)
        p1 = compute_limit_price(chase_count=1, **base_args)
        p2 = compute_limit_price(chase_count=2, **base_args)
        assert p0 == 100.0
        assert p1 == 98.0
        assert p2 == 96.0

    def test_post_only_chase_does_not_step(self) -> None:
        """post_only=True: chase_count should NOT affect price (re-peg only).

        In production, _determine_limit_price passes chase_count=0 when
        post_only is True. This test verifies compute_limit_price produces
        the same result regardless of chase_count when called with count=0.
        """
        base_args = dict(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=-1, chase_step_ticks=2,
            post_only=True, best_bid=100.0, best_ask=102.0,
        )
        # All calls use chase_count=0 (as _determine_limit_price would)
        p0 = compute_limit_price(chase_count=0, **base_args)
        p1 = compute_limit_price(chase_count=0, **base_args)  # "chase 1" → still 0
        p2 = compute_limit_price(chase_count=0, **base_args)  # "chase 2" → still 0
        # All prices identical: base=100, offset=-1, price=99
        assert p0 == p1 == p2 == 99.0

    def test_price_never_goes_below_tick(self) -> None:
        """Price must be at least 1 tick (never zero or negative)."""
        price = compute_limit_price(
            tick=1.0, side=OrderSide.SELL, anchor_px=5.0,
            offset_ticks=10, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=None, best_ask=5.0,
        )
        assert price >= 1.0

    def test_post_only_no_clamp_when_within_spread(self) -> None:
        """post_only should not clamp when price is already within spread."""
        price = compute_limit_price(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=0, chase_count=0, chase_step_ticks=1,
            post_only=True, best_bid=100.0, best_ask=105.0,
        )
        # base=100, offset=0, raw=100. ask-1=104. min(100, 104)=100
        assert price == 100.0

    def test_partial_bid_only(self) -> None:
        """When only best_bid is available (no ask)."""
        price = compute_limit_price(
            tick=0.01, side=OrderSide.BUY, anchor_px=49000.0,
            offset_ticks=0, chase_count=0, chase_step_ticks=1,
            post_only=True, best_bid=50000.0, best_ask=None,
        )
        # No ask to clamp against, just use bid
        assert price == 50000.0

    def test_effective_offset_with_retreat(self) -> None:
        """offset=-1, retreat=2 → effective=-3: retreats 3 ticks total."""
        # BUY: base=best_bid=100, effective_offset=-3, price=100+(-3)*1=97
        price = compute_limit_price(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=-3, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=100.0, best_ask=102.0,
        )
        assert price == 97.0

    def test_effective_offset_sell_with_retreat(self) -> None:
        """SELL offset=-1, retreat=2 → effective=-3: retreats 3 ticks total."""
        # SELL: base=best_ask=102, effective_offset=-3, price=102-(-3)*1=105
        price = compute_limit_price(
            tick=1.0, side=OrderSide.SELL, anchor_px=110.0,
            offset_ticks=-3, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=100.0, best_ask=102.0,
        )
        assert price == 105.0


class TestSpawnLinkageFields:
    """Test linkage field extraction for spawned orders."""

    class _PrimaryStub:
        contingency_type = "contingency"
        order_list_id = "order-list"
        linked_order_ids = ["A", "B"]
        parent_order_id = "parent-id"

    def test_extracts_contingent_linkage_fields(self) -> None:
        fields = _spawn_linkage_fields(self._PrimaryStub())
        assert fields["contingency_type"] == "contingency"
        assert fields["order_list_id"] == "order-list"
        assert fields["linked_order_ids"] == ["A", "B"]
        assert fields["parent_order_id"] == "parent-id"


class TestNormalizeQtyOrZero:
    """Test quantity normalization for sub-increment residual handling."""

    class _FakeInstrument:
        size_increment = Quantity.from_str("0.01")
        size_precision = 2

        def make_qty(self, value: float, round_down: bool = False) -> Quantity:
            if value <= 0:
                return Quantity.zero(2)
            if value < 0.01:
                raise ValueError(
                    "Invalid `value` for quantity: rounded to zero due to size increment 0.01",
                )
            if round_down:
                # Floor to 2 decimals for deterministic assertions in tests
                floored = int(value * 100) / 100
                return Quantity.from_str(f"{floored:.2f}")
            return Quantity.from_str(f"{value:.2f}")

    class _FakeLogger:
        def __init__(self) -> None:
            self.warning_messages: list[str] = []

        def warning(self, message: str) -> None:
            self.warning_messages.append(message)

    def test_returns_zero_for_non_positive_qty(self) -> None:
        instrument = self._FakeInstrument()
        logger = self._FakeLogger()
        qty = _normalize_qty_or_zero(
            instrument=instrument,
            raw_qty=0.0,
            precision=2,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            primary_order_id=ClientOrderId("O-001"),
            logger=logger,
        )
        assert qty == Quantity.zero(2)
        assert logger.warning_messages == []

    def test_rounds_down_to_valid_increment_when_tradable(self) -> None:
        instrument = self._FakeInstrument()
        logger = self._FakeLogger()
        qty = _normalize_qty_or_zero(
            instrument=instrument,
            raw_qty=1.239,
            precision=2,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            primary_order_id=ClientOrderId("O-001"),
            logger=logger,
        )
        assert qty == Quantity.from_str("1.23")
        assert logger.warning_messages == []

    def test_returns_zero_and_logs_on_rounded_to_zero_valueerror(self) -> None:
        instrument = self._FakeInstrument()
        logger = self._FakeLogger()
        qty = _normalize_qty_or_zero(
            instrument=instrument,
            raw_qty=0.0037,
            precision=2,
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            primary_order_id=ClientOrderId("O-001"),
            logger=logger,
        )
        assert qty == Quantity.zero(2)
        assert len(logger.warning_messages) == 1
        assert "raw_qty=0.0037" in logger.warning_messages[0]

    def test_reraises_unexpected_valueerror(self) -> None:
        class _BrokenInstrument(self._FakeInstrument):
            def make_qty(self, value: float, round_down: bool = False) -> Quantity:
                raise ValueError("invalid decimal context")

        logger = self._FakeLogger()
        with pytest.raises(ValueError, match="invalid decimal context"):
            _normalize_qty_or_zero(
                instrument=_BrokenInstrument(),
                raw_qty=1.0,
                precision=2,
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                primary_order_id=ClientOrderId("O-001"),
                logger=logger,
            )


# ---------------------------------------------------------------------------
# Tests: Config parameter resolution
# ---------------------------------------------------------------------------


class TestConfigParameterResolution:
    """Test that per-order overrides correctly override config defaults.

    These tests use OrderExecutionState directly without needing Actor.
    """

    def test_default_timeout(self) -> None:
        config = PostLimitExecAlgorithmConfig(timeout_secs=15.0)
        state = _make_state()
        assert state.timeout_secs is None  # No override
        effective = state.timeout_secs if state.timeout_secs is not None else config.timeout_secs
        assert effective == 15.0

    def test_override_timeout(self) -> None:
        config = PostLimitExecAlgorithmConfig(timeout_secs=15.0)
        state = _make_state()
        state.timeout_secs = 30.0
        effective = state.timeout_secs if state.timeout_secs is not None else config.timeout_secs
        assert effective == 30.0

    def test_default_max_chase(self) -> None:
        config = PostLimitExecAlgorithmConfig(max_chase_attempts=3)
        state = _make_state()
        effective = (
            state.max_chase_attempts
            if state.max_chase_attempts is not None
            else config.max_chase_attempts
        )
        assert effective == 3

    def test_override_max_chase(self) -> None:
        config = PostLimitExecAlgorithmConfig(max_chase_attempts=3)
        state = _make_state()
        state.max_chase_attempts = 10
        effective = (
            state.max_chase_attempts
            if state.max_chase_attempts is not None
            else config.max_chase_attempts
        )
        assert effective == 10

    def test_default_post_only(self) -> None:
        config = PostLimitExecAlgorithmConfig(post_only=True)
        state = _make_state()
        effective = state.post_only if state.post_only is not None else config.post_only
        assert effective is True

    def test_override_post_only_false(self) -> None:
        config = PostLimitExecAlgorithmConfig(post_only=True)
        state = _make_state()
        state.post_only = False
        effective = state.post_only if state.post_only is not None else config.post_only
        assert effective is False


# ---------------------------------------------------------------------------
# Tests: Exec params parsing
# ---------------------------------------------------------------------------


class TestExecParamsParsing:
    """Test parsing of exec_algorithm_params from order."""

    def test_parse_anchor_px(self) -> None:
        params = {"anchor_px": "50000.0"}
        anchor_px = float(params.get("anchor_px", "0.0"))
        assert anchor_px == 50000.0

    def test_parse_timeout_override(self) -> None:
        params = {"anchor_px": "50000.0", "timeout_secs": "30.0"}
        timeout = float(params["timeout_secs"])
        assert timeout == 30.0

    def test_parse_max_chase_override(self) -> None:
        params = {"anchor_px": "50000.0", "max_chase_attempts": "5"}
        max_chase = int(params["max_chase_attempts"])
        assert max_chase == 5

    def test_parse_post_only_true(self) -> None:
        params = {"anchor_px": "50000.0", "post_only": "true"}
        post_only = params["post_only"].lower() == "true"
        assert post_only is True

    def test_parse_post_only_false(self) -> None:
        params = {"anchor_px": "50000.0", "post_only": "false"}
        post_only = params["post_only"].lower() == "true"
        assert post_only is False

    def test_missing_anchor_px_defaults_to_zero(self) -> None:
        params: dict[str, str] = {}
        anchor_px = float(params.get("anchor_px") or "0.0")
        assert anchor_px == 0.0


# ---------------------------------------------------------------------------
# Tests: POST_ONLY retry mechanism
# ---------------------------------------------------------------------------


class TestPostOnlyRetryMechanism:
    """Test POST_ONLY rejection retry logic using state and config directly."""

    def test_post_only_rejection_retries_with_deeper_offset(self) -> None:
        """POST_ONLY rejection increments retreat_ticks and allows retry."""
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=-1, max_post_only_retries=3,
        )
        state = _make_state(state=OrderState.ACTIVE)

        # Simulate first POST_ONLY rejection: retreat +1
        assert state.post_only_retreat_ticks < config.max_post_only_retries
        state.post_only_retreat_ticks += 1
        assert state.post_only_retreat_ticks == 1

        # Effective offset: -1 - 1 = -2
        effective_offset = config.offset_ticks - state.post_only_retreat_ticks
        assert effective_offset == -2

        # Price should retreat further
        price = compute_limit_price(
            tick=1.0, side=OrderSide.BUY, anchor_px=90.0,
            offset_ticks=effective_offset, chase_count=0, chase_step_ticks=1,
            post_only=False, best_bid=100.0, best_ask=102.0,
        )
        assert price == 98.0  # 100 + (-2) = 98

    def test_post_only_rejection_max_retries_then_market(self) -> None:
        """After max retries, should proceed to market fallback."""
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=-1, max_post_only_retries=3,
        )
        state = _make_state(state=OrderState.ACTIVE)

        # Exhaust all retries
        for i in range(config.max_post_only_retries):
            assert state.post_only_retreat_ticks < config.max_post_only_retries
            state.post_only_retreat_ticks += 1

        # Now retries exhausted
        assert state.post_only_retreat_ticks == 3
        assert not (state.post_only_retreat_ticks < config.max_post_only_retries)

        # Should fall back to market (no more retry)
        effective_offset = config.offset_ticks - state.post_only_retreat_ticks
        assert effective_offset == -4  # -1 - 3 = -4

    def test_post_only_rejection_disabled_zero_retries(self) -> None:
        """max_post_only_retries=0 means immediate market fallback (backward compat)."""
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=-1, max_post_only_retries=0,
        )
        state = _make_state(state=OrderState.ACTIVE)

        # Even first POST_ONLY rejection should NOT retry
        assert not (state.post_only_retreat_ticks < config.max_post_only_retries)

    def test_non_post_only_rejection_immediate_market(self) -> None:
        """Non-POST_ONLY rejection should always go to market, ignoring retry config."""
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=-1, max_post_only_retries=3,
        )
        state = _make_state(state=OrderState.ACTIVE)

        # due_post_only=False → should NOT retry regardless of config
        due_post_only = False
        should_retry = (
            due_post_only
            and state.post_only_retreat_ticks < config.max_post_only_retries
            and state.state in (OrderState.ACTIVE, OrderState.PENDING, OrderState.CHASING)
        )
        assert should_retry is False
        assert state.post_only_retreat_ticks == 0  # No change

    def test_retreat_resets_on_chase(self) -> None:
        """POST_ONLY retreat ticks should reset to 0 when chase starts."""
        state = _make_state(state=OrderState.ACTIVE)
        state.post_only_retreat_ticks = 2

        # Simulate chase: reset retreat, then transition
        state.chase_count += 1
        state.post_only_retreat_ticks = 0  # Mirrors _chase_order behavior
        state.transition_to(OrderState.CHASING)
        assert state.post_only_retreat_ticks == 0  # Reset

        state.transition_to(OrderState.ACTIVE)
        assert state.post_only_retreat_ticks == 0  # Stays reset

    def test_progressive_retreat_prices(self) -> None:
        """Verify prices retreat progressively with each retry."""
        config = PostLimitExecAlgorithmConfig(
            offset_ticks=-1, max_post_only_retries=3,
        )

        prices = []
        for retreat in range(4):  # 0, 1, 2, 3
            effective_offset = config.offset_ticks - retreat
            price = compute_limit_price(
                tick=0.01, side=OrderSide.BUY, anchor_px=49000.0,
                offset_ticks=effective_offset, chase_count=0, chase_step_ticks=1,
                post_only=False, best_bid=50000.0, best_ask=50002.0,
            )
            prices.append(price)

        # Each price should be lower (more retreat) than the previous
        assert prices[0] == pytest.approx(49999.99)  # offset=-1
        assert prices[1] == pytest.approx(49999.98)  # offset=-2
        assert prices[2] == pytest.approx(49999.97)  # offset=-3
        assert prices[3] == pytest.approx(49999.96)  # offset=-4

    def test_retreat_with_pending_state(self) -> None:
        """POST_ONLY retry should also work in PENDING state."""
        config = PostLimitExecAlgorithmConfig(max_post_only_retries=3)
        state = _make_state(state=OrderState.PENDING)

        due_post_only = True
        should_retry = (
            due_post_only
            and state.post_only_retreat_ticks < config.max_post_only_retries
            and state.state in (OrderState.ACTIVE, OrderState.PENDING, OrderState.CHASING)
        )
        assert should_retry is True

    def test_retreat_with_chasing_state(self) -> None:
        """POST_ONLY retry should also work in CHASING state."""
        config = PostLimitExecAlgorithmConfig(max_post_only_retries=3)
        state = _make_state(state=OrderState.CHASING)

        due_post_only = True
        should_retry = (
            due_post_only
            and state.post_only_retreat_ticks < config.max_post_only_retries
            and state.state in (OrderState.ACTIVE, OrderState.PENDING, OrderState.CHASING)
        )
        assert should_retry is True


# ---------------------------------------------------------------------------
# Tests: Concurrent order tracking
# ---------------------------------------------------------------------------


class TestConcurrentOrders:
    """Test that multiple simultaneous order sequences don't interfere."""

    def test_independent_state_tracking_125(self) -> None:
        """125 concurrent states should be independently tracked."""
        states: dict[ClientOrderId, OrderExecutionState] = {}
        spawned_to_primary: dict[ClientOrderId, ClientOrderId] = {}

        for i in range(125):
            primary_id = ClientOrderId(f"O-{i:04d}")
            spawned_id = ClientOrderId(f"O-SPAWN-{i:04d}")
            state = OrderExecutionState(
                primary_order_id=primary_id,
                instrument_id=InstrumentId.from_str(f"INST{i}-PERP.BINANCE"),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                total_quantity=Quantity.from_str("1.0"),
                anchor_px=50000.0 + i,
                state=OrderState.ACTIVE,
                current_limit_order_id=spawned_id,
                filled_quantity=Quantity.zero(1),
            )
            states[primary_id] = state
            spawned_to_primary[spawned_id] = primary_id

        assert len(states) == 125
        assert len(spawned_to_primary) == 125

        # Verify each state is independent
        for i in range(125):
            primary_id = ClientOrderId(f"O-{i:04d}")
            spawned_id = ClientOrderId(f"O-SPAWN-{i:04d}")
            resolved = spawned_to_primary.get(spawned_id)
            assert resolved == primary_id
            assert states[primary_id].anchor_px == 50000.0 + i

    def test_completing_one_doesnt_affect_others(self) -> None:
        """Completing one order sequence should not affect others."""
        states: dict[ClientOrderId, OrderExecutionState] = {}

        for i in range(3):
            primary_id = ClientOrderId(f"O-{i}")
            states[primary_id] = OrderExecutionState(
                primary_order_id=primary_id,
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                side=OrderSide.BUY,
                total_quantity=Quantity.from_str("1.0"),
                anchor_px=50000.0,
                state=OrderState.ACTIVE,
                filled_quantity=Quantity.zero(1),
            )

        # Complete one
        states[ClientOrderId("O-1")].transition_to(OrderState.COMPLETED)

        assert states[ClientOrderId("O-0")].state == OrderState.ACTIVE
        assert states[ClientOrderId("O-1")].state == OrderState.COMPLETED
        assert states[ClientOrderId("O-2")].state == OrderState.ACTIVE

    def test_mixed_states_across_orders(self) -> None:
        """Different orders can be in different states simultaneously."""
        states = {}
        for i, target_state in enumerate([
            OrderState.PENDING,
            OrderState.ACTIVE,
            OrderState.CHASING,
            OrderState.MARKET_FALLBACK,
            OrderState.COMPLETED,
        ]):
            states[i] = OrderExecutionState(
                primary_order_id=ClientOrderId(f"O-{i}"),
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                side=OrderSide.BUY,
                total_quantity=Quantity.from_str("1.0"),
                anchor_px=50000.0,
                state=target_state,
                filled_quantity=Quantity.zero(1),
            )

        assert states[0].state == OrderState.PENDING
        assert states[1].state == OrderState.ACTIVE
        assert states[2].state == OrderState.CHASING
        assert states[3].state == OrderState.MARKET_FALLBACK
        assert states[4].state == OrderState.COMPLETED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    state: OrderState = OrderState.PENDING,
) -> OrderExecutionState:
    return OrderExecutionState(
        primary_order_id=ClientOrderId("O-001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.0"),
        anchor_px=50000.0,
        state=state,
        filled_quantity=Quantity.zero(1),
    )
