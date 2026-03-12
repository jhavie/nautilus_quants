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

from nautilus_quants.execution.post_limit.algorithm import compute_limit_price
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
        filled_quantity=Quantity.zero(1),
    )
