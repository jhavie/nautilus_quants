# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimit state machine transitions."""

from __future__ import annotations

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.state import (
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    OrderExecutionState,
    OrderState,
)


def _make_state(
    state: OrderState = OrderState.PENDING,
    chase_count: int = 0,
) -> OrderExecutionState:
    """Helper to create an OrderExecutionState for testing."""
    return OrderExecutionState(
        primary_order_id=ClientOrderId("O-001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.0"),
        anchor_px=50000.0,
        state=state,
        chase_count=chase_count,
        filled_quantity=Quantity.zero(1),
    )


class TestOrderStateEnum:
    """Test OrderState enum values."""

    def test_all_states_defined(self) -> None:
        states = {s.value for s in OrderState}
        expected = {"PENDING", "ACTIVE", "CHASING", "MARKET_FALLBACK", "COMPLETED", "FAILED"}
        assert states == expected

    def test_terminal_states(self) -> None:
        assert TERMINAL_STATES == {OrderState.COMPLETED, OrderState.FAILED}


class TestValidTransitions:
    """Test the VALID_TRANSITIONS map covers all states."""

    def test_all_states_have_transitions(self) -> None:
        for state in OrderState:
            assert state in VALID_TRANSITIONS

    def test_terminal_states_have_no_transitions(self) -> None:
        for state in TERMINAL_STATES:
            assert VALID_TRANSITIONS[state] == set()


class TestOrderExecutionStateTransitions:
    """Test OrderExecutionState.transition_to() validates transitions."""

    # --- Happy path transitions ---

    def test_pending_to_active(self) -> None:
        s = _make_state(OrderState.PENDING)
        s.transition_to(OrderState.ACTIVE)
        assert s.state == OrderState.ACTIVE

    def test_pending_to_failed(self) -> None:
        s = _make_state(OrderState.PENDING)
        s.transition_to(OrderState.FAILED)
        assert s.state == OrderState.FAILED

    def test_active_to_completed(self) -> None:
        s = _make_state(OrderState.ACTIVE)
        s.transition_to(OrderState.COMPLETED)
        assert s.state == OrderState.COMPLETED

    def test_active_to_chasing(self) -> None:
        s = _make_state(OrderState.ACTIVE)
        s.transition_to(OrderState.CHASING)
        assert s.state == OrderState.CHASING

    def test_active_to_market_fallback(self) -> None:
        s = _make_state(OrderState.ACTIVE)
        s.transition_to(OrderState.MARKET_FALLBACK)
        assert s.state == OrderState.MARKET_FALLBACK

    def test_chasing_to_active(self) -> None:
        s = _make_state(OrderState.CHASING)
        s.transition_to(OrderState.ACTIVE)
        assert s.state == OrderState.ACTIVE

    def test_chasing_to_completed(self) -> None:
        s = _make_state(OrderState.CHASING)
        s.transition_to(OrderState.COMPLETED)
        assert s.state == OrderState.COMPLETED

    def test_market_fallback_to_completed(self) -> None:
        s = _make_state(OrderState.MARKET_FALLBACK)
        s.transition_to(OrderState.COMPLETED)
        assert s.state == OrderState.COMPLETED

    def test_market_fallback_to_failed(self) -> None:
        s = _make_state(OrderState.MARKET_FALLBACK)
        s.transition_to(OrderState.FAILED)
        assert s.state == OrderState.FAILED

    # --- Invalid transitions ---

    def test_pending_to_completed_invalid(self) -> None:
        s = _make_state(OrderState.PENDING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.COMPLETED)

    def test_pending_to_chasing_invalid(self) -> None:
        s = _make_state(OrderState.PENDING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.CHASING)

    def test_active_to_pending_invalid(self) -> None:
        s = _make_state(OrderState.ACTIVE)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.PENDING)

    def test_completed_to_active_invalid(self) -> None:
        s = _make_state(OrderState.COMPLETED)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.ACTIVE)

    def test_failed_to_active_invalid(self) -> None:
        s = _make_state(OrderState.FAILED)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.ACTIVE)

    def test_chasing_to_market_fallback(self) -> None:
        """CHASING -> MARKET_FALLBACK is valid (rejected during chase)."""
        s = _make_state(OrderState.CHASING)
        s.transition_to(OrderState.MARKET_FALLBACK)
        assert s.state == OrderState.MARKET_FALLBACK

    def test_chasing_to_failed(self) -> None:
        """CHASING -> FAILED is valid (unrecoverable error during chase)."""
        s = _make_state(OrderState.CHASING)
        s.transition_to(OrderState.FAILED)
        assert s.state == OrderState.FAILED

    def test_chasing_to_pending_invalid(self) -> None:
        s = _make_state(OrderState.CHASING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            s.transition_to(OrderState.PENDING)

    # --- Full lifecycle paths ---

    def test_full_lifecycle_normal(self) -> None:
        """PENDING -> ACTIVE -> COMPLETED (normal fill)."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_with_chase(self) -> None:
        """PENDING -> ACTIVE -> CHASING -> ACTIVE -> COMPLETED."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.CHASING)
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_multiple_chases(self) -> None:
        """PENDING -> ACTIVE -> (CHASING -> ACTIVE) x 3 -> MARKET_FALLBACK -> COMPLETED."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        for _ in range(3):
            s.transition_to(OrderState.CHASING)
            s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.MARKET_FALLBACK)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_chase_then_fill(self) -> None:
        """PENDING -> ACTIVE -> CHASING -> COMPLETED (fill during cancel)."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.CHASING)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_reject_to_market(self) -> None:
        """PENDING -> ACTIVE -> MARKET_FALLBACK -> COMPLETED."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.MARKET_FALLBACK)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_pending_fail(self) -> None:
        """PENDING -> FAILED (instrument not found)."""
        s = _make_state()
        s.transition_to(OrderState.FAILED)
        assert s.is_terminal

    def test_full_lifecycle_chase_rejected_to_market(self) -> None:
        """PENDING -> ACTIVE -> CHASING -> MARKET_FALLBACK -> COMPLETED."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.CHASING)
        s.transition_to(OrderState.MARKET_FALLBACK)
        s.transition_to(OrderState.COMPLETED)
        assert s.is_terminal

    def test_full_lifecycle_chase_failed(self) -> None:
        """PENDING -> ACTIVE -> CHASING -> FAILED."""
        s = _make_state()
        s.transition_to(OrderState.ACTIVE)
        s.transition_to(OrderState.CHASING)
        s.transition_to(OrderState.FAILED)
        assert s.is_terminal


class TestOrderExecutionStateProperties:
    """Test OrderExecutionState dataclass properties."""

    def test_is_terminal_false_for_active_states(self) -> None:
        for state in (OrderState.PENDING, OrderState.ACTIVE, OrderState.CHASING, OrderState.MARKET_FALLBACK):
            s = _make_state(state)
            assert not s.is_terminal

    def test_is_terminal_true_for_terminal_states(self) -> None:
        for state in (OrderState.COMPLETED, OrderState.FAILED):
            s = _make_state(state)
            assert s.is_terminal

    def test_default_values(self) -> None:
        s = _make_state()
        assert s.state == OrderState.PENDING
        assert s.chase_count == 0
        assert s.current_limit_order_id is None
        assert s.timer_name == ""
        assert s.limit_orders_submitted == 0
        assert s.last_limit_price == 0.0
        assert s.fill_cost == 0.0
        assert s.timeout_secs is None
        assert s.max_chase_attempts is None
        assert s.chase_step_ticks is None
        assert s.post_only is None

    def test_completed_ns_default(self) -> None:
        s = _make_state()
        assert s.completed_ns == 0

    def test_used_market_fallback_default(self) -> None:
        s = _make_state()
        assert s.used_market_fallback is False

    def test_filled_quantity_auto_initialized(self) -> None:
        """filled_quantity is auto-initialized to zero if not provided."""
        s = OrderExecutionState(
            primary_order_id=ClientOrderId("O-AUTO"),
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            side=OrderSide.BUY,
            total_quantity=Quantity.from_str("1.0"),
            anchor_px=50000.0,
        )
        assert s.filled_quantity is not None
        assert s.filled_quantity == Quantity.zero(1)
