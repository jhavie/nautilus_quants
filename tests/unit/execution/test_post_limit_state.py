# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimit runtime state and snapshot encoding."""

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
    SpawnKind,
    decode_execution_states,
    encode_execution_states,
)


def _make_state(state: OrderState = OrderState.PENDING_LIMIT) -> OrderExecutionState:
    return OrderExecutionState(
        primary_order_id=ClientOrderId("primary001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.50"),
        anchor_px=50000.0,
        state=state,
        timer_name="PostLimit-primary001",
        created_ns=123,
    )


class TestOrderState:
    def test_all_states_defined(self) -> None:
        assert {state.value for state in OrderState} == {
            "PENDING_LIMIT",
            "WORKING_LIMIT",
            "CANCEL_PENDING_REPRICE",
            "CANCEL_PENDING_MARKET",
            "PENDING_MARKET",
            "WORKING_MARKET",
            "RETRY_PENDING",
            "COMPLETED",
            "FAILED",
        }

    def test_terminal_states(self) -> None:
        assert TERMINAL_STATES == {OrderState.COMPLETED, OrderState.FAILED}

    def test_every_state_has_a_transition_entry(self) -> None:
        for state in OrderState:
            assert state in VALID_TRANSITIONS


class TestOrderExecutionState:
    def test_activate_and_clear_active_order(self) -> None:
        state = _make_state()

        state.activate_order(
            client_order_id=ClientOrderId("primary001E1"),
            kind=SpawnKind.LIMIT,
            reserved_quantity=Quantity.from_str("1.50"),
        )

        assert state.active_order_id == ClientOrderId("primary001E1")
        assert state.active_order_kind == SpawnKind.LIMIT
        assert state.active_reserved_quantity == Quantity.from_str("1.50")
        assert state.active_order_accepted is False

        state.clear_active_order()

        assert state.active_order_id is None
        assert state.active_order_kind is None
        assert state.active_reserved_quantity is None
        assert state.active_order_accepted is False

    def test_valid_transition(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        state.transition_to(OrderState.WORKING_LIMIT)

        assert state.state == OrderState.WORKING_LIMIT

    def test_invalid_transition_raises(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        with pytest.raises(ValueError, match="Invalid state transition"):
            state.transition_to(OrderState.WORKING_MARKET)

    def test_is_terminal_and_is_working_properties(self) -> None:
        assert _make_state(OrderState.WORKING_LIMIT).is_working is True
        assert _make_state(OrderState.WORKING_MARKET).is_working is True
        assert _make_state(OrderState.COMPLETED).is_terminal is True
        assert _make_state(OrderState.FAILED).is_terminal is True


class TestStateSnapshots:
    def test_encode_decode_round_trip_preserves_runtime_fields(self) -> None:
        state = _make_state(OrderState.WORKING_LIMIT)
        state.activate_order(
            client_order_id=ClientOrderId("primary001E2"),
            kind=SpawnKind.LIMIT,
            reserved_quantity=Quantity.from_str("0.75"),
            accepted=True,
        )
        state.chase_count = 2
        state.spawn_sequence = 2
        state.post_only_retreat_ticks = 1
        state.filled_quantity = Quantity.from_str("0.75")
        state.filled_quote_quantity = 37500.0
        state.contract_multiplier = 1.0
        state.intent = "OPEN"
        state.limit_orders_submitted = 2
        state.last_limit_price = 49999.5
        state.fill_cost = 37499.625
        state.residual_sweep_pending = True
        state.sweep_retry_count = 2

        encoded = encode_execution_states({state.primary_order_id: state})
        decoded = decode_execution_states(encoded)

        restored = decoded[state.primary_order_id]
        assert restored.state == OrderState.WORKING_LIMIT
        assert restored.active_order_id == ClientOrderId("primary001E2")
        assert restored.active_order_kind == SpawnKind.LIMIT
        assert restored.active_reserved_quantity == Quantity.from_str("0.75")
        assert restored.active_order_accepted is True
        assert restored.spawn_sequence == 2
        assert restored.chase_count == 2
        assert restored.post_only_retreat_ticks == 1
        assert restored.filled_quantity == Quantity.from_str("0.75")
        assert restored.filled_quote_quantity == 37500.0
        assert restored.limit_orders_submitted == 2
        assert restored.last_limit_price == 49999.5
        assert restored.fill_cost == 37499.625
        assert restored.residual_sweep_pending is True
        assert restored.sweep_retry_count == 2
        assert restored.transient_retry_count == 0

    def test_decode_empty_store(self) -> None:
        encoded = encode_execution_states({})

        assert decode_execution_states(encoded) == {}
