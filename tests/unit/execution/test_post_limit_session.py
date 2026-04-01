# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for the PostLimit session state machine."""

from __future__ import annotations

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.session import PostLimitSession, SessionCommand
from nautilus_quants.execution.post_limit.state import OrderExecutionState, OrderState


def _make_state(state: OrderState = OrderState.PENDING_LIMIT) -> OrderExecutionState:
    return OrderExecutionState(
        primary_order_id=ClientOrderId("primary001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.0"),
        anchor_px=50000.0,
        state=state,
        timer_name="PostLimit-primary001",
        created_ns=123,
    )


class TestPostLimitSession:
    def test_limit_accepted_moves_to_working_limit_and_rearms_timeout(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        command = PostLimitSession(state).on_limit_accepted()

        assert state.state == OrderState.WORKING_LIMIT
        assert state.active_order_accepted is True
        assert command == SessionCommand.REARM_TIMEOUT

    def test_timeout_enters_cancel_pending_reprice_while_chase_budget_remaining(self) -> None:
        state = _make_state(OrderState.WORKING_LIMIT)
        state.post_only_retreat_ticks = 2

        command = PostLimitSession(state).on_timeout(
            max_chase_attempts=3,
            fallback_to_market=True,
        )

        assert state.state == OrderState.CANCEL_PENDING_REPRICE
        assert state.chase_count == 1
        assert state.post_only_retreat_ticks == 0
        assert command == SessionCommand.CANCEL_ACTIVE

    def test_timeout_enters_cancel_pending_market_when_chase_budget_exhausted(self) -> None:
        state = _make_state(OrderState.WORKING_LIMIT)
        state.chase_count = 2

        command = PostLimitSession(state).on_timeout(
            max_chase_attempts=2,
            fallback_to_market=True,
        )

        assert state.state == OrderState.CANCEL_PENDING_MARKET
        assert state.used_market_fallback is True
        assert command == SessionCommand.CANCEL_ACTIVE

    def test_timeout_fails_when_no_fallback_allowed(self) -> None:
        state = _make_state(OrderState.WORKING_LIMIT)
        state.chase_count = 1

        command = PostLimitSession(state).on_timeout(
            max_chase_attempts=1,
            fallback_to_market=False,
        )

        assert command == SessionCommand.FAIL

    def test_post_only_reject_retries_as_limit_until_retry_budget_exhausted(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        command = PostLimitSession(state).on_rejected(
            due_post_only=True,
            max_post_only_retries=2,
            fallback_to_market=True,
        )

        assert state.state == OrderState.PENDING_LIMIT
        assert state.post_only_retreat_ticks == 1
        assert command == SessionCommand.SUBMIT_LIMIT

    def test_post_only_reject_exhaustion_falls_back_to_market(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)
        state.post_only_retreat_ticks = 2

        command = PostLimitSession(state).on_rejected(
            due_post_only=True,
            max_post_only_retries=2,
            fallback_to_market=True,
        )

        assert state.state == OrderState.PENDING_MARKET
        assert state.used_market_fallback is True
        assert command == SessionCommand.SUBMIT_MARKET

    def test_cancel_confirmed_reprice_submits_new_limit(self) -> None:
        state = _make_state(OrderState.CANCEL_PENDING_REPRICE)

        command = PostLimitSession(state).on_cancel_confirmed(
            max_chase_attempts=3, fallback_to_market=True,
        )

        assert state.state == OrderState.PENDING_LIMIT
        assert command == SessionCommand.SUBMIT_LIMIT

    def test_exchange_cancel_retries_limit_before_market_fallback(self) -> None:
        """Exchange-initiated cancel on WORKING_LIMIT should chase, not fallback."""
        state = _make_state(OrderState.WORKING_LIMIT)
        state.chase_count = 0

        command = PostLimitSession(state).on_cancel_confirmed(
            max_chase_attempts=3, fallback_to_market=True,
        )

        assert state.state == OrderState.PENDING_LIMIT
        assert state.chase_count == 1
        assert command == SessionCommand.SUBMIT_LIMIT

    def test_exchange_cancel_falls_back_when_chases_exhausted(self) -> None:
        """Exchange cancel after max chases should fallback to market."""
        state = _make_state(OrderState.WORKING_LIMIT)
        state.chase_count = 3

        command = PostLimitSession(state).on_cancel_confirmed(
            max_chase_attempts=3, fallback_to_market=True,
        )

        assert state.state == OrderState.PENDING_MARKET
        assert state.used_market_fallback is True
        assert command == SessionCommand.SUBMIT_MARKET

    def test_exchange_cancel_fails_when_no_fallback(self) -> None:
        """Exchange cancel with no market fallback and chases exhausted should fail."""
        state = _make_state(OrderState.WORKING_LIMIT)
        state.chase_count = 3

        command = PostLimitSession(state).on_cancel_confirmed(
            max_chase_attempts=3, fallback_to_market=False,
        )

        assert command == SessionCommand.FAIL

    def test_cancel_rejected_returns_to_working_limit(self) -> None:
        state = _make_state(OrderState.CANCEL_PENDING_MARKET)

        command = PostLimitSession(state).on_cancel_rejected()

        assert state.state == OrderState.WORKING_LIMIT
        assert command == SessionCommand.REARM_TIMEOUT

    def test_transient_reject_schedules_retry_when_budget_remaining(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        command = PostLimitSession(state).on_rejected(
            due_post_only=False,
            max_post_only_retries=0,
            fallback_to_market=True,
            reason="Service temporarily unavailable. Please try again later.",
            max_transient_retries=3,
        )

        assert state.state == OrderState.RETRY_PENDING
        assert state.transient_retry_count == 1
        assert command == SessionCommand.SCHEDULE_RETRY

    def test_transient_reject_falls_back_when_budget_exhausted(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)
        state.transient_retry_count = 3

        command = PostLimitSession(state).on_rejected(
            due_post_only=False,
            max_post_only_retries=0,
            fallback_to_market=True,
            reason="Service temporarily unavailable. Please try again later.",
            max_transient_retries=3,
        )

        assert state.state == OrderState.PENDING_MARKET
        assert state.used_market_fallback is True
        assert command == SessionCommand.SUBMIT_MARKET

    def test_non_transient_reject_bypasses_retry(self) -> None:
        state = _make_state(OrderState.PENDING_LIMIT)

        command = PostLimitSession(state).on_rejected(
            due_post_only=False,
            max_post_only_retries=0,
            fallback_to_market=True,
            reason="Insufficient balance",
            max_transient_retries=3,
        )

        assert state.state == OrderState.PENDING_MARKET
        assert state.transient_retry_count == 0
        assert command == SessionCommand.SUBMIT_MARKET
