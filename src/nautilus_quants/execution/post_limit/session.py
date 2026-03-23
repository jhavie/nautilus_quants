# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Session state machine for PostLimitExecAlgorithm."""

from __future__ import annotations

from enum import Enum

from nautilus_quants.execution.post_limit.state import OrderExecutionState, OrderState


class SessionCommand(Enum):
    """Commands emitted by the session state machine."""

    NOOP = "NOOP"
    SUBMIT_LIMIT = "SUBMIT_LIMIT"
    SUBMIT_MARKET = "SUBMIT_MARKET"
    CANCEL_ACTIVE = "CANCEL_ACTIVE"
    REARM_TIMEOUT = "REARM_TIMEOUT"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


class PostLimitSession:
    """Behavior wrapper around a single OrderExecutionState."""

    def __init__(self, state: OrderExecutionState) -> None:
        self.state = state

    def on_limit_submitted(self) -> None:
        self.state.state = OrderState.PENDING_LIMIT
        self.state.active_order_accepted = False

    def on_market_submitted(self) -> None:
        self.state.state = OrderState.PENDING_MARKET
        self.state.active_order_accepted = False
        self.state.used_market_fallback = True

    def on_limit_accepted(self) -> SessionCommand:
        if self.state.state != OrderState.PENDING_LIMIT:
            return SessionCommand.NOOP
        self.state.transition_to(OrderState.WORKING_LIMIT)
        self.state.active_order_accepted = True
        return SessionCommand.REARM_TIMEOUT

    def on_market_accepted(self) -> SessionCommand:
        if self.state.state != OrderState.PENDING_MARKET:
            return SessionCommand.NOOP
        self.state.transition_to(OrderState.WORKING_MARKET)
        self.state.active_order_accepted = True
        return SessionCommand.NOOP

    def on_timeout(self, *, max_chase_attempts: int, fallback_to_market: bool) -> SessionCommand:
        if self.state.state != OrderState.WORKING_LIMIT:
            return SessionCommand.NOOP

        if self.state.chase_count < max_chase_attempts:
            self.state.chase_count += 1
            self.state.post_only_retreat_ticks = 0
            self.state.transition_to(OrderState.CANCEL_PENDING_REPRICE)
            return SessionCommand.CANCEL_ACTIVE

        if fallback_to_market:
            self.state.transition_to(OrderState.CANCEL_PENDING_MARKET)
            self.state.used_market_fallback = True
            return SessionCommand.CANCEL_ACTIVE

        return SessionCommand.FAIL

    def on_rejected(
        self,
        *,
        due_post_only: bool,
        max_post_only_retries: int,
        fallback_to_market: bool,
    ) -> SessionCommand:
        if due_post_only and self.state.state in {
            OrderState.PENDING_LIMIT,
            OrderState.WORKING_LIMIT,
            OrderState.CANCEL_PENDING_REPRICE,
        }:
            if self.state.post_only_retreat_ticks < max_post_only_retries:
                self.state.post_only_retreat_ticks += 1
                self.state.state = OrderState.PENDING_LIMIT
                return SessionCommand.SUBMIT_LIMIT

        if self.state.state in {OrderState.PENDING_MARKET, OrderState.WORKING_MARKET}:
            return SessionCommand.FAIL

        if fallback_to_market:
            self.state.state = OrderState.PENDING_MARKET
            self.state.used_market_fallback = True
            return SessionCommand.SUBMIT_MARKET

        return SessionCommand.FAIL

    def on_denied(self, *, fallback_to_market: bool) -> SessionCommand:
        return self.on_rejected(
            due_post_only=False,
            max_post_only_retries=0,
            fallback_to_market=fallback_to_market,
        )

    def on_cancel_confirmed(self, *, fallback_to_market: bool) -> SessionCommand:
        if self.state.state == OrderState.CANCEL_PENDING_REPRICE:
            self.state.state = OrderState.PENDING_LIMIT
            return SessionCommand.SUBMIT_LIMIT

        if self.state.state == OrderState.CANCEL_PENDING_MARKET:
            self.state.state = OrderState.PENDING_MARKET
            return SessionCommand.SUBMIT_MARKET

        if self.state.state == OrderState.WORKING_LIMIT and fallback_to_market:
            self.state.state = OrderState.PENDING_MARKET
            self.state.used_market_fallback = True
            return SessionCommand.SUBMIT_MARKET

        return SessionCommand.NOOP

    def on_cancel_rejected(self) -> SessionCommand:
        if self.state.state in {
            OrderState.CANCEL_PENDING_REPRICE,
            OrderState.CANCEL_PENDING_MARKET,
        }:
            self.state.state = OrderState.WORKING_LIMIT
            return SessionCommand.REARM_TIMEOUT
        return SessionCommand.NOOP

    def mark_complete(self) -> SessionCommand:
        if not self.state.is_terminal:
            self.state.transition_to(OrderState.COMPLETED)
        return SessionCommand.COMPLETE

    def mark_failed(self) -> SessionCommand:
        if not self.state.is_terminal:
            self.state.transition_to(OrderState.FAILED)
        return SessionCommand.FAIL
