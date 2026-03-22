# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for PostLimitExecAlgorithm hook routing behavior."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.algorithm import PostLimitExecAlgorithm
from nautilus_quants.execution.post_limit.session import SessionCommand
from nautilus_quants.execution.post_limit.state import (
    OrderExecutionState,
    OrderState,
    SpawnKind,
)


class _FakeClock:
    def __init__(self) -> None:
        self._timer_names: set[str] = set()
        self.set_timer = MagicMock(side_effect=self._set_timer)
        self.cancel_timer = MagicMock(side_effect=self._cancel_timer)
        self.cancel_timers = MagicMock(side_effect=self._cancel_all)
        self._ts = 1_000

    @property
    def timer_names(self) -> set[str]:
        return self._timer_names

    def timestamp_ns(self) -> int:
        self._ts += 1
        return self._ts

    def _set_timer(self, *, name, interval, callback) -> None:
        self._timer_names.add(name)

    def _cancel_timer(self, name) -> None:
        self._timer_names.discard(name)

    def _cancel_all(self) -> None:
        self._timer_names.clear()


def _make_state(
    *,
    state: OrderState,
    active_child: str = "primary001E1",
    kind: SpawnKind = SpawnKind.LIMIT,
) -> OrderExecutionState:
    order_state = OrderExecutionState(
        primary_order_id=ClientOrderId("primary001"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        side=OrderSide.BUY,
        total_quantity=Quantity.from_str("1.0"),
        anchor_px=50000.0,
        state=state,
        timer_name="PostLimit-primary001",
        created_ns=123,
    )
    order_state.activate_order(
        client_order_id=ClientOrderId(active_child),
        kind=kind,
        reserved_quantity=Quantity.from_str("1.0"),
    )
    return order_state


def _patch_algo_environment(algo: PostLimitExecAlgorithm):
    stack = ExitStack()
    clock = _FakeClock()
    cache = MagicMock()
    cache.add = MagicMock()
    log = MagicMock()
    stack.enter_context(
        patch.object(
            PostLimitExecAlgorithm, "clock", new_callable=PropertyMock, return_value=clock
        ),
    )
    stack.enter_context(
        patch.object(
            PostLimitExecAlgorithm, "cache", new_callable=PropertyMock, return_value=cache
        ),
    )
    stack.enter_context(
        patch.object(PostLimitExecAlgorithm, "log", new_callable=PropertyMock, return_value=log),
    )
    return stack, cache, clock, log


class TestPostLimitExecAlgorithmHooks:
    def test_on_order_fails_fast_for_non_market_orders(self) -> None:
        algo = PostLimitExecAlgorithm()
        with _patch_algo_environment(algo)[0]:
            algo._record_failed_primary = MagicMock()  # type: ignore[method-assign]

            order = SimpleNamespace(
                client_order_id=ClientOrderId("primary001"),
                instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                quantity=Quantity.from_str("1.0"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                is_reduce_only=False,
                exec_algorithm_params={"anchor_px": "50000"},
            )

            algo.on_order(order)

            algo._record_failed_primary.assert_called_once_with(order, anchor_px=0.0)

    def test_on_order_accepted_arms_timeout_only_after_limit_accept(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, _, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.PENDING_LIMIT)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_accepted(SimpleNamespace(client_order_id=state.active_order_id))

            assert state.state == OrderState.WORKING_LIMIT
            algo._arm_timeout.assert_called_once_with(state)

    def test_on_time_event_uses_timer_index_without_state_scan(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.WORKING_LIMIT)
            algo._states = {
                state.primary_order_id: state,
                ClientOrderId("other"): _make_state(
                    state=OrderState.WORKING_LIMIT,
                    active_child="otherE1",
                ),
            }
            algo._timer_to_primary[state.timer_name] = state.primary_order_id
            algo.cancel_order = MagicMock()  # type: ignore[method-assign]
            algo._lookup_state_by_child = MagicMock(side_effect=AssertionError("should not scan"))  # type: ignore[method-assign]
            cache.order.return_value = SimpleNamespace(is_closed=False)

            algo.on_time_event(SimpleNamespace(name=state.timer_name))

            assert state.state == OrderState.CANCEL_PENDING_REPRICE
            algo.cancel_order.assert_called_once()

    def test_on_order_denied_transitions_to_market_submission_path(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, _, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.PENDING_LIMIT)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._cancel_timer = MagicMock()  # type: ignore[method-assign]
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]
            algo._clear_active_child = MagicMock()  # type: ignore[method-assign]
            algo._execute_session_command = MagicMock()  # type: ignore[method-assign]

            algo.on_order_denied(SimpleNamespace(client_order_id=ClientOrderId("primary001E1")))

            assert state.state == OrderState.PENDING_MARKET
            algo._execute_session_command.assert_called_once_with(
                state,
                SessionCommand.SUBMIT_MARKET,
            )

    def test_on_order_cancel_rejected_returns_to_working_limit(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, _, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.CANCEL_PENDING_REPRICE)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_cancel_rejected(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1")),
            )

            assert state.state == OrderState.WORKING_LIMIT
            algo._arm_timeout.assert_called_once_with(state)

    def test_on_order_canceled_reprice_submits_new_limit(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.CANCEL_PENDING_REPRICE)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]
            algo.submit_order = MagicMock()  # type: ignore[method-assign]

            primary = SimpleNamespace(
                client_order_id=state.primary_order_id,
                leaves_qty=Quantity.from_str("12.32"),
                is_closed=False,
            )
            cache.order.side_effect = lambda order_id: (
                primary if order_id == state.primary_order_id else None
            )
            cache.instrument.return_value = SimpleNamespace(
                price_increment=0.01,
                make_price=lambda value: value,
            )
            cache.quote_tick.return_value = SimpleNamespace(bid_price=100.0, ask_price=100.1)

            spawned = SimpleNamespace(
                client_order_id=ClientOrderId("primary001E2"),
                quantity=Quantity.from_str("12.32"),
            )
            with (
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.compute_remaining_quantity",
                    return_value=Quantity.from_str("12.32"),
                ),
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.ChildOrderFactory.create_limit",
                    return_value=spawned,
                ),
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.PrimaryMirror.reduce_primary",
                    return_value=Quantity.from_str("12.32"),
                ),
            ):
                algo.on_order_canceled(
                    SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
                )

            algo.submit_order.assert_called_once_with(spawned)
            assert state.state == OrderState.PENDING_LIMIT
            assert state.active_order_id == ClientOrderId("primary001E2")
            assert (
                algo._active_child_to_primary[ClientOrderId("primary001E2")]
                == state.primary_order_id
            )

    def test_on_order_canceled_mirror_reduce_error_fails_session_without_raise(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.CANCEL_PENDING_REPRICE)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]
            algo.submit_order = MagicMock()  # type: ignore[method-assign]

            primary = SimpleNamespace(
                client_order_id=state.primary_order_id,
                leaves_qty=Quantity.from_str("1.00"),
                is_closed=False,
            )
            cache.order.side_effect = lambda order_id: (
                primary if order_id == state.primary_order_id else None
            )
            cache.instrument.return_value = SimpleNamespace(
                price_increment=0.01,
                make_price=lambda value: value,
            )
            cache.quote_tick.return_value = SimpleNamespace(bid_price=100.0, ask_price=100.1)

            spawned = SimpleNamespace(
                client_order_id=ClientOrderId("primary001E2"),
                quantity=Quantity.from_str("1.00"),
            )
            with (
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.compute_remaining_quantity",
                    return_value=Quantity.from_str("1.00"),
                ),
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.ChildOrderFactory.create_limit",
                    return_value=spawned,
                ),
                patch(
                    "nautilus_quants.execution.post_limit.algorithm.PrimaryMirror.reduce_primary",
                    side_effect=ValueError("forced failure"),
                ),
            ):
                algo.on_order_canceled(
                    SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
                )

            assert state.state == OrderState.FAILED
            algo.submit_order.assert_not_called()

    def test_on_stop_no_longer_persists_state_via_cache(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, clock, _ = _patch_algo_environment(algo)
        with stack:
            algo._states[ClientOrderId("primary001")] = _make_state(state=OrderState.WORKING_LIMIT)

            algo.on_stop()

            clock.cancel_timers.assert_called_once()
            cache.add.assert_not_called()
