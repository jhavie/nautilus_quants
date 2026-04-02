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

    def test_market_accepted_arms_timeout_with_market_timeout_secs(self) -> None:
        from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig

        algo = PostLimitExecAlgorithm(PostLimitExecAlgorithmConfig(market_timeout_secs=1500.0))
        stack, _, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.PENDING_MARKET, kind=SpawnKind.MARKET)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_accepted(SimpleNamespace(client_order_id=state.active_order_id))

            assert state.state == OrderState.WORKING_MARKET
            algo._arm_timeout.assert_called_once_with(state, timeout_override=1500.0)

    def test_market_timeout_not_armed_when_disabled(self) -> None:
        from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig

        algo = PostLimitExecAlgorithm(PostLimitExecAlgorithmConfig(market_timeout_secs=0))
        stack, _, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.PENDING_MARKET, kind=SpawnKind.MARKET)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_accepted(SimpleNamespace(client_order_id=state.active_order_id))

            assert state.state == OrderState.WORKING_MARKET
            algo._arm_timeout.assert_not_called()

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

    def test_sweep_partial_fill_keeps_active_child(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.WORKING_MARKET, kind=SpawnKind.SWEEP)
            state.residual_sweep_pending = True
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            cache.order.return_value = SimpleNamespace(
                leaves_qty=Quantity.from_str("0.25"),
                status=SimpleNamespace(name="PARTIALLY_FILLED"),
            )

            algo.on_order_filled(
                SimpleNamespace(
                    client_order_id=state.active_order_id,
                    last_qty=Quantity.from_str("0.75"),
                    last_px=100.0,
                )
            )

            assert state.active_order_id == ClientOrderId("primary001E1")
            assert state.residual_sweep_pending is True
            assert state.sweep_retry_count == 0

    def test_sweep_full_fill_clears_active_child(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.WORKING_MARKET, kind=SpawnKind.SWEEP)
            state.residual_sweep_pending = True
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            cache.order.return_value = SimpleNamespace(
                leaves_qty=Quantity.from_str("0.00"),
                status=SimpleNamespace(name="FILLED"),
            )

            algo.on_order_filled(
                SimpleNamespace(
                    client_order_id=state.active_order_id,
                    last_qty=Quantity.from_str("1.0"),
                    last_px=100.0,
                )
            )

            assert state.active_order_id is None
            assert state.residual_sweep_pending is False
            assert state.sweep_retry_count == 0

    def test_sweep_rejected_retries_with_leaves_quantity(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.WORKING_MARKET, kind=SpawnKind.SWEEP)
            state.residual_sweep_pending = True
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._submit_market_child = MagicMock()  # type: ignore[method-assign]
            cache.order.return_value = SimpleNamespace(
                leaves_qty=Quantity.from_str("0.30"),
            )

            algo.on_order_rejected(
                SimpleNamespace(
                    client_order_id=ClientOrderId("primary001E1"),
                    reason="exchange_reject",
                    due_post_only=False,
                )
            )

            assert state.sweep_retry_count == 1
            algo._submit_market_child.assert_called_once_with(
                state,
                kind=SpawnKind.SWEEP,
                quantity=Quantity.from_str("0.30"),
            )
            assert state.residual_sweep_pending is True

    def test_sweep_rejected_exhausted_retries_stops_retrying(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.WORKING_MARKET, kind=SpawnKind.SWEEP)
            state.residual_sweep_pending = True
            state.sweep_retry_count = algo._config.max_sweep_retries
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = state.primary_order_id
            algo._submit_market_child = MagicMock()  # type: ignore[method-assign]
            cache.order.return_value = SimpleNamespace(
                leaves_qty=Quantity.from_str("0.30"),
            )

            algo.on_order_rejected(
                SimpleNamespace(
                    client_order_id=ClientOrderId("primary001E1"),
                    reason="exchange_reject",
                    due_post_only=False,
                )
            )

            algo._submit_market_child.assert_not_called()
            assert state.active_order_id is None
            assert state.residual_sweep_pending is False
            assert state.sweep_retry_count == 0


class TestStaleChildEventGuard:
    """Tests for stale child event guard — events from replaced children must not
    corrupt the session of the currently active child."""

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _setup_stale_scenario(
        algo: PostLimitExecAlgorithm,
        cache: MagicMock,
        *,
        session_state: OrderState = OrderState.WORKING_LIMIT,
        active_kind: SpawnKind = SpawnKind.LIMIT,
    ) -> OrderExecutionState:
        """Set up state with E2 active; E1 is the stale (replaced) child."""
        state = OrderExecutionState(
            primary_order_id=ClientOrderId("primary001"),
            instrument_id=InstrumentId.from_str("LINK-USDT-SWAP.OKX"),
            side=OrderSide.BUY,
            total_quantity=Quantity.from_str("223.21"),
            anchor_px=8.965,
            state=session_state,
            timer_name="PostLimit-primary001",
            created_ns=100,
        )
        # E2 is the currently active child
        state.activate_order(
            client_order_id=ClientOrderId("primary001E2"),
            kind=active_kind,
            reserved_quantity=Quantity.from_str("223.11"),
        )
        algo._states[state.primary_order_id] = state
        algo._active_child_to_primary[ClientOrderId("primary001E2")] = (
            state.primary_order_id
        )
        # E1 is the stale child — resolvable via exec_spawn_id fallback
        stale_order = SimpleNamespace(
            exec_spawn_id=state.primary_order_id,
            leaves_qty=Quantity.from_str("208.36"),
        )
        active_order = SimpleNamespace(
            is_closed=False,
            leaves_qty=Quantity.from_str("223.11"),
        )

        def order_lookup(order_id):
            if order_id == ClientOrderId("primary001E1"):
                return stale_order
            if order_id == ClientOrderId("primary001E2"):
                return active_order
            if order_id == state.primary_order_id:
                return SimpleNamespace(
                    client_order_id=state.primary_order_id,
                    leaves_qty=Quantity.from_str("0.10"),
                    is_closed=False,
                )
            return None

        cache.order.side_effect = order_lookup
        return state

    # ── stale terminal events (cancel / expire) ─────────────────────────

    def test_stale_canceled_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, log = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(algo, cache)
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]

            algo.on_order_canceled(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.active_order_id == ClientOrderId("primary001E2")
            assert state.state == OrderState.WORKING_LIMIT
            algo._restore_primary_from_active.assert_not_called()
            log.warning.assert_called()

    def test_stale_expired_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, log = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(algo, cache)
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]

            algo.on_order_expired(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.active_order_id == ClientOrderId("primary001E2")
            assert state.state == OrderState.WORKING_LIMIT
            algo._restore_primary_from_active.assert_not_called()

    # ── stale accepted ──────────────────────────────────────────────────

    def test_stale_accepted_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(
                algo, cache, session_state=OrderState.PENDING_LIMIT
            )
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_accepted(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.active_order_id == ClientOrderId("primary001E2")
            assert state.state == OrderState.PENDING_LIMIT
            algo._arm_timeout.assert_not_called()

    # ── stale rejected / denied ─────────────────────────────────────────

    def test_stale_rejected_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(
                algo, cache, session_state=OrderState.PENDING_LIMIT
            )
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]
            algo._execute_session_command = MagicMock()  # type: ignore[method-assign]

            algo.on_order_rejected(
                SimpleNamespace(
                    client_order_id=ClientOrderId("primary001E1"),
                    reason="post_only",
                    due_post_only=True,
                )
            )

            assert state.active_order_id == ClientOrderId("primary001E2")
            algo._restore_primary_from_active.assert_not_called()
            algo._execute_session_command.assert_not_called()

    def test_stale_denied_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(
                algo, cache, session_state=OrderState.PENDING_LIMIT
            )
            algo._restore_primary_from_active = MagicMock()  # type: ignore[method-assign]
            algo._execute_session_command = MagicMock()  # type: ignore[method-assign]

            algo.on_order_denied(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.active_order_id == ClientOrderId("primary001E2")
            algo._restore_primary_from_active.assert_not_called()
            algo._execute_session_command.assert_not_called()

    # ── stale cancel_rejected ───────────────────────────────────────────

    def test_stale_cancel_rejected_event_ignored(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(
                algo, cache, session_state=OrderState.CANCEL_PENDING_REPRICE
            )
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            algo.on_order_cancel_rejected(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.state == OrderState.CANCEL_PENDING_REPRICE
            algo._arm_timeout.assert_not_called()

    # ── stale fill — track + cancel active child ────────────────────────

    def test_stale_fill_tracked_and_active_child_canceled(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, log = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(algo, cache)
            algo.cancel_order = MagicMock()  # type: ignore[method-assign]

            with patch(
                "nautilus_quants.execution.post_limit.algorithm.compute_remaining_quantity",
                return_value=Quantity.from_str("208.36"),
            ):
                algo.on_order_filled(
                    SimpleNamespace(
                        client_order_id=ClientOrderId("primary001E1"),
                        last_qty=Quantity.from_str("14.75"),
                        last_px=8.967,
                    )
                )

            # Fill tracked in state
            assert float(state.filled_quantity) == 14.75
            assert state.fill_cost > 0
            # Active child E2 canceled for resubmit
            algo.cancel_order.assert_called_once()
            log.warning.assert_called()

    def test_stale_fill_completes_session_when_remaining_zero(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = self._setup_stale_scenario(algo, cache)
            algo.cancel_order = MagicMock()  # type: ignore[method-assign]
            algo._complete_session = MagicMock()  # type: ignore[method-assign]

            with patch(
                "nautilus_quants.execution.post_limit.algorithm.compute_remaining_quantity",
                return_value=Quantity.zero(2),
            ):
                algo.on_order_filled(
                    SimpleNamespace(
                        client_order_id=ClientOrderId("primary001E1"),
                        last_qty=Quantity.from_str("223.21"),
                        last_px=8.967,
                    )
                )

            algo.cancel_order.assert_called_once()
            algo._complete_session.assert_called_once_with(state)

    # ── regression: active child events still work normally ─────────────

    def test_active_child_events_still_processed_normally(self) -> None:
        algo = PostLimitExecAlgorithm()
        stack, cache, _, _ = _patch_algo_environment(algo)
        with stack:
            state = _make_state(state=OrderState.PENDING_LIMIT)
            algo._states[state.primary_order_id] = state
            algo._active_child_to_primary[state.active_order_id] = (
                state.primary_order_id
            )
            algo._arm_timeout = MagicMock()  # type: ignore[method-assign]

            # Event from E1 which IS the active child → should process normally
            algo.on_order_accepted(
                SimpleNamespace(client_order_id=ClientOrderId("primary001E1"))
            )

            assert state.state == OrderState.WORKING_LIMIT
            algo._arm_timeout.assert_called_once_with(state)
