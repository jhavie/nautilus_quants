# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Post-only limit execution with timeout-driven reprice and market fallback."""

from __future__ import annotations

import pickle
from datetime import timedelta
from typing import TYPE_CHECKING

from nautilus_trader.execution.algorithm import ExecAlgorithm
from nautilus_trader.model.enums import OrderType, TimeInForce
from nautilus_trader.model.events.order import (
    OrderAccepted,
    OrderCancelRejected,
    OrderCanceled,
    OrderDenied,
    OrderExpired,
    OrderFilled,
    OrderRejected,
)
from nautilus_trader.model.events.position import PositionChanged, PositionClosed
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Quantity

from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig
from nautilus_quants.execution.post_limit.pricing import (
    compute_limit_price,
    compute_remaining_quantity,
    compute_residual_notional,
    determine_limit_price,
    normalize_qty_or_zero,
)
from nautilus_quants.execution.post_limit.request import (
    PostLimitRequest,
    PostLimitRequestError,
)
from nautilus_quants.execution.post_limit.session import PostLimitSession, SessionCommand
from nautilus_quants.execution.post_limit.spawn import (
    ChildOrderFactory,
    PrimaryMirror,
    spawn_linkage_fields,
)
from nautilus_quants.execution.post_limit.state import (
    OrderExecutionState,
    OrderState,
    SpawnKind,
    decode_execution_states,
    encode_execution_states,
)
from nautilus_quants.utils.cache_keys import EXECUTION_STATES_CACHE_KEY

if TYPE_CHECKING:
    from nautilus_trader.common.events import TimeEvent
    from nautilus_trader.model.orders import Order


_normalize_qty_or_zero = normalize_qty_or_zero
_spawn_linkage_fields = spawn_linkage_fields


class PostLimitExecAlgorithm(ExecAlgorithm):
    """Execution algorithm which converts market intent into limit-first execution."""

    def __init__(self, config: PostLimitExecAlgorithmConfig | None = None) -> None:
        resolved_config = config or PostLimitExecAlgorithmConfig()
        super().__init__(resolved_config)
        self._config = resolved_config
        self._states: dict[ClientOrderId, OrderExecutionState] = {}
        self._active_child_to_primary: dict[ClientOrderId, ClientOrderId] = {}
        self._timer_to_primary: dict[str, ClientOrderId] = {}
        self._retry_timer_to_primary: dict[str, ClientOrderId] = {}
        self._subscribed_instruments: set[InstrumentId] = set()

    def on_start(self) -> None:
        self.log.info(
            "PostLimitExecAlgorithm started: "
            f"offset_ticks={self._config.offset_ticks}, "
            f"timeout_secs={self._config.timeout_secs}, "
            f"max_chase={self._config.max_chase_attempts}, "
            f"chase_step_ticks={self._config.chase_step_ticks}, "
            f"fallback_to_market={self._config.fallback_to_market}, "
            f"post_only={self._config.post_only}, "
            f"max_post_only_retries={self._config.max_post_only_retries}, "
            f"enable_residual_sweep={self._config.enable_residual_sweep}, "
            f"max_sweep_retries={self._config.max_sweep_retries}, "
            f"max_transient_retries={self._config.max_transient_retries}, "
            f"transient_retry_delay_secs={self._config.transient_retry_delay_secs}, "
            f"market_timeout_secs={self._config.market_timeout_secs}"
        )
        self._rebuild_runtime_indexes(rearm_timers=True)

    def on_stop(self) -> None:
        self.clock.cancel_timers()
        self._timer_to_primary.clear()
        self.log.info(
            "PostLimitExecAlgorithm stopped: "
            f"tracked_sequences={len(self._states)}, "
            f"active={sum(1 for state in self._states.values() if not state.is_terminal)}"
        )

    def on_reset(self) -> None:
        self.clock.cancel_timers()
        self._states.clear()
        self._active_child_to_primary.clear()
        self._timer_to_primary.clear()
        self._retry_timer_to_primary.clear()
        self._subscribed_instruments.clear()
        self._publish_execution_states()

    def on_save(self) -> dict[str, bytes]:
        return {"sessions": self._encode_states()}

    def on_load(self, state: dict[str, bytes]) -> None:
        raw = state.get("sessions") or state.get("states")
        if raw is None:
            return
        self._states = self._decode_states(raw)
        self._rebuild_runtime_indexes(rearm_timers=False)
        self._publish_execution_states()

    def on_order(self, order: Order) -> None:
        if order.order_type != OrderType.MARKET:
            self.log.error(
                f"PostLimit only supports MarketOrders: {order.client_order_id} "
                f"got={order.order_type}"
            )
            self._record_failed_primary(order, anchor_px=0.0)
            return

        try:
            request = PostLimitRequest.parse(order.exec_algorithm_params)
        except PostLimitRequestError as exc:
            self.log.error(f"PostLimit invalid exec params for {order.client_order_id}: {exc}")
            self._record_failed_primary(order, anchor_px=0.0)
            return

        instrument = self.cache.instrument(order.instrument_id)
        if instrument is None:
            self.log.error(f"PostLimit instrument not found: {order.instrument_id}")
            self._record_failed_primary(order, anchor_px=request.anchor_px)
            return

        state = OrderExecutionState(
            primary_order_id=order.client_order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            total_quantity=order.quantity,
            anchor_px=request.anchor_px,
            reduce_only=order.is_reduce_only,
            timer_name=f"PostLimit-{order.client_order_id.value}",
            created_ns=self.clock.timestamp_ns(),
            timeout_secs=request.timeout_secs,
            max_chase_attempts=request.max_chase_attempts,
            chase_step_ticks=request.chase_step_ticks,
            post_only=request.post_only,
            target_quote_quantity=request.target_quote_quantity,
            contract_multiplier=request.contract_multiplier,
        )
        self._states[state.primary_order_id] = state
        self._ensure_quote_subscription(state.instrument_id)

        self.log.info(
            f"PostLimit on_order: {order.client_order_id} "
            f"{order.side.name} {order.quantity} {order.instrument_id} "
            f"anchor_px={request.anchor_px} "
            f"target_quote_quantity={state.target_quote_quantity}"
        )
        self._submit_limit_child(state)
        self._publish_execution_states()

    def on_order_accepted(self, event: OrderAccepted) -> None:
        state = self._lookup_state_by_child(event.client_order_id)
        if state is None:
            return
        if self._is_stale_child(event.client_order_id, state):
            return
        state.active_order_accepted = True

        if state.active_order_kind == SpawnKind.SWEEP:
            return

        session = PostLimitSession(state)
        if state.active_order_kind == SpawnKind.LIMIT:
            command = session.on_limit_accepted()
            if command == SessionCommand.REARM_TIMEOUT:
                self._arm_timeout(state)
        elif state.active_order_kind == SpawnKind.MARKET:
            session.on_market_accepted()
            if self._config.market_timeout_secs > 0:
                self._arm_timeout(
                    state, timeout_override=self._config.market_timeout_secs,
                )
        self._publish_execution_states()

    def on_time_event(self, event: TimeEvent) -> None:
        # Check retry timers first.
        retry_primary_id = self._retry_timer_to_primary.pop(event.name, None)
        if retry_primary_id is not None:
            self._on_retry_timer(retry_primary_id)
            return

        primary_id = self._timer_to_primary.get(event.name)
        if primary_id is None:
            return

        state = self._states.get(primary_id)
        if state is None or state.is_terminal or state.active_order_id is None:
            return

        current_order = self.cache.order(state.active_order_id)
        if current_order is None or current_order.is_closed:
            return

        session = PostLimitSession(state)
        command = session.on_timeout(
            max_chase_attempts=self._get_max_chase(state),
            fallback_to_market=self._config.fallback_to_market,
        )
        if (
            state.active_order_kind == SpawnKind.MARKET
            and command == SessionCommand.CANCEL_ACTIVE
        ):
            self.log.warning(
                f"PostLimit MARKET TIMEOUT: {state.primary_order_id} "
                f"{state.instrument_id} abandoned after "
                f"{self._config.market_timeout_secs:.0f}s unfilled"
            )
        if command == SessionCommand.CANCEL_ACTIVE:
            self.cancel_order(current_order)
        elif command == SessionCommand.FAIL:
            self._fail_session(state)
        self._publish_execution_states()

    def on_order_filled(self, event: OrderFilled) -> None:
        state = self._lookup_state_by_child(event.client_order_id)
        if state is None:
            return
        if self._is_stale_child(event.client_order_id, state):
            self._handle_stale_fill(event, state)
            return

        if state.active_order_kind == SpawnKind.SWEEP:
            sweep_order = self.cache.order(event.client_order_id)
            if sweep_order is not None and self._is_order_fully_filled(sweep_order):
                self._clear_active_child(state)
                state.residual_sweep_pending = False
                state.sweep_retry_count = 0
            self._publish_execution_states()
            return

        fill_qty = event.last_qty
        fill_px = float(event.last_px)
        state.fill_cost += fill_px * float(fill_qty)
        state.filled_quantity = Quantity(
            state.filled_quantity + fill_qty,
            state.filled_quantity.precision,
        )
        if state.target_quote_quantity is not None:
            state.filled_quote_quantity += fill_px * float(fill_qty) * state.contract_multiplier

        remaining = compute_remaining_quantity(self.cache, state, self.log)
        if remaining <= Quantity.zero(remaining.precision):
            self._complete_session(state)
        else:
            self._publish_execution_states()

    def on_order_rejected(self, event: OrderRejected) -> None:
        state = self._lookup_state_by_child(event.client_order_id)
        if state is None:
            return
        if self._is_stale_child(event.client_order_id, state):
            self.log.warning(
                f"PostLimit ignoring stale rejected for {event.client_order_id}: "
                f"active={state.active_order_id} primary={state.primary_order_id}"
            )
            return

        if state.active_order_kind == SpawnKind.SWEEP:
            self.log.warning(f"PostLimit sweep order rejected: {event.client_order_id}")
            sweep_order = self.cache.order(event.client_order_id)
            retry_qty = self._extract_positive_leaves_qty(sweep_order)
            if self._retry_sweep(
                state,
                reason=f"rejected:{event.reason}",
                quantity=retry_qty,
            ):
                return
            self._clear_active_child(state)
            state.residual_sweep_pending = False
            state.sweep_retry_count = 0
            self._publish_execution_states()
            return

        self.log.warning(
            f"PostLimit order rejected: {event.client_order_id} "
            f"reason={event.reason} primary={state.primary_order_id}"
        )
        self._cancel_timer(state)
        self._restore_primary_from_active(state)
        self._clear_active_child(state)

        command = PostLimitSession(state).on_rejected(
            due_post_only=event.due_post_only,
            max_post_only_retries=self._config.max_post_only_retries,
            fallback_to_market=self._config.fallback_to_market,
            reason=str(event.reason),
            max_transient_retries=self._config.max_transient_retries,
        )
        self._execute_session_command(state, command)
        self._publish_execution_states()

    def on_order_denied(self, event: OrderDenied) -> None:
        state = self._lookup_state_by_child(event.client_order_id)
        if state is None:
            return
        if self._is_stale_child(event.client_order_id, state):
            self.log.warning(
                f"PostLimit ignoring stale denied for {event.client_order_id}: "
                f"active={state.active_order_id} primary={state.primary_order_id}"
            )
            return

        if state.active_order_kind == SpawnKind.SWEEP:
            self.log.warning(f"PostLimit sweep order denied: {event.client_order_id}")
            sweep_order = self.cache.order(event.client_order_id)
            retry_qty = self._extract_positive_leaves_qty(sweep_order)
            if self._retry_sweep(state, reason="denied", quantity=retry_qty):
                return
            self._clear_active_child(state)
            state.residual_sweep_pending = False
            state.sweep_retry_count = 0
            self._publish_execution_states()
            return

        self.log.warning(
            f"PostLimit order denied: {event.client_order_id} primary={state.primary_order_id}"
        )
        self._cancel_timer(state)
        self._restore_primary_from_active(state)
        self._clear_active_child(state)
        command = PostLimitSession(state).on_denied(
            fallback_to_market=self._config.fallback_to_market,
        )
        self._execute_session_command(state, command)
        self._publish_execution_states()

    def on_order_canceled(self, event: OrderCanceled) -> None:
        self._handle_terminal_child_event(event.client_order_id)

    def on_order_expired(self, event: OrderExpired) -> None:
        self._handle_terminal_child_event(event.client_order_id)

    def on_order_cancel_rejected(self, event: OrderCancelRejected) -> None:
        state = self._lookup_state_by_child(event.client_order_id)
        if state is None or state.active_order_kind != SpawnKind.LIMIT:
            return
        if self._is_stale_child(event.client_order_id, state):
            return

        self.log.warning(
            f"PostLimit cancel rejected: {event.client_order_id} "
            f"primary={state.primary_order_id}"
        )
        command = PostLimitSession(state).on_cancel_rejected()
        if command == SessionCommand.REARM_TIMEOUT:
            self._arm_timeout(state)
        self._publish_execution_states()

    def on_position_changed(self, event: PositionChanged) -> None:
        if not self._config.enable_residual_sweep:
            return
        if any(
            state.instrument_id == event.instrument_id
            and state.active_order_kind == SpawnKind.SWEEP
            for state in self._states.values()
        ):
            return

        for state in self._states.values():
            if not state.residual_sweep_pending:
                continue
            if state.instrument_id != event.instrument_id:
                continue
            if state.active_order_kind == SpawnKind.SWEEP:
                continue
            self._maybe_submit_residual_sweep(state)
            if state.active_order_kind == SpawnKind.SWEEP:
                break

    def on_position_closed(self, event: PositionClosed) -> None:
        for state in self._states.values():
            if state.instrument_id != event.instrument_id:
                continue
            state.residual_sweep_pending = False
            state.sweep_retry_count = 0
            if state.active_order_kind == SpawnKind.SWEEP:
                self._clear_active_child(state)
        self._publish_execution_states()

    def _submit_limit_child(self, state: OrderExecutionState) -> None:
        primary = self.cache.order(state.primary_order_id)
        if primary is None or primary.is_closed:
            self.log.warning(f"PostLimit primary unavailable: {state.primary_order_id}")
            self._fail_session(state)
            return

        instrument = self.cache.instrument(state.instrument_id)
        if instrument is None:
            self.log.error(f"PostLimit instrument not found: {state.instrument_id}")
            self._fail_session(state)
            return

        remaining = compute_remaining_quantity(self.cache, state, self.log)
        if remaining <= Quantity.zero(remaining.precision):
            self._complete_session(state)
            return
        mode = "target_quote" if state.target_quote_quantity is not None else "fixed"
        leaves_qty = getattr(primary, "leaves_qty", None)

        limit_price = determine_limit_price(
            cache=self.cache,
            instrument=instrument,
            state=state,
            config=self._config,
            post_only_default=self._config.post_only,
            chase_step_ticks_default=self._config.chase_step_ticks,
        )
        factory = ChildOrderFactory(clock=self.clock, exec_algorithm_id=self.id)
        mirror = PrimaryMirror(cache=self.cache, clock=self.clock, logger=self.log)
        spawned = factory.create_limit(
            primary=primary,
            state=state,
            quantity=remaining,
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            post_only=self._get_post_only(state),
            reduce_only=state.reduce_only,
        )
        try:
            mirrored_reserved_quantity = mirror.reduce_primary(primary, spawned.quantity)
        except Exception as exc:
            self.log.error(
                "PostLimit limit-child mirror reduce failed: "
                f"primary={state.primary_order_id} "
                f"child={spawned.client_order_id} "
                f"desired_qty={remaining} "
                f"requested={spawned.quantity} "
                f"leaves={primary.leaves_qty} "
                f"mode={mode} "
                f"error={exc}"
            )
            self._fail_session(state)
            return
        delta_qty = float(spawned.quantity) - float(mirrored_reserved_quantity)
        log_message = (
            "PostLimit limit-child prepared: "
            f"primary={state.primary_order_id} "
            f"child={spawned.client_order_id} "
            f"mode={mode} "
            f"desired_qty={remaining} "
            f"submitted_qty={spawned.quantity} "
            f"leaves_qty={leaves_qty} "
            f"mirror_reserved_qty={mirrored_reserved_quantity} "
            f"delta_qty={delta_qty:.8f}"
        )
        if delta_qty > 0:
            self.log.warning(log_message)
        else:
            self.log.debug(log_message)
        factory.register_child(
            state,
            spawned,
            SpawnKind.LIMIT,
            reserved_quantity=mirrored_reserved_quantity,
        )
        state.last_limit_price = float(limit_price)
        self._active_child_to_primary[spawned.client_order_id] = state.primary_order_id
        PostLimitSession(state).on_limit_submitted()
        self.submit_order(spawned)
        self._publish_execution_states()

    def _submit_market_child(
        self,
        state: OrderExecutionState,
        *,
        kind: SpawnKind = SpawnKind.MARKET,
        quantity: Quantity | None = None,
    ) -> None:
        primary = self.cache.order(state.primary_order_id)
        if primary is None or primary.is_closed:
            self.log.warning(f"PostLimit primary unavailable: {state.primary_order_id}")
            self._fail_session(state)
            return

        remaining = quantity or compute_remaining_quantity(self.cache, state, self.log)
        if remaining <= Quantity.zero(remaining.precision):
            self._complete_session(state)
            return
        mode = "target_quote" if state.target_quote_quantity is not None else "fixed"
        leaves_qty = getattr(primary, "leaves_qty", None)

        factory = ChildOrderFactory(clock=self.clock, exec_algorithm_id=self.id)
        spawned = factory.create_market(
            primary=primary,
            state=state,
            quantity=remaining,
            time_in_force=TimeInForce.GTC,
            reduce_only=True if kind == SpawnKind.SWEEP else state.reduce_only,
        )
        mirrored_reserved_quantity = spawned.quantity
        if kind == SpawnKind.MARKET:
            mirror = PrimaryMirror(cache=self.cache, clock=self.clock, logger=self.log)
            try:
                mirrored_reserved_quantity = mirror.reduce_primary(primary, spawned.quantity)
            except Exception as exc:
                self.log.error(
                    "PostLimit market-child mirror reduce failed: "
                    f"primary={state.primary_order_id} "
                    f"child={spawned.client_order_id} "
                    f"desired_qty={remaining} "
                    f"requested={spawned.quantity} "
                    f"leaves={primary.leaves_qty} "
                    f"mode={mode} "
                    f"error={exc}"
                )
                self._fail_session(state)
                return
            PostLimitSession(state).on_market_submitted()
        delta_qty = float(spawned.quantity) - float(mirrored_reserved_quantity)
        log_message = (
            "PostLimit market-child prepared: "
            f"primary={state.primary_order_id} "
            f"child={spawned.client_order_id} "
            f"mode={mode} "
            f"desired_qty={remaining} "
            f"submitted_qty={spawned.quantity} "
            f"leaves_qty={leaves_qty} "
            f"mirror_reserved_qty={mirrored_reserved_quantity} "
            f"delta_qty={delta_qty:.8f}"
        )
        if delta_qty > 0:
            self.log.warning(log_message)
        else:
            self.log.debug(log_message)
        state.activate_order(
            client_order_id=spawned.client_order_id,
            kind=kind,
            reserved_quantity=mirrored_reserved_quantity,
        )
        self._active_child_to_primary[spawned.client_order_id] = state.primary_order_id
        self.submit_order(spawned)
        self._publish_execution_states()

    def _maybe_submit_residual_sweep(self, state: OrderExecutionState) -> None:
        positions = self.cache.positions_open(instrument_id=state.instrument_id)
        if not positions:
            state.residual_sweep_pending = False
            state.sweep_retry_count = 0
            return

        instrument = self.cache.instrument(state.instrument_id)
        if instrument is None:
            return

        min_notional = (
            float(instrument.min_notional)
            if instrument.min_notional
            else self._config.residual_sweep_min_notional_fallback
        )
        for position in positions:
            if position.is_closed:
                continue
            residual_value = compute_residual_notional(
                cache=self.cache,
                instrument_id=state.instrument_id,
                anchor_px=state.anchor_px,
                quantity=position.quantity,
            )
            if residual_value is None or residual_value >= min_notional:
                continue

            self.log.warning(
                f"PostLimit sweep residual: {state.instrument_id} "
                f"qty={position.quantity} value={residual_value:.2f} "
                f"< min_notional={min_notional}"
            )
            state.sweep_retry_count = 0
            self._submit_market_child(
                state,
                kind=SpawnKind.SWEEP,
                quantity=position.quantity,
            )
            return

    def _handle_terminal_child_event(self, child_order_id: ClientOrderId) -> None:
        state = self._lookup_state_by_child(child_order_id)
        if state is None:
            return
        if self._is_stale_child(child_order_id, state):
            self.log.warning(
                f"PostLimit ignoring stale terminal event for {child_order_id}: "
                f"active={state.active_order_id} primary={state.primary_order_id}"
            )
            return

        if state.active_order_kind == SpawnKind.SWEEP:
            sweep_order = self.cache.order(child_order_id)
            retry_qty = self._extract_positive_leaves_qty(sweep_order)
            if self._retry_sweep(state, reason="terminal", quantity=retry_qty):
                return
            self._clear_active_child(state)
            state.residual_sweep_pending = False
            state.sweep_retry_count = 0
            self._publish_execution_states()
            return

        self._cancel_timer(state)
        self._restore_primary_from_active(state)
        self._clear_active_child(state)
        command = PostLimitSession(state).on_cancel_confirmed(
            max_chase_attempts=self._get_max_chase(state),
            fallback_to_market=self._config.fallback_to_market,
        )
        self._execute_session_command(state, command)
        self._publish_execution_states()

    def _execute_session_command(
        self,
        state: OrderExecutionState,
        command: SessionCommand,
    ) -> None:
        if command == SessionCommand.SUBMIT_LIMIT:
            self._submit_limit_child(state)
        elif command == SessionCommand.SUBMIT_MARKET:
            self._submit_market_child(state)
        elif command == SessionCommand.FAIL:
            self._fail_session(state)
        elif command == SessionCommand.COMPLETE:
            self._complete_session(state)
        elif command == SessionCommand.REARM_TIMEOUT:
            self._arm_timeout(state)
        elif command == SessionCommand.SCHEDULE_RETRY:
            self._schedule_transient_retry(state)

    def _record_failed_primary(self, order: Order, *, anchor_px: float) -> None:
        state = self._states.get(order.client_order_id)
        if state is None:
            state = OrderExecutionState(
                primary_order_id=order.client_order_id,
                instrument_id=order.instrument_id,
                side=order.side,
                total_quantity=order.quantity,
                anchor_px=anchor_px,
                reduce_only=order.is_reduce_only,
                timer_name=f"PostLimit-{order.client_order_id.value}",
                created_ns=self.clock.timestamp_ns(),
            )
            self._states[state.primary_order_id] = state
        self._fail_session(state)

    def _complete_session(self, state: OrderExecutionState) -> None:
        self._cancel_timer(state)
        self._clear_active_child(state)
        state.completed_ns = self.clock.timestamp_ns()
        PostLimitSession(state).mark_complete()
        if state.reduce_only and self._config.enable_residual_sweep:
            state.residual_sweep_pending = True
            state.sweep_retry_count = 0
        elapsed_ms = (state.completed_ns - state.created_ns) / 1_000_000
        self.log.info(
            f"PostLimit complete: {state.primary_order_id} "
            f"filled={state.filled_quantity}/{state.total_quantity} "
            f"limit_orders={state.limit_orders_submitted} "
            f"chases={state.chase_count} "
            f"post_only_retries={state.post_only_retreat_ticks} "
            f"elapsed={elapsed_ms:.0f}ms"
        )
        self._publish_execution_states()

    def _fail_session(self, state: OrderExecutionState) -> None:
        self._cancel_timer(state)
        self._clear_active_child(state)
        state.completed_ns = self.clock.timestamp_ns()
        PostLimitSession(state).mark_failed()
        self._publish_execution_states()

    def _arm_timeout(
        self,
        state: OrderExecutionState,
        timeout_override: float | None = None,
    ) -> None:
        self._cancel_timer(state)
        timeout = timeout_override or self._get_timeout_secs(state)
        self._timer_to_primary[state.timer_name] = state.primary_order_id
        self.clock.set_timer(
            name=state.timer_name,
            interval=timedelta(seconds=timeout),
            callback=self.on_time_event,
        )

    def _schedule_transient_retry(self, state: OrderExecutionState) -> None:
        """Schedule a delayed re-submission after a transient venue error."""
        retry_name = f"PostLimit-Retry-{state.primary_order_id.value}"
        # Cancel any existing retry timer for this session.
        self._retry_timer_to_primary.pop(retry_name, None)
        if retry_name in self.clock.timer_names:
            self.clock.cancel_timer(retry_name)

        delay = self._config.transient_retry_delay_secs
        self._retry_timer_to_primary[retry_name] = state.primary_order_id
        self.clock.set_timer(
            name=retry_name,
            interval=timedelta(seconds=delay),
            callback=self.on_time_event,
        )
        self.log.warning(
            f"PostLimit transient retry scheduled: "
            f"primary={state.primary_order_id} "
            f"attempt={state.transient_retry_count}/{self._config.max_transient_retries} "
            f"delay={delay}s"
        )

    def _on_retry_timer(self, primary_id: ClientOrderId) -> None:
        """Handle retry timer expiry: re-submit the limit child."""
        state = self._states.get(primary_id)
        if state is None or state.is_terminal:
            return
        if state.state != OrderState.RETRY_PENDING:
            return

        state.state = OrderState.PENDING_LIMIT
        self.log.info(
            f"PostLimit transient retry firing: "
            f"primary={state.primary_order_id} "
            f"retry={state.transient_retry_count}"
        )
        self._submit_limit_child(state)
        self._publish_execution_states()

    def _cancel_timer(self, state: OrderExecutionState) -> None:
        self._timer_to_primary.pop(state.timer_name, None)
        if state.timer_name in self.clock.timer_names:
            self.clock.cancel_timer(state.timer_name)
        # Also cancel any pending retry timer.
        retry_name = f"PostLimit-Retry-{state.primary_order_id.value}"
        self._retry_timer_to_primary.pop(retry_name, None)
        if retry_name in self.clock.timer_names:
            self.clock.cancel_timer(retry_name)

    def _clear_active_child(self, state: OrderExecutionState) -> None:
        if state.active_order_id is not None:
            self._active_child_to_primary.pop(state.active_order_id, None)
        state.clear_active_order()

    def _restore_primary_from_active(self, state: OrderExecutionState) -> None:
        if state.active_order_id is None:
            return
        child_order = self.cache.order(state.active_order_id)
        primary = self.cache.order(state.primary_order_id)
        if child_order is None or primary is None:
            return
        mirror = PrimaryMirror(cache=self.cache, clock=self.clock, logger=self.log)
        mirror.restore_primary(primary, child_order, state.active_reserved_quantity)

    def _is_stale_child(
        self,
        child_order_id: ClientOrderId,
        state: OrderExecutionState,
    ) -> bool:
        """Return True when event is for a child that is no longer active."""
        if state.active_order_id is None:
            return True
        return state.active_order_id != child_order_id

    def _handle_stale_fill(
        self,
        event: OrderFilled,
        state: OrderExecutionState,
    ) -> None:
        """Track a late fill from a deactivated child and cancel-resubmit active child."""
        if state.is_terminal:
            return

        fill_qty = event.last_qty
        fill_px = float(event.last_px)
        state.fill_cost += fill_px * float(fill_qty)
        state.filled_quantity = Quantity(
            state.filled_quantity + fill_qty,
            state.filled_quantity.precision,
        )
        if state.target_quote_quantity is not None:
            state.filled_quote_quantity += fill_px * float(fill_qty) * state.contract_multiplier

        self.log.warning(
            f"PostLimit stale fill tracked: child={event.client_order_id} "
            f"active={state.active_order_id} primary={state.primary_order_id} "
            f"last_qty={fill_qty} filled={state.filled_quantity}/{state.total_quantity}"
        )

        remaining = compute_remaining_quantity(self.cache, state, self.log)
        if remaining <= Quantity.zero(remaining.precision):
            if state.active_order_id is not None:
                active_order = self.cache.order(state.active_order_id)
                if active_order is not None and not active_order.is_closed:
                    self.cancel_order(active_order)
            self._complete_session(state)
            return

        if state.active_order_id is not None:
            active_order = self.cache.order(state.active_order_id)
            if active_order is not None and not active_order.is_closed:
                self.cancel_order(active_order)

        self._publish_execution_states()

    def _lookup_state_by_child(
        self,
        child_order_id: ClientOrderId,
    ) -> OrderExecutionState | None:
        primary_id = self._active_child_to_primary.get(child_order_id)
        if primary_id is not None:
            return self._states.get(primary_id)

        order = self.cache.order(child_order_id)
        if order is not None and order.exec_spawn_id is not None:
            return self._states.get(order.exec_spawn_id)
        return None

    def _ensure_quote_subscription(self, instrument_id: InstrumentId) -> None:
        if instrument_id in self._subscribed_instruments:
            return
        self.subscribe_quote_ticks(instrument_id)
        self._subscribed_instruments.add(instrument_id)
        self.log.info(f"PostLimit subscribed QuoteTick: {instrument_id}")

    def _rebuild_runtime_indexes(self, *, rearm_timers: bool) -> None:
        self._active_child_to_primary.clear()
        self._timer_to_primary.clear()
        for state in self._states.values():
            if not state.is_terminal:
                self._ensure_quote_subscription(state.instrument_id)
            if state.active_order_id is not None:
                self._active_child_to_primary[state.active_order_id] = state.primary_order_id
            if rearm_timers and state.state == OrderState.WORKING_LIMIT:
                self._arm_timeout(state)
            if rearm_timers and state.state == OrderState.RETRY_PENDING:
                self._schedule_transient_retry(state)

    def _get_timeout_secs(self, state: OrderExecutionState) -> float:
        return state.timeout_secs if state.timeout_secs is not None else self._config.timeout_secs

    def _get_max_chase(self, state: OrderExecutionState) -> int:
        return (
            state.max_chase_attempts
            if state.max_chase_attempts is not None
            else self._config.max_chase_attempts
        )

    def _get_post_only(self, state: OrderExecutionState) -> bool:
        return state.post_only if state.post_only is not None else self._config.post_only

    def _extract_positive_leaves_qty(self, order: Order | None) -> Quantity | None:
        if order is None:
            return None
        leaves_qty = getattr(order, "leaves_qty", None)
        if leaves_qty is None:
            return None
        if leaves_qty <= Quantity.zero(leaves_qty.precision):
            return None
        return leaves_qty

    def _resolve_sweep_retry_quantity(self, state: OrderExecutionState) -> Quantity | None:
        if state.active_order_id is not None:
            active_order = self.cache.order(state.active_order_id)
            active_leaves = self._extract_positive_leaves_qty(active_order)
            if active_leaves is not None:
                return active_leaves

        for position in self.cache.positions_open(instrument_id=state.instrument_id):
            if position.is_closed:
                continue
            if position.quantity <= Quantity.zero(position.quantity.precision):
                continue
            return position.quantity
        return None

    def _retry_sweep(
        self,
        state: OrderExecutionState,
        *,
        reason: str,
        quantity: Quantity | None = None,
    ) -> bool:
        if state.sweep_retry_count >= self._config.max_sweep_retries:
            self.log.warning(
                "PostLimit sweep retries exhausted: "
                f"primary={state.primary_order_id} "
                f"retries={state.sweep_retry_count}/{self._config.max_sweep_retries} "
                f"reason={reason}"
            )
            return False

        retry_qty = quantity or self._resolve_sweep_retry_quantity(state)
        if retry_qty is None or retry_qty <= Quantity.zero(retry_qty.precision):
            self.log.warning(
                "PostLimit sweep retry skipped due to non-positive remaining quantity: "
                f"primary={state.primary_order_id} reason={reason}"
            )
            return False

        state.sweep_retry_count += 1
        self._clear_active_child(state)
        self.log.warning(
            "PostLimit sweep retry: "
            f"primary={state.primary_order_id} "
            f"attempt={state.sweep_retry_count}/{self._config.max_sweep_retries} "
            f"qty={retry_qty} reason={reason}"
        )
        self._submit_market_child(state, kind=SpawnKind.SWEEP, quantity=retry_qty)
        return True

    def _is_order_fully_filled(self, order: Order) -> bool:
        leaves_qty = getattr(order, "leaves_qty", None)
        if leaves_qty is not None:
            return leaves_qty <= Quantity.zero(leaves_qty.precision)

        status = getattr(order, "status", None)
        return bool(status is not None and getattr(status, "name", None) == "FILLED")

    def _encode_states(self) -> bytes:
        return encode_execution_states(self._states)

    def _decode_states(self, raw: bytes) -> dict[ClientOrderId, OrderExecutionState]:
        try:
            return decode_execution_states(raw)
        except Exception:
            legacy_states = pickle.loads(raw)  # noqa: S301
            return legacy_states

    def _publish_execution_states(self) -> None:
        try:
            self.cache.add(EXECUTION_STATES_CACHE_KEY, self._encode_states())
        except Exception:
            # Cache publication is best-effort and should not break execution hooks.
            return
