# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""PostLimitExecAlgorithm - Post-only limit order execution with chase and market fallback."""

from __future__ import annotations

import pickle
from datetime import timedelta
from typing import TYPE_CHECKING

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.algorithm import ExecAlgorithm
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import LimitOrder, MarketOrder

from nautilus_quants.utils.cache_keys import EXECUTION_STATES_CACHE_KEY
from nautilus_quants.execution.post_limit.config import PostLimitExecAlgorithmConfig
from nautilus_quants.execution.post_limit.state import OrderExecutionState, OrderState

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderCanceled, OrderFilled, OrderRejected
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.orders import Order


def compute_limit_price(
    tick: float,
    side: OrderSide,
    anchor_px: float,
    offset_ticks: int,
    chase_count: int,
    chase_step_ticks: int,
    post_only: bool,
    best_bid: float | None,
    best_ask: float | None,
) -> float:
    """Pure-function price calculation for testability.

    Returns the raw float price (caller is responsible for rounding via
    ``instrument.make_price()``).
    """
    total_offset = offset_ticks + chase_count * chase_step_ticks

    if side == OrderSide.BUY:
        base = best_bid if best_bid is not None else anchor_px
        price = base + total_offset * tick
        if post_only and best_ask is not None:
            # Clamp to avoid crossing the ask (stay on maker side)
            price = min(price, best_ask - tick)
    else:  # SELL
        base = best_ask if best_ask is not None else anchor_px
        price = base - total_offset * tick
        if post_only and best_bid is not None:
            # Clamp to avoid crossing the bid (stay on maker side)
            # At 1-tick spread this correctly clamps to best_ask
            price = max(price, best_bid + tick)

    # Guard: crossed/locked book could produce negative; floor at 1 tick
    return max(price, tick)


def _spawn_linkage_fields(primary: Order) -> dict[str, object]:
    """Fields that preserve parent contingent/list linkage on spawned orders."""
    return {
        "contingency_type": primary.contingency_type,
        "order_list_id": primary.order_list_id,
        "linked_order_ids": primary.linked_order_ids,
        "parent_order_id": primary.parent_order_id,
    }


def _normalize_qty_or_zero(
    *,
    instrument: Instrument,
    raw_qty: float,
    precision: int,
    instrument_id: InstrumentId,
    primary_order_id: ClientOrderId,
    logger,
) -> Quantity:
    """Normalize quantity to instrument increment, returning zero for dust residuals."""
    if raw_qty <= 0:
        return Quantity.zero(precision)

    try:
        return instrument.make_qty(raw_qty, round_down=True)
    except ValueError as exc:
        if "rounded to zero" not in str(exc):
            raise

        logger.warning(
            "PostLimit residual below increment, treating as zero: "
            f"instrument_id={instrument_id} "
            f"primary_order_id={primary_order_id} "
            f"raw_qty={raw_qty} "
            f"size_increment={instrument.size_increment} "
            f"size_precision={instrument.size_precision}"
        )
        return Quantity.zero(precision)


class PostLimitExecAlgorithm(ExecAlgorithm):
    """Execution algorithm that converts MarketOrders into post-only limit orders.

    Flow: BBO-pegged limit order -> timeout -> chase (cancel + re-post at new price)
    -> max chase exhausted -> market order fallback.

    The algorithm reads ``anchor_px`` from ``exec_algorithm_params`` on the
    incoming (primary) MarketOrder and uses it as fallback when OrderBook data
    is unavailable.
    """

    def __init__(self, config: PostLimitExecAlgorithmConfig | None = None) -> None:
        if config is None:
            config = PostLimitExecAlgorithmConfig()
        super().__init__(config)

        self._config: PostLimitExecAlgorithmConfig = config

        # primary_order_id -> OrderExecutionState
        self._states: dict[ClientOrderId, OrderExecutionState] = {}
        # spawned_order_id -> primary_order_id (reverse lookup)
        self._spawned_to_primary: dict[ClientOrderId, ClientOrderId] = {}
        # primary_order_id -> spawn sequence counter (for hyphen-free spawn IDs)
        self._spawn_sequence: dict[ClientOrderId, int] = {}
        # Instruments with active QuoteTick subscriptions
        self._subscribed_instruments: set[InstrumentId] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.log.info(
            f"PostLimitExecAlgorithm started: "
            f"offset_ticks={self._config.offset_ticks}, "
            f"timeout_secs={self._config.timeout_secs}, "
            f"max_chase={self._config.max_chase_attempts}, "
            f"chase_step_ticks={self._config.chase_step_ticks}, "
            f"fallback_to_market={self._config.fallback_to_market}, "
            f"post_only={self._config.post_only}, "
            f"max_post_only_retries={self._config.max_post_only_retries}"
        )

    def on_stop(self) -> None:
        self.clock.cancel_timers()
        if self._states:
            self.cache.add(EXECUTION_STATES_CACHE_KEY, pickle.dumps(self._states))
        self.log.info(
            f"PostLimitExecAlgorithm stopped: "
            f"tracked_sequences={len(self._states)}, "
            f"active={sum(1 for s in self._states.values() if not s.is_terminal)}"
        )

    def on_save(self) -> dict[str, bytes]:
        return {"states": pickle.dumps(self._states)}

    def on_load(self, state: dict[str, bytes]) -> None:
        raw = state.get("states")
        if raw is None:
            return
        self._states = pickle.loads(raw)  # noqa: S301
        # Rebuild reverse lookup and timers for active sequences
        for exec_state in self._states.values():
            if exec_state.current_limit_order_id is not None:
                self._spawned_to_primary[exec_state.current_limit_order_id] = (
                    exec_state.primary_order_id
                )
            if exec_state.state == OrderState.ACTIVE:
                self._start_timeout_timer(exec_state)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def on_order(self, order: Order) -> None:
        """Handle an incoming primary order.

        The strategy submits a MarketOrder with ``exec_algorithm_id="PostLimit"``
        and optional ``exec_algorithm_params`` containing:
        - ``anchor_px``: target fill price (required fallback, float as str)
        - ``timeout_secs``: per-order timeout override
        - ``max_chase_attempts``: per-order chase limit override
        - ``chase_step_ticks``: per-order chase step override
        - ``post_only``: per-order post_only override ("true"/"false")
        - ``target_quote_quantity``: target USDT value for qty recalculation on chase
        - ``contract_multiplier``: instrument contract multiplier (default 1.0)
        - ``intent``: strategy intent for diagnostics ("OPEN" or "FLIP")
        """
        if order.order_type != OrderType.MARKET:
            self.log.warning(
                f"PostLimit only handles MarketOrders, got {order.order_type}. "
                f"Passing through as-is: {order.client_order_id}"
            )
            self.submit_order(order)
            return

        # Parse exec_algorithm_params
        params = order.exec_algorithm_params or {}
        anchor_px_str = params.get("anchor_px")
        try:
            anchor_px = float(anchor_px_str) if anchor_px_str else 0.0
        except (ValueError, TypeError):
            self.log.error(
                f"PostLimit: invalid anchor_px={anchor_px_str!r} "
                f"for {order.client_order_id}, passing order through"
            )
            self.submit_order(order)
            return

        # Create execution state
        exec_state = OrderExecutionState(
            primary_order_id=order.client_order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            total_quantity=order.quantity,
            anchor_px=anchor_px,
            reduce_only=order.is_reduce_only,
            created_ns=self.clock.timestamp_ns(),
            timer_name=f"PostLimit-{order.client_order_id.value}",
            filled_quantity=Quantity.zero(order.quantity.precision),
        )

        # Parse per-order overrides
        if "timeout_secs" in params:
            exec_state.timeout_secs = float(params["timeout_secs"])
        if "max_chase_attempts" in params:
            exec_state.max_chase_attempts = int(params["max_chase_attempts"])
        if "chase_step_ticks" in params:
            exec_state.chase_step_ticks = int(params["chase_step_ticks"])
        if "post_only" in params:
            exec_state.post_only = params["post_only"].lower() == "true"
        if "target_quote_quantity" in params:
            try:
                target_quote_quantity = float(params["target_quote_quantity"])
            except (ValueError, TypeError):
                self.log.error(
                    "PostLimit invalid target_quote_quantity, ignoring: "
                    f"primary_order_id={order.client_order_id} "
                    f"value={params['target_quote_quantity']!r}"
                )
            else:
                if target_quote_quantity <= 0:
                    self.log.error(
                        "PostLimit target_quote_quantity must be > 0, ignoring: "
                        f"primary_order_id={order.client_order_id} "
                        f"value={target_quote_quantity}"
                    )
                else:
                    exec_state.target_quote_quantity = target_quote_quantity

        if "contract_multiplier" in params:
            try:
                contract_multiplier = float(params["contract_multiplier"])
            except (ValueError, TypeError):
                self.log.error(
                    "PostLimit invalid contract_multiplier, using default 1.0: "
                    f"primary_order_id={order.client_order_id} "
                    f"value={params['contract_multiplier']!r}"
                )
            else:
                if contract_multiplier <= 0:
                    self.log.error(
                        "PostLimit contract_multiplier must be > 0, using default 1.0: "
                        f"primary_order_id={order.client_order_id} "
                        f"value={contract_multiplier}"
                    )
                else:
                    exec_state.contract_multiplier = contract_multiplier

        if "intent" in params:
            exec_state.intent = str(params["intent"]).upper()

        self._states[order.client_order_id] = exec_state

        # Subscribe to QuoteTick for real-time BBO (once per instrument)
        if exec_state.instrument_id not in self._subscribed_instruments:
            self.subscribe_quote_ticks(exec_state.instrument_id)
            self._subscribed_instruments.add(exec_state.instrument_id)
            self.log.info(f"PostLimit subscribed QuoteTick: {exec_state.instrument_id}")

        self.log.info(
            f"PostLimit on_order: {order.client_order_id} "
            f"{order.side.name} {order.quantity} {order.instrument_id} "
            f"anchor_px={anchor_px} "
            f"intent={exec_state.intent} "
            f"target_quote_quantity={exec_state.target_quote_quantity}"
        )

        self._spawn_and_submit_limit(exec_state)

    # ------------------------------------------------------------------
    # Spawn ID generation (hyphen-free for OKX compatibility)
    # ------------------------------------------------------------------

    def _generate_spawn_id(self, primary: Order) -> ClientOrderId:
        """Generate a hyphen-free spawn order ID.

        OKX rejects clOrdId values containing hyphens. The base ExecAlgorithm's
        ``spawn_limit``/``spawn_market`` append ``-E{n}`` which introduces a
        hyphen. This method produces ``{base}E{n}`` instead.
        """
        seq = self._spawn_sequence.get(primary.client_order_id, 0) + 1
        self._spawn_sequence[primary.client_order_id] = seq
        return ClientOrderId(f"{primary.client_order_id.value}E{seq}")

    def _create_spawned_limit(
        self,
        primary: Order,
        quantity: Quantity,
        price: Price,
        time_in_force: TimeInForce,
        post_only: bool,
        reduce_only: bool,
    ) -> LimitOrder:
        """Create a spawned LimitOrder with a hyphen-free client order ID."""
        spawn_id = self._generate_spawn_id(primary)
        return LimitOrder(
            trader_id=primary.trader_id,
            strategy_id=primary.strategy_id,
            instrument_id=primary.instrument_id,
            client_order_id=spawn_id,
            order_side=primary.side,
            quantity=quantity,
            price=price,
            init_id=UUID4(),
            ts_init=self.clock.timestamp_ns(),
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            exec_algorithm_id=self.id,
            exec_spawn_id=primary.client_order_id,
            **_spawn_linkage_fields(primary),
        )

    def _create_spawned_market(
        self,
        primary: Order,
        quantity: Quantity,
        time_in_force: TimeInForce,
        reduce_only: bool,
    ) -> MarketOrder:
        """Create a spawned MarketOrder with a hyphen-free client order ID."""
        spawn_id = self._generate_spawn_id(primary)
        return MarketOrder(
            trader_id=primary.trader_id,
            strategy_id=primary.strategy_id,
            instrument_id=primary.instrument_id,
            client_order_id=spawn_id,
            order_side=primary.side,
            quantity=quantity,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            init_id=UUID4(),
            ts_init=self.clock.timestamp_ns(),
            exec_algorithm_id=self.id,
            exec_spawn_id=primary.client_order_id,
            **_spawn_linkage_fields(primary),
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _get_timeout_secs(self, state: OrderExecutionState) -> float:
        return state.timeout_secs if state.timeout_secs is not None else self._config.timeout_secs

    def _get_max_chase(self, state: OrderExecutionState) -> int:
        return (
            state.max_chase_attempts
            if state.max_chase_attempts is not None
            else self._config.max_chase_attempts
        )

    def _get_chase_step_ticks(self, state: OrderExecutionState) -> int:
        return (
            state.chase_step_ticks
            if state.chase_step_ticks is not None
            else self._config.chase_step_ticks
        )

    def _get_post_only(self, state: OrderExecutionState) -> bool:
        return state.post_only if state.post_only is not None else self._config.post_only

    def _remaining_quantity(self, state: OrderExecutionState) -> Quantity:
        """Calculate unfilled quantity.

        When target_quote_quantity is set, recalculates remaining quantity from
        remaining USDT value / current BBO price. This ensures chase iterations
        always target the correct USDT value regardless of price drift.
        """
        instrument = self.cache.instrument(state.instrument_id)
        if instrument is None:
            return state.total_quantity

        if state.target_quote_quantity is not None:
            remaining_value = state.target_quote_quantity - state.filled_quote_quantity
            if remaining_value <= 0:
                return Quantity.zero(state.total_quantity.precision)

            # Get current BBO for recalculation
            exec_price = self._get_bbo_price(state)
            if exec_price is not None and exec_price > 0:
                raw_qty = remaining_value / (exec_price * state.contract_multiplier)
                return _normalize_qty_or_zero(
                    instrument=instrument,
                    raw_qty=raw_qty,
                    precision=state.total_quantity.precision,
                    instrument_id=state.instrument_id,
                    primary_order_id=state.primary_order_id,
                    logger=self.log,
                )

        # Fallback: fixed quantity arithmetic
        remaining_raw = float(state.total_quantity - state.filled_quantity)
        return _normalize_qty_or_zero(
            instrument=instrument,
            raw_qty=remaining_raw,
            precision=state.total_quantity.precision,
            instrument_id=state.instrument_id,
            primary_order_id=state.primary_order_id,
            logger=self.log,
        )

    def _get_bbo_price(self, state: OrderExecutionState) -> float | None:
        """Get best available execution price for quantity recalculation."""
        quote = self.cache.quote_tick(state.instrument_id)
        if quote is not None:
            if state.side == OrderSide.BUY:
                return float(quote.ask_price) if quote.ask_price else None
            return float(quote.bid_price) if quote.bid_price else None
        return state.anchor_px if state.anchor_px > 0 else None

    def _determine_limit_price(
        self,
        instrument: Instrument,
        side: OrderSide,
        anchor_px: float,
        chase_count: int,
        state: OrderExecutionState,
    ) -> Price:
        """Calculate the limit price based on BBO with tick offsets.

        Priority: QuoteTick BBO > OrderBook BBO > anchor_px fallback.

        QuoteTick (exchange bookTicker) is preferred over OrderBook because:
        - Lower latency: pushed on every BBO change vs periodic depth snapshots
        - More reliable: direct exchange BBO vs locally reconstructed book top
        - Always available: no L2 subscription required

        Applies offset_ticks + chase cumulative offset.
        Clamps to avoid crossing the opposite BBO when post_only is True.
        """
        tick = float(instrument.price_increment)

        best_bid: float | None = None
        best_ask: float | None = None

        # Priority 1: QuoteTick (exchange BBO stream, lowest latency)
        quote = self.cache.quote_tick(state.instrument_id)
        if quote is not None:
            if quote.bid_price:
                best_bid = float(quote.bid_price)
            if quote.ask_price:
                best_ask = float(quote.ask_price)

        # Priority 2: OrderBook top-of-book (locally reconstructed from L2 deltas)
        if best_bid is None or best_ask is None:
            book = self.cache.order_book(state.instrument_id)
            if book is not None:
                if best_bid is None:
                    bid_price = book.best_bid_price()
                    if bid_price is not None:
                        best_bid = float(bid_price)
                if best_ask is None:
                    ask_price = book.best_ask_price()
                    if ask_price is not None:
                        best_ask = float(ask_price)

        effective_offset = self._config.offset_ticks - state.post_only_retreat_ticks
        post_only = self._get_post_only(state)

        raw_price = compute_limit_price(
            tick=tick,
            side=side,
            anchor_px=anchor_px,
            offset_ticks=effective_offset,
            chase_count=0 if post_only else chase_count,  # post_only: re-peg only
            chase_step_ticks=self._get_chase_step_ticks(state),
            post_only=post_only,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        return instrument.make_price(raw_price)

    def _spawn_and_submit_limit(self, state: OrderExecutionState) -> None:
        """Spawn a limit order from the primary and submit it."""
        primary = self.cache.order(state.primary_order_id)
        if primary is None or primary.is_closed:
            self.log.warning(
                f"Primary order {state.primary_order_id} is closed/missing, "
                f"cannot spawn limit"
            )
            self._fail_order(state)
            return

        instrument = self.cache.instrument(state.instrument_id)
        if instrument is None:
            self.log.error(f"Instrument not found: {state.instrument_id}")
            self._fail_order(state)
            return

        remaining = self._remaining_quantity(state)
        if remaining <= Quantity.zero(remaining.precision):
            self.log.info(
                f"PostLimit {state.primary_order_id}: "
                f"no remaining quantity, completing"
            )
            self._complete_order(state)
            return

        limit_price = self._determine_limit_price(
            instrument=instrument,
            side=state.side,
            anchor_px=state.anchor_px,
            chase_count=state.chase_count,
            state=state,
        )

        spawned = self._create_spawned_limit(
            primary=primary,
            quantity=remaining,
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            post_only=self._get_post_only(state),
            reduce_only=state.reduce_only,
        )

        # Track the spawned order
        state.current_limit_order_id = spawned.client_order_id
        state.limit_orders_submitted += 1
        state.last_limit_price = float(limit_price)

        self._spawned_to_primary[spawned.client_order_id] = state.primary_order_id

        self.submit_order(spawned)

        # Transition to ACTIVE
        if state.state == OrderState.PENDING:
            state.transition_to(OrderState.ACTIVE)
        elif state.state == OrderState.CHASING:
            # Re-entering ACTIVE after chase cancel
            state.transition_to(OrderState.ACTIVE)

        self._start_timeout_timer(state)

        self.log.debug(
            f"PostLimit spawned limit: {spawned.client_order_id} "
            f"price={limit_price} side={state.side.name} qty={remaining} "
            f"post_only={self._get_post_only(state)} "
            f"chase={state.chase_count}"
        )

    def _start_timeout_timer(self, state: OrderExecutionState) -> None:
        """Start (or restart) the timeout timer for the current limit order."""
        # Cancel existing timer if any
        if state.timer_name in self.clock.timer_names:
            self.clock.cancel_timer(state.timer_name)

        timeout = self._get_timeout_secs(state)
        self.clock.set_timer(
            name=state.timer_name,
            interval=timedelta(seconds=timeout),
            callback=self._on_timeout,
        )

    def _cancel_timer(self, state: OrderExecutionState) -> None:
        """Cancel the timeout timer if running."""
        if state.timer_name in self.clock.timer_names:
            self.clock.cancel_timer(state.timer_name)

    def _on_timeout(self, event) -> None:
        """Handle timeout: chase or fall back to market."""
        # Extract primary_order_id from timer name
        # Timer name format: "PostLimit-{client_order_id_value}"
        prefix = "PostLimit-"
        if not event.name.startswith(prefix):
            return

        order_id_str = event.name[len(prefix):]
        # Find the matching state
        state: OrderExecutionState | None = None
        for s in self._states.values():
            if s.primary_order_id.value == order_id_str:
                state = s
                break

        if state is None or state.is_terminal:
            return

        if state.state != OrderState.ACTIVE:
            self.log.debug(
                f"PostLimit timeout for {state.primary_order_id} "
                f"but state={state.state.value}, ignoring"
            )
            return

        # Check if limit order is still open
        if state.current_limit_order_id is None:
            return

        current_order = self.cache.order(state.current_limit_order_id)
        if current_order is None or current_order.is_closed:
            # Order already filled/canceled, nothing to do
            return

        max_chase = self._get_max_chase(state)

        if state.chase_count < max_chase:
            self.log.info(
                f"PostLimit timeout: {state.primary_order_id} "
                f"chase {state.chase_count + 1}/{max_chase}"
            )
            self._chase_order(state)
        else:
            self.log.info(
                f"PostLimit timeout: {state.primary_order_id} "
                f"max chase exhausted ({max_chase}), falling back to market"
            )
            self._fallback_to_market(state)

    def _chase_order(self, state: OrderExecutionState) -> None:
        """Cancel current limit and prepare to re-post at a more aggressive price."""
        state.chase_count += 1
        state.post_only_retreat_ticks = 0  # Reset retreat on chase (new pricing round)
        state.transition_to(OrderState.CHASING)

        if state.current_limit_order_id is not None:
            current_order = self.cache.order(state.current_limit_order_id)
            if current_order is not None and not current_order.is_closed:
                self.cancel_order(current_order)
            else:
                # Order already closed (filled between timeout and cancel)
                # on_order_filled should handle this
                pass

    def _fallback_to_market(self, state: OrderExecutionState) -> None:
        """Switch to market order fallback."""
        if not self._config.fallback_to_market:
            self.log.warning(
                f"PostLimit {state.primary_order_id}: "
                f"fallback_to_market=False and max chase exhausted, failing"
            )
            self._fail_order(state)
            return

        state.used_market_fallback = True
        state.transition_to(OrderState.MARKET_FALLBACK)

        # Cancel the current limit order if still open
        if state.current_limit_order_id is not None:
            current_order = self.cache.order(state.current_limit_order_id)
            if current_order is not None and not current_order.is_closed:
                self.cancel_order(current_order)
                return  # on_order_canceled will spawn the market order

        # No active limit order to cancel, spawn market immediately
        self._spawn_and_submit_market(state)

    def _spawn_and_submit_market(self, state: OrderExecutionState) -> None:
        """Spawn and submit a market order for the remaining quantity."""
        primary = self.cache.order(state.primary_order_id)
        if primary is None or primary.is_closed:
            self.log.warning(
                f"Primary {state.primary_order_id} closed, cannot spawn market fallback"
            )
            self._fail_order(state)
            return

        remaining = self._remaining_quantity(state)
        if remaining <= Quantity.zero(remaining.precision):
            self._complete_order(state)
            return

        spawned = self._create_spawned_market(
            primary=primary,
            quantity=remaining,
            time_in_force=TimeInForce.GTC,
            reduce_only=state.reduce_only,
        )

        self._spawned_to_primary[spawned.client_order_id] = state.primary_order_id
        state.current_limit_order_id = spawned.client_order_id

        self.submit_order(spawned)

        self.log.info(
            f"PostLimit market fallback: {spawned.client_order_id} "
            f"qty={remaining} for primary {state.primary_order_id}"
        )

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle fill events for spawned orders."""
        state = self._lookup_state_by_spawned(event.client_order_id)
        if state is None:
            return

        # Update filled quantity and cost
        fill_qty = event.last_qty
        fill_px = float(event.last_px)
        state.fill_cost += fill_px * float(fill_qty)
        state.filled_quantity = Quantity(
            state.filled_quantity + fill_qty,
            state.filled_quantity.precision,
        )

        # Track filled quote quantity for target_quote_quantity recalculation
        if state.target_quote_quantity is not None:
            state.filled_quote_quantity += (
                fill_px * float(fill_qty) * state.contract_multiplier
            )

        remaining = self._remaining_quantity(state)
        is_fully_filled = remaining <= Quantity.zero(remaining.precision)

        if is_fully_filled:
            self.log.info(
                f"PostLimit filled: {state.primary_order_id} "
                f"total_qty={state.total_quantity} "
                f"limit_orders={state.limit_orders_submitted} "
                f"chase_count={state.chase_count}"
            )
            self._complete_order(state)
        else:
            self.log.debug(
                f"PostLimit partial fill: {state.primary_order_id} "
                f"filled={state.filled_quantity}/{state.total_quantity} "
                f"remaining={remaining}"
            )
            # Stay in current state, wait for more fills or timeout

    def on_order_canceled(self, event: OrderCanceled) -> None:
        """Handle cancel events - re-spawn if chasing, market fallback if needed."""
        state = self._lookup_state_by_spawned(event.client_order_id)
        if state is None:
            return

        if state.state == OrderState.CHASING:
            # Cancel completed, now re-spawn at a better price
            self.log.debug(
                f"PostLimit chase cancel confirmed: {state.primary_order_id} "
                f"chase={state.chase_count}"
            )
            self._spawn_and_submit_limit(state)

        elif state.state == OrderState.MARKET_FALLBACK:
            # Limit cancel completed, now spawn market
            self.log.debug(
                f"PostLimit fallback cancel confirmed: {state.primary_order_id}"
            )
            self._spawn_and_submit_market(state)

        elif state.state == OrderState.ACTIVE:
            # Unexpected cancel while ACTIVE (e.g., exchange-initiated)
            self.log.warning(
                f"PostLimit unexpected cancel in ACTIVE: {state.primary_order_id}"
            )
            # Treat as needing market fallback
            self._fallback_to_market(state)

    def on_order_rejected(self, event: OrderRejected) -> None:
        """Handle rejection - retry on POST_ONLY or fall back to market order."""
        state = self._lookup_state_by_spawned(event.client_order_id)
        if state is None:
            return

        self.log.warning(
            f"PostLimit order rejected: {event.client_order_id} "
            f"reason={event.reason} "
            f"primary={state.primary_order_id}"
        )

        self._cancel_timer(state)

        # POST_ONLY rejection: retreat 1 more tick and retry
        if (
            event.due_post_only
            and state.post_only_retreat_ticks < self._config.max_post_only_retries
            and state.state in (OrderState.ACTIVE, OrderState.PENDING, OrderState.CHASING)
        ):
            state.post_only_retreat_ticks += 1
            self.log.info(
                f"PostLimit POST_ONLY retry: {state.primary_order_id} "
                f"retreat +{state.post_only_retreat_ticks} ticks"
            )
            self._spawn_and_submit_limit(state)
            return

        # Non-POST_ONLY rejection, or retries exhausted → market fallback
        if state.state in (OrderState.ACTIVE, OrderState.PENDING, OrderState.CHASING):
            self._fallback_to_market(state)
        elif state.state == OrderState.MARKET_FALLBACK:
            # Market order also rejected - terminal failure
            self.log.error(
                f"PostLimit market fallback rejected: {state.primary_order_id}"
            )
            self._fail_order(state)

    def on_order_expired(self, event) -> None:
        """Handle expiration as if it were a cancel."""
        state = self._lookup_state_by_spawned(event.client_order_id)
        if state is None:
            return

        self.log.info(
            f"PostLimit order expired: {event.client_order_id} "
            f"primary={state.primary_order_id}"
        )

        # Treat expiry like a cancel in ACTIVE state
        if state.state == OrderState.ACTIVE:
            max_chase = self._get_max_chase(state)
            if state.chase_count < max_chase:
                state.chase_count += 1
                state.transition_to(OrderState.CHASING)
                self._spawn_and_submit_limit(state)
            else:
                self._fallback_to_market(state)

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _fail_order(self, state: OrderExecutionState) -> None:
        """Mark execution sequence as failed and clean up."""
        self._cancel_timer(state)
        state.completed_ns = self.clock.timestamp_ns()
        state.transition_to(OrderState.FAILED)

    def _complete_order(self, state: OrderExecutionState) -> None:
        """Mark execution sequence as complete and sweep residual positions."""
        self._cancel_timer(state)
        state.completed_ns = self.clock.timestamp_ns()

        if not state.is_terminal:
            state.transition_to(OrderState.COMPLETED)

        elapsed_ms = (self.clock.timestamp_ns() - state.created_ns) / 1_000_000

        self.log.info(
            f"PostLimit complete: {state.primary_order_id} "
            f"filled={state.filled_quantity}/{state.total_quantity} "
            f"limit_orders={state.limit_orders_submitted} "
            f"chases={state.chase_count} "
            f"post_only_retries={state.post_only_retreat_ticks} "
            f"elapsed={elapsed_ms:.0f}ms"
        )

        # Sweep residual positions after reduce_only (close) orders
        if state.reduce_only:
            self._sweep_residual_position(state)

    def _sweep_residual_position(self, state: OrderExecutionState) -> None:
        """Detect and sweep residual positions below min notional value.

        After a reduce_only order completes, precision loss from multiple
        partial fills can leave a tiny residual position (e.g., 0.09 ETC ≈ 0.79
        USDT). This prevents ``on_position_closed`` from firing, blocking the
        ExposureManager. A market sweep order brings the position to zero.
        """
        positions = self.cache.positions_open(instrument_id=state.instrument_id)
        if not positions:
            return

        instrument = self.cache.instrument(state.instrument_id)
        if instrument is None:
            return

        primary = self.cache.order(state.primary_order_id)
        if primary is None or primary.is_closed:
            return

        for position in positions:
            if position.is_closed:
                continue

            # Estimate residual value
            quote = self.cache.quote_tick(state.instrument_id)
            if quote is not None:
                mid_price = (float(quote.bid_price) + float(quote.ask_price)) / 2
            else:
                mid_price = state.anchor_px

            if mid_price <= 0:
                continue

            remaining_value = float(position.quantity) * mid_price
            min_notional = (
                float(instrument.min_notional) if instrument.min_notional else 5.0
            )

            if remaining_value < min_notional:
                self.log.warning(
                    f"PostLimit sweep residual: {state.instrument_id} "
                    f"qty={position.quantity} value={remaining_value:.2f} "
                    f"< min_notional={min_notional}"
                )
                sweep = self._create_spawned_market(
                    primary=primary,
                    quantity=position.quantity,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
                self._spawned_to_primary[sweep.client_order_id] = (
                    state.primary_order_id
                )
                self.submit_order(sweep)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_state_by_spawned(
        self, spawned_order_id: ClientOrderId
    ) -> OrderExecutionState | None:
        """Look up the execution state for a spawned order's event."""
        primary_id = self._spawned_to_primary.get(spawned_order_id)
        if primary_id is None:
            return None
        return self._states.get(primary_id)
