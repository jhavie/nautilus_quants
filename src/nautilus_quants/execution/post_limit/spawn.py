# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Child order creation and primary quantity mirroring for PostLimit."""

from __future__ import annotations

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.events.order import OrderUpdated
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import LimitOrder, MarketOrder

from nautilus_quants.execution.post_limit.state import OrderExecutionState, SpawnKind


def spawn_linkage_fields(primary) -> dict[str, object]:
    """Return contingent-order linkage shared by all children."""
    return {
        "contingency_type": primary.contingency_type,
        "order_list_id": primary.order_list_id,
        "linked_order_ids": primary.linked_order_ids,
        "parent_order_id": primary.parent_order_id,
        "tags": primary.tags,
    }


class ChildOrderFactory:
    """Creates OKX-safe child orders while preserving primary metadata."""

    def __init__(self, *, clock, exec_algorithm_id) -> None:
        self._clock = clock
        self._exec_algorithm_id = exec_algorithm_id

    def create_limit(
        self,
        *,
        primary,
        state: OrderExecutionState,
        quantity: Quantity,
        price: Price,
        time_in_force,
        post_only: bool,
        reduce_only: bool,
    ) -> LimitOrder:
        spawn_id = self._next_spawn_id(state)
        return LimitOrder(
            trader_id=primary.trader_id,
            strategy_id=primary.strategy_id,
            instrument_id=primary.instrument_id,
            client_order_id=spawn_id,
            order_side=primary.side,
            quantity=quantity,
            price=price,
            quote_quantity=primary.is_quote_quantity,
            init_id=UUID4(),
            ts_init=self._clock.timestamp_ns(),
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            exec_algorithm_id=self._exec_algorithm_id,
            exec_spawn_id=primary.client_order_id,
            **spawn_linkage_fields(primary),
        )

    def create_market(
        self,
        *,
        primary,
        state: OrderExecutionState,
        quantity: Quantity,
        time_in_force,
        reduce_only: bool,
    ) -> MarketOrder:
        spawn_id = self._next_spawn_id(state)
        return MarketOrder(
            trader_id=primary.trader_id,
            strategy_id=primary.strategy_id,
            instrument_id=primary.instrument_id,
            client_order_id=spawn_id,
            order_side=primary.side,
            quantity=quantity,
            time_in_force=time_in_force,
            quote_quantity=primary.is_quote_quantity,
            reduce_only=reduce_only,
            init_id=UUID4(),
            ts_init=self._clock.timestamp_ns(),
            exec_algorithm_id=self._exec_algorithm_id,
            exec_spawn_id=primary.client_order_id,
            **spawn_linkage_fields(primary),
        )

    def register_child(
        self,
        state: OrderExecutionState,
        order,
        kind: SpawnKind,
        *,
        reserved_quantity: Quantity | None = None,
    ) -> None:
        state.activate_order(
            client_order_id=order.client_order_id,
            kind=kind,
            reserved_quantity=order.quantity if reserved_quantity is None else reserved_quantity,
        )
        if kind == SpawnKind.LIMIT:
            state.limit_orders_submitted += 1

    @staticmethod
    def _next_spawn_id(state: OrderExecutionState) -> ClientOrderId:
        state.spawn_sequence += 1
        return ClientOrderId(f"{state.primary_order_id.value}E{state.spawn_sequence}")


class PrimaryMirror:
    """Maintains a primary order quantity mirror for exec_spawn bookkeeping."""

    def __init__(self, *, cache, clock, logger) -> None:
        self._cache = cache
        self._clock = clock
        self._log = logger

    def reduce_primary(self, primary, quantity: Quantity) -> Quantity:
        leaves_qty = primary.leaves_qty
        mirrorable = min(quantity, leaves_qty)
        if quantity > leaves_qty:
            delta = Quantity(float(quantity) - float(leaves_qty), quantity.precision)
            self._log.warning(
                "PostLimit mirror reduce above leaves, continuing best-effort: "
                f"primary={primary.client_order_id} "
                f"requested={quantity} "
                f"leaves={leaves_qty} "
                f"mirror_reserved={mirrorable} "
                f"delta={delta}"
            )

        if mirrorable <= Quantity.zero(mirrorable.precision):
            self._log.debug(
                "PostLimit mirror skipped reduce with non-positive mirrorable qty: "
                f"primary={primary.client_order_id} "
                f"requested={quantity} "
                f"leaves={leaves_qty}"
            )
            return Quantity.zero(quantity.precision)

        new_qty = Quantity(
            primary.quantity - mirrorable,
            primary.quantity.precision,
        )
        if new_qty <= Quantity.zero(new_qty.precision):
            self._log.debug(
                f"PostLimit mirror skipped quantity update at floor: {primary.client_order_id} "
                f"requested={quantity} "
                f"mirror_reserved={mirrorable} "
                f"current={primary.quantity}"
            )
            return Quantity.zero(quantity.precision)

        self._apply_quantity_update(primary, new_qty)
        return mirrorable

    def restore_primary(self, primary, child_order, reserved_quantity: Quantity | None) -> None:
        if reserved_quantity is None:
            return

        restore_qty = min(reserved_quantity, child_order.leaves_qty)
        if restore_qty <= Quantity.zero(restore_qty.precision):
            return

        restored_qty = Quantity(
            primary.quantity + restore_qty,
            primary.quantity.precision,
        )
        self._apply_quantity_update(primary, restored_qty)

    def _apply_quantity_update(self, primary, quantity: Quantity) -> None:
        ts_now = self._clock.timestamp_ns()
        updated = OrderUpdated(
            trader_id=primary.trader_id,
            strategy_id=primary.strategy_id,
            instrument_id=primary.instrument_id,
            client_order_id=primary.client_order_id,
            venue_order_id=primary.venue_order_id,
            account_id=primary.account_id,
            quantity=quantity,
            price=None,
            trigger_price=None,
            event_id=UUID4(),
            ts_event=ts_now,
            ts_init=ts_now,
        )
        primary.apply(updated)
        self._cache.update_order(primary)
        self._log.debug(
            f"PostLimit mirrored primary quantity: {primary.client_order_id} -> {quantity}",
        )
