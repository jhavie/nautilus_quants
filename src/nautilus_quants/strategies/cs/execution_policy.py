# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
ExecutionPolicy - Strategy pattern for order submission.

Provides a clean interface between strategy orchestration and order mechanics.
Current implementation: MarketExecutionPolicy (direct market orders).
Future: AnchorExecutionPolicy, PostLimitExecutionPolicy (configuration-driven).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders.base import Order

if TYPE_CHECKING:
    from nautilus_trader.model.position import Position
    from nautilus_trader.trading.strategy import Strategy


class ExecutionPolicy(Protocol):
    """Order submission interface. Strategy delegates all order creation here."""

    def submit_open(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        position_id: PositionId | None = None,
        quote_quantity: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        """Submit an opening order."""
        ...

    def submit_close(
        self,
        position: Position,
        tags: list[str] | None = None,
    ) -> None:
        """Submit a closing order for an existing position."""
        ...


class MarketExecutionPolicy:
    """Direct market order execution."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def submit_open(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        position_id: PositionId | None = None,
        quote_quantity: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        order = self._strategy.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            quote_quantity=quote_quantity,
            tags=tags,
        )
        self._strategy.submit_order(order, position_id=position_id)

    def submit_close(
        self,
        position: Position,
        tags: list[str] | None = None,
    ) -> None:
        order = self._strategy.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=Order.closing_side(position.side),
            quantity=position.quantity,
            reduce_only=True,
            tags=tags,
        )
        self._strategy.submit_order(order, position_id=position.id)
