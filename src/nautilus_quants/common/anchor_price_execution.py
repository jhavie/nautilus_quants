# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
AnchorPriceExecutionMixin - Shared anchor_px deterministic fill order submission.

Provides helper methods for submitting MarketOrders with an ``anchor_px``
exec_algorithm_param so the SignalCloseFillModel can fill at deterministic
signal-time prices.

All three strategy families (FMZ, CrossSectional, WorldQuant) use the same
anchor_px semantics; this mixin eliminates the duplicated order-submission
boilerplate while leaving strategy-specific lifecycle and signal logic intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.orders.base import Order

if TYPE_CHECKING:
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import PositionId
    from nautilus_trader.model.objects import Quantity


class AnchorPriceExecutionMixin:
    """Shared anchor_px deterministic fill order submission.

    Assumes the host class provides ``self.cache``, ``self.order_factory``,
    ``self.submit_order``, ``self.log``, and ``self.id`` — all supplied by
    the Nautilus ``Strategy`` base class.
    """

    @staticmethod
    def _anchor_params(price: float | None) -> dict[str, str] | None:
        """Build ``exec_algorithm_params``; return *None* for invalid prices."""
        if price is not None and price > 0:
            return {"anchor_px": str(price)}
        return None

    def _submit_anchor_open(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        exec_price: float,
        position_id: PositionId | None = None,
    ) -> Order:
        """Submit a MarketOrder with ``anchor_px`` for opening a position."""
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=quantity,
            exec_algorithm_params={"anchor_px": str(exec_price)},
        )
        self.submit_order(order, position_id=position_id)
        return order

    def _submit_anchor_close(
        self,
        position,
        exec_price: float | None = None,
    ) -> None:
        """Submit a ``reduce_only`` MarketOrder with ``anchor_px`` for closing."""
        params = self._anchor_params(exec_price)
        if params is None:
            self.log.warning(
                f"CLOSE {position.instrument_id}: no anchor_px, using default matching"
            )
        order = self.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=Order.closing_side(position.side),
            quantity=position.quantity,
            reduce_only=True,
            exec_algorithm_params=params,
        )
        self.submit_order(order, position_id=position.id)

    def _close_instrument_positions(
        self,
        instrument_id: InstrumentId,
        exec_price: float | None = None,
    ) -> None:
        """Close all open positions for *instrument_id* using ``anchor_px``."""
        for position in self.cache.positions_open(
            instrument_id=instrument_id, strategy_id=self.id
        ):
            if position.is_closed:
                continue
            self._submit_anchor_close(position, exec_price)
