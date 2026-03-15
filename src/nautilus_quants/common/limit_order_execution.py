# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
LimitOrderExecutionMixin - Limit order execution via PostLimitExecAlgorithm.

Parallel to AnchorPriceExecutionMixin.  Strategies inherit both mixins and
select execution mode via ``config.execution_mode``.

Order submission is identical (MarketOrder), but with ``exec_algorithm_id="PostLimit"``
so that the PostLimitExecAlgorithm intercepts and converts to BBO-pegged limit orders
with chase and market fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nautilus_trader.model.identifiers import ExecAlgorithmId, InstrumentId
from nautilus_trader.model.orders.base import Order

if TYPE_CHECKING:
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import PositionId
    from nautilus_trader.model.objects import Quantity

_POST_LIMIT_ID = ExecAlgorithmId("PostLimit")


class LimitOrderExecutionMixin:
    """Limit order execution via PostLimitExecAlgorithm.

    Assumes the host class provides ``self.cache``, ``self.order_factory``,
    ``self.submit_order``, ``self.log``, and ``self.id`` — all supplied by
    the Nautilus ``Strategy`` base class.
    """

    def _submit_limit_open(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        exec_price: float,
        position_id: PositionId | None = None,
        exec_params_override: dict[str, str] | None = None,
    ) -> Order:
        """Submit a MarketOrder routed through PostLimitExecAlgorithm (open)."""
        params: dict[str, str] = {"anchor_px": str(exec_price)}
        if exec_params_override:
            params.update(exec_params_override)

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=quantity,
            exec_algorithm_id=_POST_LIMIT_ID,
            exec_algorithm_params=params,
        )
        self.submit_order(order, position_id=position_id)
        return order

    def _submit_limit_close(
        self,
        position,
        exec_price: float | None = None,
        exec_params_override: dict[str, str] | None = None,
    ) -> None:
        """Submit a reduce_only MarketOrder routed through PostLimitExecAlgorithm (close)."""
        params: dict[str, str] = {}
        if exec_price is not None and exec_price > 0:
            params["anchor_px"] = str(exec_price)
        if exec_params_override:
            params.update(exec_params_override)

        order = self.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=Order.closing_side(position.side),
            quantity=position.quantity,
            reduce_only=True,
            exec_algorithm_id=_POST_LIMIT_ID,
            exec_algorithm_params=params or None,
        )
        self.submit_order(order, position_id=position.id)

    def _close_instrument_positions_limit(
        self,
        instrument_id: InstrumentId,
        exec_price: float | None = None,
    ) -> None:
        """Close all open positions for *instrument_id* via PostLimitExecAlgorithm."""
        for position in self.cache.positions_open(
            instrument_id=instrument_id, strategy_id=self.id
        ):
            if position.is_closed:
                continue
            self._submit_limit_close(position, exec_price)
