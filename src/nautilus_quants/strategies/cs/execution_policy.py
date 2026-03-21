# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
ExecutionPolicy - Strategy pattern for order submission.

Provides a clean interface between strategy orchestration and order mechanics.
Implementations:
- MarketExecutionPolicy: direct market orders
- PostLimitExecutionPolicy: BBO-pegged limit orders via PostLimitExecAlgorithm
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ExecAlgorithmId, InstrumentId, PositionId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders.base import Order

if TYPE_CHECKING:
    from nautilus_trader.model.position import Position
    from nautilus_trader.trading.strategy import Strategy

_POST_LIMIT_ID = ExecAlgorithmId("PostLimit")


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
        target_quote_value: float | None = None,
        contract_multiplier: float = 1.0,
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
        target_quote_value: float | None = None,
        contract_multiplier: float = 1.0,
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


class PostLimitExecutionPolicy:
    """BBO-pegged limit order execution via PostLimitExecAlgorithm.

    Submits MarketOrders with exec_algorithm_id="PostLimit". The
    PostLimitExecAlgorithm intercepts and converts to BBO-pegged limit orders
    with chase and market fallback.

    When target_quote_value is provided, PostLimit recalculates remaining
    quantity from remaining USDT value / BBO price on each chase iteration,
    eliminating price drift.

    QuoteTick subscription is handled automatically by the algorithm on first
    order (algorithm.py on_order → subscribe_quote_ticks).
    """

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
        target_quote_value: float | None = None,
        contract_multiplier: float = 1.0,
    ) -> None:
        anchor_px = self._get_anchor_price(instrument_id, order_side)
        params: dict[str, str] = {}
        if anchor_px is not None:
            params["anchor_px"] = str(anchor_px)
        if target_quote_value is not None:
            params["target_quote_value"] = str(target_quote_value)
            params["contract_multiplier"] = str(contract_multiplier)

        order = self._strategy.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            exec_algorithm_id=_POST_LIMIT_ID,
            exec_algorithm_params=params or None,
            tags=tags,
        )
        self._strategy.submit_order(order, position_id=position_id)

    def submit_close(
        self,
        position: Position,
        tags: list[str] | None = None,
    ) -> None:
        anchor_px = self._get_anchor_price(
            position.instrument_id,
            Order.closing_side(position.side),
        )
        params = {"anchor_px": str(anchor_px)} if anchor_px else None
        order = self._strategy.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=Order.closing_side(position.side),
            quantity=position.quantity,
            reduce_only=True,
            exec_algorithm_id=_POST_LIMIT_ID,
            exec_algorithm_params=params,
            tags=tags,
        )
        self._strategy.submit_order(order, position_id=position.id)

    def _get_anchor_price(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
    ) -> float | None:
        """Get fallback price: QuoteTick BBO > bar close > None."""
        quote = self._strategy.cache.quote_tick(instrument_id)
        if quote is not None:
            if side == OrderSide.BUY:
                return float(quote.ask_price) if quote.ask_price else None
            return float(quote.bid_price) if quote.bid_price else None
        # Fallback: latest bar close
        if hasattr(self._strategy, "_bar_type_to_inst_id"):
            inst_str = str(instrument_id)
            for bar_type, iid_str in self._strategy._bar_type_to_inst_id.items():
                if iid_str == inst_str:
                    bar = self._strategy.cache.bar(bar_type)
                    if bar is not None:
                        return float(bar.close)
        return None
