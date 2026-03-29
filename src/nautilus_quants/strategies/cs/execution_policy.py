# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
ExecutionPolicy - Strategy pattern for order submission.

Provides a clean interface between strategy orchestration and order mechanics.
Implementations:
- MarketExecutionPolicy: direct market orders
- PostLimitExecutionPolicy: BBO-pegged limit orders via PostLimitExecAlgorithm
- BracketExecutionPolicyWrapper: decorator that upgrades entries to bracket orders
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from nautilus_trader.model.enums import ContingencyType, OrderSide, OrderType
from nautilus_trader.model.identifiers import ExecAlgorithmId, InstrumentId, PositionId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders.base import Order

if TYPE_CHECKING:
    from nautilus_trader.model.position import Position
    from nautilus_trader.trading.strategy import Strategy

    from nautilus_quants.strategies.cs.config import BracketConfig

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
        target_quote_quantity: float | None = None,
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

    def submit_reduce(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        tags: list[str] | None = None,
    ) -> None:
        """Submit a reduce-only order to partially reduce an existing position."""
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
        target_quote_quantity: float | None = None,
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

    def submit_reduce(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        tags: list[str] | None = None,
    ) -> None:
        order = self._strategy.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            reduce_only=True,
            tags=tags,
        )
        self._strategy.submit_order(order)


class PostLimitExecutionPolicy:
    """BBO-pegged limit order execution via PostLimitExecAlgorithm.

    Submits MarketOrders with exec_algorithm_id="PostLimit". The
    PostLimitExecAlgorithm intercepts and converts to BBO-pegged limit orders
    with chase and market fallback.

    When target_quote_quantity is provided, PostLimit recalculates remaining
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
        target_quote_quantity: float | None = None,
        contract_multiplier: float = 1.0,
    ) -> None:
        anchor_px = self._get_anchor_price(instrument_id, order_side)
        params: dict[str, str] = {}
        if anchor_px is not None:
            params["anchor_px"] = str(anchor_px)
        if target_quote_quantity is not None:
            params["target_quote_quantity"] = str(target_quote_quantity)
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

    def submit_reduce(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        tags: list[str] | None = None,
    ) -> None:
        anchor_px = self._get_anchor_price(instrument_id, order_side)
        params = {"anchor_px": str(anchor_px)} if anchor_px else None
        order = self._strategy.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            reduce_only=True,
            exec_algorithm_id=_POST_LIMIT_ID,
            exec_algorithm_params=params,
            tags=tags,
        )
        self._strategy.submit_order(order)

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


_ORDER_TYPE_MAP: dict[str, OrderType] = {
    "LIMIT": OrderType.LIMIT,
    "MARKET_IF_TOUCHED": OrderType.MARKET_IF_TOUCHED,
    "STOP_MARKET": OrderType.STOP_MARKET,
    "STOP_LIMIT": OrderType.STOP_LIMIT,
}


class BracketExecutionPolicyWrapper:
    """Decorator that upgrades entry orders to bracket orders with TP/SL.

    submit_open: replaces inner policy — creates bracket order via
        order_factory.bracket() + submit_order_list().
    submit_close: cancels active TP/SL orders, then delegates to inner policy.

    Always cancels existing contingent orders before submitting a new bracket.
    """

    def __init__(
        self,
        inner: ExecutionPolicy,
        strategy: Strategy,
        config: BracketConfig,
    ) -> None:
        self._inner = inner
        self._strategy = strategy
        self._config = config
        self._entry_algo_id: ExecAlgorithmId | None = (
            ExecAlgorithmId(config.entry_exec_algorithm_id)
            if config.entry_exec_algorithm_id
            else None
        )

    @property
    def _has_bracket(self) -> bool:
        return (
            self._config.take_profit_pct is not None
            or self._config.stop_loss_pct is not None
        )

    def submit_open(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        position_id: PositionId | None = None,
        quote_quantity: bool = False,
        tags: list[str] | None = None,
        target_quote_quantity: float | None = None,
        contract_multiplier: float = 1.0,
    ) -> None:
        self._cancel_contingent_orders(instrument_id)

        if not self._has_bracket:
            self._inner.submit_open(
                instrument_id=instrument_id,
                order_side=order_side,
                quantity=quantity,
                position_id=position_id,
                quote_quantity=quote_quantity,
                tags=tags,
                target_quote_quantity=target_quote_quantity,
                contract_multiplier=contract_multiplier,
            )
            return

        entry_price = self._get_entry_price(instrument_id, order_side)
        if entry_price is None:
            self._strategy.log.warning(
                f"BracketWrapper: no price for {instrument_id}, "
                "falling back to inner policy"
            )
            self._inner.submit_open(
                instrument_id=instrument_id,
                order_side=order_side,
                quantity=quantity,
                position_id=position_id,
                quote_quantity=quote_quantity,
                tags=tags,
                target_quote_quantity=target_quote_quantity,
                contract_multiplier=contract_multiplier,
            )
            return

        instrument = self._strategy.cache.instrument(instrument_id)
        if instrument is None:
            self._strategy.log.warning(
                f"BracketWrapper: instrument not cached: {instrument_id}"
            )
            return

        # Build bracket kwargs
        bracket_kwargs: dict = {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "quantity": quantity,
            "entry_tags": list(tags or []) + ["ENTRY"],
        }

        # Entry exec algorithm (e.g. PostLimit)
        if self._entry_algo_id is not None:
            bracket_kwargs["entry_exec_algorithm_id"] = self._entry_algo_id
            entry_params = self._build_entry_exec_params(
                instrument_id, order_side, target_quote_quantity,
                contract_multiplier,
            )
            if entry_params:
                bracket_kwargs["entry_exec_algorithm_params"] = entry_params

        # Take-profit
        if self._config.take_profit_pct is not None:
            tp_raw = self._compute_tp_price(entry_price, order_side)
            tp_price = instrument.make_price(tp_raw)
            tp_type = _ORDER_TYPE_MAP.get(
                self._config.tp_order_type, OrderType.LIMIT,
            )
            bracket_kwargs["tp_order_type"] = tp_type
            if tp_type == OrderType.LIMIT:
                bracket_kwargs["tp_price"] = tp_price
            else:
                bracket_kwargs["tp_trigger_price"] = tp_price
            bracket_kwargs["tp_tags"] = ["TAKE_PROFIT"]

        # Stop-loss
        if self._config.stop_loss_pct is not None:
            sl_raw = self._compute_sl_price(entry_price, order_side)
            sl_trigger_price = instrument.make_price(sl_raw)
            sl_type = _ORDER_TYPE_MAP.get(
                self._config.sl_order_type, OrderType.STOP_MARKET,
            )
            bracket_kwargs["sl_order_type"] = sl_type
            bracket_kwargs["sl_trigger_price"] = sl_trigger_price
            bracket_kwargs["sl_tags"] = ["STOP_LOSS"]

        bracket_list = self._strategy.order_factory.bracket(**bracket_kwargs)
        self._strategy.submit_order_list(bracket_list)
        self._strategy.log.info(
            f"BracketWrapper: submitted bracket {bracket_list.id} "
            f"for {instrument_id} {order_side.name} qty={quantity} "
            f"entry_price={entry_price}"
        )

    def submit_close(
        self,
        position: Position,
        tags: list[str] | None = None,
    ) -> None:
        self._cancel_contingent_orders(position.instrument_id)
        self._inner.submit_close(position, tags=tags)

    def submit_reduce(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        tags: list[str] | None = None,
    ) -> None:
        # TODO(bracket-reduce): After partial reduce, existing TP/SL orders still
        # reference the original entry quantity. In NETTING mode, if TP/SL triggers
        # with qty > remaining position, it will REVERSE the position (e.g. +700 → -300).
        # Current risk: zero — fixed position_mode never triggers resize.
        # Fix needed when weighted/equal_weight + bracket is used: cancel old TP/SL
        # and recreate with remaining_qty. Complex due to PostLimit chase/partial fills.
        self._inner.submit_reduce(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Price computation
    # ------------------------------------------------------------------

    def _compute_tp_price(self, entry_price: float, side: OrderSide) -> float:
        pct = self._config.take_profit_pct
        if side == OrderSide.BUY:
            return entry_price * (1.0 + pct)
        return entry_price * (1.0 - pct)

    def _compute_sl_price(self, entry_price: float, side: OrderSide) -> float:
        pct = self._config.stop_loss_pct
        if side == OrderSide.BUY:
            return entry_price * (1.0 - pct)
        return entry_price * (1.0 + pct)

    def _get_entry_price(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
    ) -> float | None:
        quote = self._strategy.cache.quote_tick(instrument_id)
        if quote is not None:
            if side == OrderSide.BUY:
                return float(quote.ask_price) if quote.ask_price else None
            return float(quote.bid_price) if quote.bid_price else None
        if hasattr(self._strategy, "_bar_type_to_inst_id"):
            inst_str = str(instrument_id)
            for bar_type, iid_str in self._strategy._bar_type_to_inst_id.items():
                if iid_str == inst_str:
                    bar = self._strategy.cache.bar(bar_type)
                    if bar is not None:
                        return float(bar.close)
        return None

    # ------------------------------------------------------------------
    # PostLimit exec algorithm params
    # ------------------------------------------------------------------

    def _build_entry_exec_params(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        target_quote_quantity: float | None,
        contract_multiplier: float,
    ) -> dict[str, str] | None:
        params: dict[str, str] = {}
        anchor_px = self._get_entry_price(instrument_id, side)
        if anchor_px is not None:
            params["anchor_px"] = str(anchor_px)
        if target_quote_quantity is not None:
            params["target_quote_quantity"] = str(target_quote_quantity)
            params["contract_multiplier"] = str(contract_multiplier)
        return params or None

    # ------------------------------------------------------------------
    # Contingent order cleanup
    # ------------------------------------------------------------------

    def _cancel_contingent_orders(self, instrument_id: InstrumentId) -> None:
        """Cancel active TP/SL orders for an instrument before close or flip."""
        for order in self._strategy.cache.orders_open(
            instrument_id=instrument_id,
            strategy_id=self._strategy.id,
        ):
            if order.contingency_type in (
                ContingencyType.OCO,
                ContingencyType.OUO,
            ) and not order.is_closed:
                self._strategy.cancel_order(order)
                self._strategy.log.debug(
                    f"BracketWrapper: cancelled contingent order "
                    f"{order.client_order_id} for {instrument_id}"
                )
