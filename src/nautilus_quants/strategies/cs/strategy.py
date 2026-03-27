# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
CSStrategy - Lean orchestrator for cross-sectional factor strategies.

Receives RebalanceOrders from DecisionEngineActor and submits orders
via ExecutionPolicy. NETTING mode: flips are one-shot orders whose
quantity is computed from cache position + target at execution time.

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
    - Receives decisions, does not compute signals (Principle V)
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.orders.base import Order
from nautilus_trader.model.events.position import (
    PositionChanged,
    PositionClosed,
    PositionOpened,
)
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.strategies.cs.config import CSStrategyConfig
from nautilus_quants.strategies.cs.execution_policy import (
    BracketExecutionPolicyWrapper,
    ExecutionPolicy,
    MarketExecutionPolicy,
    PostLimitExecutionPolicy,
)
from nautilus_quants.strategies.cs.types import RebalanceOrders

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


_EXECUTION_POLICIES: dict[str, type] = {
    "MarketExecutionPolicy": MarketExecutionPolicy,
    "PostLimitExecutionPolicy": PostLimitExecutionPolicy,
}


class CSStrategy(BarSubscriptionMixin, Strategy):
    """
    Cross-sectional strategy orchestrator (NETTING mode).

    Data flow:
        DecisionEngineActor → RebalanceOrders (via MessageBus)
        → CSStrategy.on_data() → _execute_order() for each order
        → ExecutionPolicy.submit_open/close
    """

    def __init__(self, config: CSStrategyConfig) -> None:
        super().__init__(config)
        # Auto-claim instruments so reconciled positions use this strategy's ID
        # instead of EXTERNAL after restart. (cdef readonly list — mutate, not reassign)
        if not self.external_order_claims:
            self.external_order_claims.extend(
                InstrumentId.from_str(iid) for iid in config.instrument_ids
            )
        self._execution_policy: ExecutionPolicy = self._create_execution_policy(config)
        self._instruments: dict[InstrumentId, Instrument] = {}

    def _create_execution_policy(self, config: CSStrategyConfig) -> ExecutionPolicy:
        policy_cls = _EXECUTION_POLICIES.get(config.execution_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown execution_policy: {config.execution_policy}. "
                f"Available: {list(_EXECUTION_POLICIES)}"
            )
        inner = policy_cls(self)

        if config.bracket is not None:
            bracket_cfg = config.bracket
            if (
                config.execution_policy == "PostLimitExecutionPolicy"
                and not bracket_cfg.entry_exec_algorithm_id
            ):
                bracket_cfg = bracket_cfg.__class__(
                    take_profit_pct=bracket_cfg.take_profit_pct,
                    stop_loss_pct=bracket_cfg.stop_loss_pct,
                    tp_order_type=bracket_cfg.tp_order_type,
                    sl_order_type=bracket_cfg.sl_order_type,
                    entry_exec_algorithm_id="PostLimit",
                )
            return BracketExecutionPolicyWrapper(inner, self, bracket_cfg)

        return inner

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Cache instruments and subscribe to data feeds."""
        for iid_str in self.config.instrument_ids:
            iid = InstrumentId.from_str(iid_str)
            instrument = self.cache.instrument(iid)
            if instrument is None:
                self.log.warning(f"Instrument not found: {iid}")
            else:
                self._instruments[iid] = instrument

        if not self._instruments:
            self.log.error("No instruments found in cache")
            self.stop()
            return

        self.log.info(f"Cached {len(self._instruments)} instruments")

        if self.config.bar_types:
            self._subscribe_bar_types(self.config.bar_types)

        self.subscribe_data(DataType(RebalanceOrders), client_id=ClientId(self.id.value))
        self.log.info(
            f"CSStrategy started: execution_policy={self.config.execution_policy}"
        )

    def on_stop(self) -> None:
        """Cancel in-flight orders, then close all positions with market orders.

        Must cancel first to prevent double-fills when PostLimit has active
        child orders. close_position() sends direct market orders that bypass
        the ExecAlgorithm entirely.
        """
        # 1. Cancel all open/in-flight orders (cleans up PostLimit state machine)
        for order in self.cache.orders_open(strategy_id=self.id):
            if not order.is_closed:
                self.cancel_order(order)
        for order in self.cache.orders_inflight(strategy_id=self.id):
            if not order.is_closed:
                self.cancel_order(order)

        # 2. Close all positions with direct market orders
        for position in self.cache.positions_open(strategy_id=self.id):
            if not position.is_closed:
                self.close_position(position, tags=["STRATEGY_STOP"])

        self.log.info("CSStrategy stopped")
    # Data handling
    # -------------------------------------------------------------------------

    def on_data(self, data: object) -> None:
        """Receive RebalanceOrders from DecisionEngineActor."""
        if not isinstance(data, RebalanceOrders):
            return

        all_orders = data.orders
        self.log.info(f"Received RebalanceOrders: {len(all_orders)} orders")

        for order_dict in all_orders:
            self._execute_rebalance(order_dict)

    # -------------------------------------------------------------------------
    # Nautilus position hooks — logging only (NETTING mode)
    # -------------------------------------------------------------------------

    def on_position_opened(self, event: PositionOpened) -> None:
        """Log new position opened."""
        self.log.info(
            f"Position opened: {event.instrument_id} "
            f"side={event.side} qty={event.quantity}"
        )

    def on_position_changed(self, event: PositionChanged) -> None:
        """Log position direction change (flip confirmation in NETTING mode)."""
        self.log.info(
            f"Position changed: {event.instrument_id} "
            f"side={event.side} qty={event.quantity}"
        )

    def on_position_closed(self, event: PositionClosed) -> None:
        """Log position closed (delisting protection or strategy stop)."""
        self.log.info(f"Position closed: {event.instrument_id}")

    # -------------------------------------------------------------------------
    # Order execution — unified rebalance
    # -------------------------------------------------------------------------

    def _execute_rebalance(self, order_dict: dict[str, Any]) -> None:
        """Execute a rebalance order by computing delta from cache state.

        Derives the operation (open/close/flip/resize) from the difference
        between the target position value and the current position in cache.
        """
        inst_id = InstrumentId.from_str(order_dict["instrument_id"])
        instrument = self._instruments.get(inst_id)
        if instrument is None:
            self.log.warning(f"Skip rebalance: instrument not cached: {inst_id}")
            return

        target_value = float(order_dict.get("target_quote_quantity", 0))
        target_side = (
            OrderSide.BUY if order_dict["order_side"] == "BUY" else OrderSide.SELL
        )

        # Read current position from cache
        position = None
        for p in self.cache.positions_open(
            instrument_id=inst_id, strategy_id=self.id,
        ):
            position = p
            break

        # --- No current position ---
        if position is None or position.is_closed:
            if target_value <= 0:
                return
            self._submit_new_position(inst_id, instrument, target_side, target_value)
            return

        # --- Target = 0 → close ---
        if target_value <= 0:
            tags = order_dict.get("tags")
            self._execution_policy.submit_close(position, tags=tags)
            self.log.debug(f"CLOSE {inst_id} tags={tags}")
            return

        current_is_long = position.side == PositionSide.LONG
        target_is_long = target_side == OrderSide.BUY

        exec_price = self._get_exec_price(inst_id, target_side)
        if exec_price is None or exec_price <= 0:
            self.log.warning(f"Skip rebalance {inst_id}: no price available")
            return

        multiplier = float(instrument.multiplier)
        current_value = float(position.quantity) * exec_price * multiplier
        tags = order_dict.get("tags")

        # --- Direction flip → one-shot NETTING flip ---
        if current_is_long != target_is_long:
            target_qty = Decimal(str(target_value / (exec_price * multiplier)))
            flip_qty = instrument.make_qty(Decimal(str(position.quantity)) + target_qty)
            flip_notional = float(flip_qty) * exec_price * multiplier
            self._execution_policy.submit_open(
                instrument_id=inst_id,
                order_side=target_side,
                quantity=flip_qty,
                tags=tags,
                target_quote_quantity=flip_notional,
                contract_multiplier=multiplier,
            )
            self.log.info(
                f"FLIP {target_side.name} {inst_id}: "
                f"current={position.quantity} + target_qty "
                f"= {flip_qty} (price={exec_price})"
            )
            return

        # --- Same direction → resize (skip if delta too small) ---
        delta_value = target_value - current_value
        if abs(delta_value) / max(current_value, 1) < self.config.min_rebalance_pct:
            return

        delta_qty = instrument.make_qty(
            Decimal(str(abs(delta_value) / (exec_price * multiplier)))
        )
        if delta_value > 0:
            self._execution_policy.submit_open(
                instrument_id=inst_id,
                order_side=target_side,
                quantity=delta_qty,
                tags=tags,
                target_quote_quantity=abs(delta_value),
                contract_multiplier=multiplier,
            )
            self.log.debug(f"RESIZE_UP {inst_id}: +{delta_qty}")
        else:
            reduce_side = Order.closing_side(position.side)
            self._execution_policy.submit_open(
                instrument_id=inst_id,
                order_side=reduce_side,
                quantity=delta_qty,
                tags=tags,
                target_quote_quantity=abs(delta_value),
                contract_multiplier=multiplier,
            )
            self.log.debug(f"RESIZE_DOWN {inst_id}: -{delta_qty}")

    def _submit_new_position(
        self,
        inst_id: InstrumentId,
        instrument: Instrument,
        side: OrderSide,
        target_value: float,
    ) -> None:
        """Submit an order to open a new position."""
        exec_price = self._get_exec_price(inst_id, side)
        if exec_price is None or exec_price <= 0:
            self.log.warning(f"Skip OPEN {inst_id}: no price available")
            return
        multiplier = float(instrument.multiplier)
        raw_qty = Decimal(str(target_value / (exec_price * multiplier)))
        quantity = instrument.make_qty(raw_qty)
        self._execution_policy.submit_open(
            instrument_id=inst_id,
            order_side=side,
            quantity=quantity,
            target_quote_quantity=target_value,
            contract_multiplier=multiplier,
        )
        self.log.debug(
            f"OPEN {side.name} {inst_id}: target={target_value} "
            f"price={exec_price} qty={quantity}"
        )

    def _get_exec_price(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
    ) -> float | None:
        """Get execution price: QuoteTick BBO > bar close > None."""
        quote = self.cache.quote_tick(instrument_id)
        if quote is not None:
            if side == OrderSide.BUY:
                return float(quote.ask_price) if quote.ask_price else None
            return float(quote.bid_price) if quote.bid_price else None
        return self._get_last_close(instrument_id)

    def _get_last_close(self, instrument_id: InstrumentId) -> float | None:
        """Get the last bar close price for an instrument."""
        for bar_type, inst_id_str in self._bar_type_to_inst_id.items():
            if inst_id_str == str(instrument_id):
                bar = self.cache.bar(bar_type)
                if bar is not None:
                    return float(bar.close)
        return None

    def _close_all_positions(self) -> None:
        """Close all open positions on strategy stop."""
        for position in sorted(
            self.cache.positions_open(strategy_id=self.id),
            key=lambda p: str(p.instrument_id),
        ):
            if position.is_closed:
                continue
            self._execution_policy.submit_close(position, tags=["STRATEGY_STOP"])
