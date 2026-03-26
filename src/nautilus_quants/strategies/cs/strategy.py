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
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events.position import (
    PositionChanged,
    PositionClosed,
    PositionOpened,
)
from nautilus_trader.model.identifiers import InstrumentId
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

        self.subscribe_data(DataType(RebalanceOrders))
        self.log.info(
            f"CSStrategy started: execution_policy={self.config.execution_policy}"
        )

    def on_stop(self) -> None:
        """Close all positions and clean up."""
        self._close_all_positions()
        self.log.info("CSStrategy stopped")

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def on_data(self, data: object) -> None:
        """Receive RebalanceOrders from DecisionEngineActor."""
        if not isinstance(data, RebalanceOrders):
            return

        all_orders = data.orders
        self.log.info(
            f"Received RebalanceOrders: {len(data.closes)} closes, "
            f"{len(data.opens)} opens, {len(data.flips)} flips"
        )

        for order_dict in all_orders:
            self._execute_order(order_dict)

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
    # Order execution dispatch
    # -------------------------------------------------------------------------

    def _execute_order(self, order_dict: dict[str, Any]) -> None:
        """Route order to appropriate execution method based on decision intent."""
        intent = order_dict["intent"]
        if intent == "CLOSE":
            self._execute_close(order_dict["instrument_id"], order_dict.get("tags"))
        elif intent == "OPEN":
            self._execute_open(order_dict)
        elif intent == "FLIP":
            self._execute_flip(order_dict)
        else:
            self.log.error(f"Unknown intent: {intent}")

    def _execute_flip(self, order_dict: dict[str, Any]) -> None:
        """Execute one-shot NETTING flip: actual_position_qty + target_qty.

        1. Read actual position quantity from cache (reflects PnL/ADL)
        2. Calculate target quantity from quote_tick BBO price
        3. Flip quantity = actual + target
        """
        inst_id = InstrumentId.from_str(order_dict["instrument_id"])
        instrument = self._instruments.get(inst_id)
        if instrument is None:
            self.log.warning(f"Skip FLIP: instrument not cached: {inst_id}")
            return

        side = OrderSide.BUY if order_dict["order_side"] == "BUY" else OrderSide.SELL

        # 1. Read actual position quantity from cache
        current_qty = Decimal(0)
        for position in self.cache.positions_open(
            instrument_id=inst_id, strategy_id=self.id,
        ):
            current_qty = Decimal(str(position.quantity))
            break

        if current_qty <= 0:
            self.log.warning(
                f"FLIP {inst_id}: no open position found, falling back to OPEN"
            )
            self._execute_open(order_dict)
            return

        # 2. Calculate target quantity from BBO price
        exec_price = self._get_exec_price(inst_id, side)
        if exec_price is None or exec_price <= 0:
            self.log.warning(f"Skip FLIP {inst_id}: no price available")
            return

        target_quote_quantity = float(order_dict["target_quote_quantity"])
        multiplier = float(instrument.multiplier)
        target_qty = Decimal(str(target_quote_quantity / (exec_price * multiplier)))

        # 3. Flip quantity = actual position + target
        total_qty = instrument.make_qty(current_qty + target_qty)
        total_target_quote_quantity = float(total_qty) * exec_price * multiplier
        tags = order_dict.get("tags")

        self._execution_policy.submit_open(
            instrument_id=inst_id,
            order_side=side,
            quantity=total_qty,
            tags=tags,
            target_quote_quantity=total_target_quote_quantity,
            contract_multiplier=multiplier,
            intent="FLIP",
        )
        self.log.info(
            f"FLIP {side.name} {inst_id}: current={current_qty} + "
            f"target={target_qty} = {total_qty} (price={exec_price}, "
            f"target_quote_quantity={total_target_quote_quantity:.6f})"
        )

    def _execute_open(self, order_dict: dict[str, Any]) -> None:
        """Convert a RebalanceOrders open instruction into an actual order."""
        inst_id = InstrumentId.from_str(order_dict["instrument_id"])
        instrument = self._instruments.get(inst_id)
        if instrument is None:
            self.log.warning(f"Skip OPEN: instrument not cached: {inst_id}")
            return

        target_quote_quantity = float(order_dict.get("target_quote_quantity", 0))
        if target_quote_quantity <= 0:
            self.log.warning(
                f"Skip OPEN {inst_id}: target_quote_quantity={target_quote_quantity}"
            )
            return

        side = OrderSide.BUY if order_dict["order_side"] == "BUY" else OrderSide.SELL

        # Use BBO price for quantity calculation at execution time
        exec_price = self._get_exec_price(inst_id, side)
        if exec_price is None or exec_price <= 0:
            self.log.warning(f"Skip OPEN {inst_id}: no price available")
            return

        multiplier = float(instrument.multiplier)
        raw_qty = Decimal(str(target_quote_quantity / (exec_price * multiplier)))
        quantity = instrument.make_qty(raw_qty)
        tags = order_dict.get("tags")

        self._execution_policy.submit_open(
            instrument_id=inst_id,
            order_side=side,
            quantity=quantity,
            tags=tags,
            target_quote_quantity=target_quote_quantity,
            contract_multiplier=multiplier,
            intent="OPEN",
        )
        self.log.debug(
            "OPEN "
            f"{side.name} {inst_id}: target_quote_quantity={target_quote_quantity} "
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

    def _execute_close(
        self,
        instrument_id_str: str,
        tags: list[str] | None = None,
    ) -> None:
        """Close all open positions for an instrument."""
        inst_id = InstrumentId.from_str(instrument_id_str)
        for position in self.cache.positions_open(
            instrument_id=inst_id, strategy_id=self.id,
        ):
            if position.is_closed:
                continue
            self._execution_policy.submit_close(position, tags=tags)
        self.log.debug(f"CLOSE {inst_id} tags={tags}")

    def _close_all_positions(self) -> None:
        """Close all open positions on strategy stop."""
        for position in sorted(
            self.cache.positions_open(strategy_id=self.id),
            key=lambda p: str(p.instrument_id),
        ):
            if position.is_closed:
                continue
            self._execution_policy.submit_close(position, tags=["STRATEGY_STOP"])
