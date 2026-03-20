# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
CSStrategy - Lean orchestrator for cross-sectional factor strategies.

Receives RebalanceOrders from DecisionEngineActor, delegates execution
ordering to ExposureManager, and submits orders via ExecutionPolicy.
Nautilus position hooks (on_position_closed/opened) drive sequential execution.

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
    - Receives decisions, does not compute signals (Principle V)
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events.position import PositionClosed, PositionOpened
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.strategies.cs.config import CSStrategyConfig
from nautilus_quants.strategies.cs.execution_policy import (
    ExecutionPolicy,
    MarketExecutionPolicy,
)
from nautilus_quants.strategies.cs.exposure_manager import ExposureManager, ExposurePolicy
from nautilus_quants.strategies.cs.types import RebalanceOrders

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


class CSStrategy(BarSubscriptionMixin, Strategy):
    """
    Cross-sectional strategy orchestrator.

    Data flow:
        DecisionEngineActor → RebalanceOrders (via MessageBus)
        → CSStrategy.on_data() → ExposureManager.submit_plan()
        → execute closes → on_position_closed() → release opens
    """

    def __init__(self, config: CSStrategyConfig) -> None:
        super().__init__(config)
        exposure_policy = ExposurePolicy(config.exposure_policy)
        self._exposure_manager = ExposureManager(policy=exposure_policy)
        self._execution_policy: ExecutionPolicy = self._create_execution_policy(config)
        self._instruments: dict[InstrumentId, Instrument] = {}

    def _create_execution_policy(self, config: CSStrategyConfig) -> ExecutionPolicy:
        if config.execution_mode == "market":
            return MarketExecutionPolicy(self)
        raise ValueError(f"Unknown execution_mode: {config.execution_mode}")

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
            f"CSStrategy started: execution_mode={self.config.execution_mode}, "
            f"exposure_policy={self.config.exposure_policy}"
        )

    def on_stop(self) -> None:
        """Close all positions and clean up."""
        self._exposure_manager.on_stop()
        self._close_all_positions()
        self.log.info("CSStrategy stopped")

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def on_data(self, data: object) -> None:
        """Receive RebalanceOrders from DecisionEngineActor."""
        if not isinstance(data, RebalanceOrders):
            return

        closes = data.closes
        opens = data.opens

        self.log.info(
            f"Received RebalanceOrders: {len(closes)} closes, {len(opens)} opens"
        )

        immediate_closes, immediate_opens = self._exposure_manager.submit_plan(
            closes, opens,
        )

        for order_dict in immediate_closes:
            self._execute_close(order_dict["instrument_id"], order_dict.get("tags"))
        for order_dict in immediate_opens:
            self._execute_open(order_dict)

        if self._exposure_manager.has_pending:
            self.log.info(f"ExposureManager: {self._exposure_manager.state_summary}")

    # -------------------------------------------------------------------------
    # Nautilus position hooks — drive sequential execution
    # -------------------------------------------------------------------------

    def on_position_closed(self, event: PositionClosed) -> None:
        """Position closed confirmed → release queued orders from ExposureManager."""
        inst_id = str(event.instrument_id)
        released = self._exposure_manager.on_close_filled(inst_id)
        for order_dict in released:
            if order_dict["action"] == "OPEN":
                self._execute_open(order_dict)
            else:
                self._execute_close(order_dict["instrument_id"], order_dict.get("tags"))

    def on_position_opened(self, event: PositionOpened) -> None:
        """Position opened confirmed → release queued orders from ExposureManager."""
        inst_id = str(event.instrument_id)
        released = self._exposure_manager.on_open_filled(inst_id)
        for order_dict in released:
            if order_dict["action"] == "CLOSE":
                self._execute_close(order_dict["instrument_id"], order_dict.get("tags"))
            else:
                self._execute_open(order_dict)

    # -------------------------------------------------------------------------
    # Order execution (via ExecutionPolicy)
    # -------------------------------------------------------------------------

    def _execute_open(self, order_dict: dict[str, Any]) -> None:
        """Convert a RebalanceOrders open instruction into an actual order."""
        inst_id = InstrumentId.from_str(order_dict["instrument_id"])
        instrument = self._instruments.get(inst_id)
        if instrument is None:
            self.log.warning(f"Skip OPEN: instrument not cached: {inst_id}")
            return

        quote_qty = order_dict.get("quote_quantity", 0)
        if quote_qty <= 0:
            self.log.warning(f"Skip OPEN {inst_id}: quote_quantity={quote_qty}")
            return

        # Get latest price from bar data for quantity calculation
        exec_price = self._get_last_close(inst_id)
        if exec_price is None or exec_price <= 0:
            self.log.warning(f"Skip OPEN {inst_id}: no price available")
            return

        side = OrderSide.BUY if order_dict["order_side"] == "BUY" else OrderSide.SELL
        multiplier = float(instrument.multiplier)
        raw_qty = Decimal(str(quote_qty / (exec_price * multiplier)))
        quantity = instrument.make_qty(raw_qty)
        tags = order_dict.get("tags")

        self._execution_policy.submit_open(
            instrument_id=inst_id,
            order_side=side,
            quantity=quantity,
            tags=tags,
        )
        self.log.debug(
            f"OPEN {side.name} {inst_id}: value={quote_qty} price={exec_price} qty={quantity}"
        )

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
