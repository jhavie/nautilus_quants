# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorStrategy - Demonstrates CustomData flow for factor consumption.

This strategy subscribes to factor signals published by FactorEngineActor
and makes trading decisions based on factor values.

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus Signal subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


class FactorStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for FactorStrategy.

    Parameters
    ----------
    instrument_id : str
        The instrument to trade (e.g., "ETHUSDT.BINANCE").
    bar_type : str
        Bar type string for price updates.
    factor_signal_name : str
        Name of the factor signal to subscribe to (e.g., "factor.breakout").
    entry_threshold : float, default 1.0
        Factor value threshold for entry (signal == threshold triggers entry).
    order_amount : float, default 10000.0
        Order amount in quote currency (USDT).
    stop_loss_pct : float, default 0.02
        Stop loss percentage.
    """

    instrument_id: str
    bar_type: str
    factor_signal_name: str = "factor.alpha_breakout_long"
    entry_threshold: float = 1.0
    order_amount: float = 10000.0
    stop_loss_pct: float = 0.02


class FactorStrategy(Strategy):
    """
    Strategy that consumes factor signals via MessageBus.

    This strategy demonstrates the complete Constitution-compliant flow:
    1. FactorEngineActor computes factors from bars
    2. FactorEngineActor publishes signals via MessageBus
    3. This strategy subscribes to signals and trades on factor values

    The strategy is completely decoupled from factor computation logic.
    """

    def __init__(self, config: FactorStrategyConfig) -> None:
        """
        Initialize the FactorStrategy.

        Parameters
        ----------
        config : FactorStrategyConfig
            Strategy configuration.
        """
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.factor_signal_name = config.factor_signal_name

        # State
        self.instrument: Instrument | None = None
        self.entry_price: float | None = None
        self.position_side: str | None = None
        self._current_close: float | None = None
        self._bar_count: int = 0
        self._signal_count: int = 0

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)

        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.instrument_id}")
            self.stop()
            return

        # Subscribe to bars for price updates and stop loss checks
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}")

        # Subscribe to factor signal via MessageBus
        self.subscribe_signal(self.factor_signal_name)
        self.log.info(f"Subscribed to signal: {self.factor_signal_name}")

        self.log.info(
            f"FactorStrategy started: {self.instrument_id}, "
            f"factor={self.factor_signal_name}, threshold={self.config.entry_threshold}"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        self.log.info(
            f"FactorStrategy stopped: bars={self._bar_count}, signals={self._signal_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar updates for stop loss checks.

        Parameters
        ----------
        bar : Bar
            The received bar.
        """
        if bar.bar_type != self.bar_type:
            return

        self._bar_count += 1
        self._current_close = float(bar.close)

        # Check stop loss on every bar
        if self.position_side is not None and self.entry_price is not None:
            self._check_stop_loss(self._current_close)

    def on_signal(self, signal) -> None:
        """
        Handle factor signals from FactorEngineActor.

        This is the key Constitution-compliant pattern:
        - Factor computation is delegated to FactorEngineActor
        - Strategy only receives signals and makes trading decisions

        Parameters
        ----------
        signal : Signal
            The factor signal containing name, value, and timestamp.
        """
        self._signal_count += 1

        # Parse signal value (JSON format from FactorEngineActor)
        try:
            data = json.loads(signal.value)
            instrument = data.get("instrument", "")
            factor_value = data.get("value", 0.0)
        except (json.JSONDecodeError, TypeError):
            # Direct value format
            factor_value = float(signal.value) if signal.value else 0.0
            instrument = str(self.instrument_id)

        # Only process signals for our instrument
        if instrument and instrument != str(self.instrument_id):
            return

        # Log signal (every 10th to reduce noise)
        if self._signal_count % 10 == 0:
            self.log.info(
                f"Signal #{self._signal_count}: {signal.value} -> factor_value={factor_value}"
            )

        # Check for entry condition
        if self.position_side is None:
            if factor_value == self.config.entry_threshold:
                if self._current_close is not None:
                    self._open_long(self._current_close)

    def _check_stop_loss(self, close: float) -> bool:
        """
        Check stop loss condition.

        Parameters
        ----------
        close : float
            Current close price.

        Returns
        -------
        bool
            True if position was closed.
        """
        if self.entry_price is None:
            return False

        if self.position_side == "LONG":
            if close < self.entry_price * (1 - self.config.stop_loss_pct):
                self._close_position("STOP_LOSS")
                return True

        return False

    def _calculate_quantity(self, price: float) -> Decimal:
        """Calculate order quantity based on order amount and price."""
        raw_qty = self.config.order_amount / price
        return Decimal(str(raw_qty))

    def _open_long(self, price: float) -> None:
        """
        Open a long position.

        Parameters
        ----------
        price : float
            Entry price.
        """
        if self.instrument is None:
            return

        self.entry_price = price
        self.position_side = "LONG"

        raw_qty = self._calculate_quantity(price)
        qty = self.instrument.make_qty(raw_qty)

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
        )
        self.submit_order(order)

        self.log.info(
            f"OPEN LONG @ {price:.2f}, qty={qty}, amount={self.config.order_amount} USDT"
        )

    def _close_position(self, reason: str) -> None:
        """
        Close the current position.

        Parameters
        ----------
        reason : str
            Reason for closing.
        """
        side = self.position_side
        self.close_all_positions(self.instrument_id)
        self.log.info(f"CLOSE ({side}): {reason} @ entry {self.entry_price:.2f}")
        self.entry_price = None
        self.position_side = None
