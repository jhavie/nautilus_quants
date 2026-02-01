# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FactorStrategy - Consumes FactorValues via MessageBus for trading decisions.

This strategy subscribes to factor signals published by FactorEngineActor
and makes trading decisions based on factor values.

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
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
    signal_name : str
        Name of the factor to trade on (e.g., "alpha_breakout_long").
    entry_threshold : float, default 1.0
        Factor value threshold for entry (signal == threshold triggers entry).
    stop_loss_pct : float, default 0.02
        Stop loss percentage (0.02 = 2%).
    order_amount : float, default 10000.0
        Order amount in quote currency (USDT).
    enable_long : bool, default True
        Enable long positions.
    enable_short : bool, default False
        Enable short positions.
    bar_type : str, default ""
        Bar type string for stop loss checks (injected by CLI from data config).
        If empty, will try to get from cache (legacy behavior).
    sma_period : int, default 200
        SMA period for trend exit (exit when price below SMA for N bars).
    exit_bars_below_sma : int, default 5
        Number of consecutive bars below SMA to trigger exit.
    """

    instrument_id: str
    signal_name: str
    entry_threshold: float = 1.0
    stop_loss_pct: float = 0.02
    order_amount: float = 10000.0
    enable_long: bool = True
    enable_short: bool = False
    bar_type: str = ""
    sma_period: int = 200
    exit_bars_below_sma: int = 5


class FactorStrategy(Strategy):
    """
    Strategy that consumes factor signals via CustomData.

    This strategy demonstrates the complete Constitution-compliant flow:
    1. FactorEngineActor computes factors from bars
    2. FactorEngineActor publishes FactorValues as CustomData
    3. This strategy subscribes to CustomData and trades on factor values

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
        self.signal_name = config.signal_name
        self._bar_type: BarType | None = None

        # State
        self.instrument: Instrument | None = None
        self.entry_price: float | None = None
        self.position_side: str | None = None
        self._current_close: float | None = None
        self._bar_count: int = 0
        self._signal_count: int = 0
        
        # History for trend exit (store signal bar closes)
        self._signal_closes: list[float] = []
        self._current_sma: float | None = None

        # Funding rate tracking
        self._last_funding_rate: Decimal | None = None
        self._funding_rate_count: int = 0

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)

        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.instrument_id}")
            self.stop()
            return

        # Subscribe to source bars (1m) for stop loss checks
        # bar_type must be injected by CLI from data config
        if not self.config.bar_type:
            self.log.error(
                "bar_type not configured. Ensure backtest is run via CLI which injects "
                "bar_type from data config automatically."
            )
            self.stop()
            return
        
        self._bar_type = BarType.from_str(self.config.bar_type)
        self.subscribe_bars(self._bar_type)
        self.log.info(f"Subscribed to bars: {self._bar_type}")

        # Subscribe to FactorValues Data (now a proper Nautilus Data subclass)
        from nautilus_trader.model.data import DataType
        from nautilus_quants.factors.types import FactorValues

        self.subscribe_data(DataType(FactorValues))
        self.log.info("Subscribed to FactorValues Data")

        # Subscribe to funding rate updates
        self.subscribe_funding_rates(self.instrument_id)
        self.log.info(f"Subscribed to funding rates: {self.instrument_id}")

        self.log.info(
            f"FactorStrategy started: {self.instrument_id}, "
            f"signal={self.signal_name}, threshold={self.config.entry_threshold}, "
            f"enable_long={self.config.enable_long}, enable_short={self.config.enable_short}"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        # Close any open positions
        if self.position_side is not None:
            self._close_position("STRATEGY_STOP")

        self.log.info(
            f"FactorStrategy stopped: bars={self._bar_count}, signals={self._signal_count}, "
            f"funding_rates={self._funding_rate_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar updates for stop loss checks.

        Parameters
        ----------
        bar : Bar
            The received bar.
        """
        if self._bar_type and bar.bar_type != self._bar_type:
            return

        self._bar_count += 1
        self._current_close = float(bar.close)

        # Check stop loss on every bar
        if self.position_side is not None and self.entry_price is not None:
            self._check_stop_loss(self._current_close)

    def on_data(self, data) -> None:
        """
        Handle FactorValues from FactorEngineActor.

        This is the key Constitution-compliant pattern:
        - Factor computation is delegated to FactorEngineActor
        - Strategy only receives Data and makes trading decisions

        Parameters
        ----------
        data : FactorValues
            The FactorValues Data object (now a proper Nautilus Data subclass).
        """
        from nautilus_quants.factors.types import FactorValues

        # FactorValues is now a Data subclass, no need to parse from CustomData
        if not isinstance(data, FactorValues):
            return

        self._signal_count += 1

        # Get target factor value for our instrument
        factor_value = data.get(self.signal_name, str(self.instrument_id))

        if factor_value is None:
            return

        # Update signal bar close history (for SMA calculation and trend exit)
        if self._current_close is not None:
            self._signal_closes.append(self._current_close)
            # Keep only necessary history
            max_history = self.config.sma_period + 10
            if len(self._signal_closes) > max_history:
                self._signal_closes = self._signal_closes[-max_history:]

        # Calculate SMA for trend exit
        if len(self._signal_closes) >= self.config.sma_period:
            self._current_sma = sum(self._signal_closes[-self.config.sma_period:]) / self.config.sma_period

        # Log signal (every 100th to reduce noise)
        if self._signal_count % 100 == 0:
            self.log.info(f"Signal #{self._signal_count}: factor_value={factor_value}, sma={self._current_sma}")

        # Check for entry/exit conditions
        if self.position_side is None and self._current_close is not None:
            # Entry: check factor signal
            if factor_value == self.config.entry_threshold:
                if self.config.enable_long:
                    self._open_long(self._current_close)
            elif factor_value == -self.config.entry_threshold:
                if self.config.enable_short:
                    self._open_short(self._current_close)
        else:
            # Exit: check trend reversal (SMA exit)
            if self._current_sma is not None:
                self._check_trend_exit()

    def on_position_opened(self, event) -> None:
        """
        Handle position opened event.

        Parameters
        ----------
        event : PositionOpened
            The position opened event.
        """
        self.log.info(f"Position opened: {event.side} @ {event.avg_px_open}")

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        """Handle funding rate updates.

        Parameters
        ----------
        funding_rate : FundingRateUpdate
            The funding rate update event.
        """
        self._funding_rate_count += 1
        self._last_funding_rate = funding_rate.rate

        self.log.info(
            f"FUNDING_RATE: {funding_rate.instrument_id} "
            f"rate={funding_rate.rate:.8f} "
            f"ts={funding_rate.ts_event}"
        )

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

        stop_loss_pct = self.config.stop_loss_pct

        if self.position_side == "LONG":
            if close < self.entry_price * (1 - stop_loss_pct):
                self._close_position("STOP_LOSS")
                return True
        elif self.position_side == "SHORT":
            if close > self.entry_price * (1 + stop_loss_pct):
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

    def _open_short(self, price: float) -> None:
        """
        Open a short position.

        Parameters
        ----------
        price : float
            Entry price.
        """
        if self.instrument is None:
            return

        if not self.config.enable_short:
            return

        self.entry_price = price
        self.position_side = "SHORT"

        raw_qty = self._calculate_quantity(price)
        qty = self.instrument.make_qty(raw_qty)

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
        )
        self.submit_order(order)

        self.log.info(
            f"OPEN SHORT @ {price:.2f}, qty={qty}, amount={self.config.order_amount} USDT"
        )

    def _check_trend_exit(self) -> None:
        """
        Check trend reversal exit conditions (SMA exit).
        
        Exit long when last N bars are all below SMA.
        Exit short when last N bars are all above SMA.
        """
        if self._current_sma is None:
            return
        
        exit_bars = self.config.exit_bars_below_sma
        
        if len(self._signal_closes) < exit_bars:
            return
        
        recent_closes = self._signal_closes[-exit_bars:]
        sma = self._current_sma
        
        if self.position_side == "LONG":
            # Exit long if all recent closes are below SMA
            if all(c < sma for c in recent_closes):
                self._close_position("SMA_EXIT")
        elif self.position_side == "SHORT":
            # Exit short if all recent closes are above SMA
            if all(c > sma for c in recent_closes):
                self._close_position("SMA_EXIT")

    def _close_position(self, reason: str) -> None:
        """
        Close the current position.

        Parameters
        ----------
        reason : str
            Reason for closing.
        """
        # Guard against re-entry (prevent recursive calls)
        if self.position_side is None:
            return
            
        side = self.position_side
        entry = self.entry_price
        
        # Clear state BEFORE calling close_all_positions to prevent re-entry
        self.entry_price = None
        self.position_side = None
        
        # Now safe to close
        self.close_all_positions(self.instrument_id)
        entry_str = f"{entry:.2f}" if entry else "N/A"
        self.log.info(f"CLOSE ({side}): {reason} @ entry {entry_str}")
