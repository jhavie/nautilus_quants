"""Price-Volume Breakout Strategy with Multi-Timeframe Support.

Entry conditions (Long):
1. Current close breaks above highest close of last N bars
2. Current volume breaks above highest volume of last N bars
3. Current close is above SMA

Exit conditions:
1. Stop loss: checked every bar (fast risk path)
2. Trend reversal: N consecutive bars close below SMA

Multi-Timeframe Support:
- Uses NautilusTrader native bar aggregation
- `interval` parameter controls signal timeframe (e.g., "1h", "4h", "1d")
- Stop loss is checked on every 1-min bar (fast path)
- Signals are checked on aggregated bars (slow path)
"""
from decimal import Decimal

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.strategies.breakout.signal import PriceVolumeBreakoutSignal
from nautilus_quants.strategies.utils import create_aggregated_bar_type


class PriceVolumeBreakoutStrategyConfig(StrategyConfig, frozen=True):
    """Price-volume breakout strategy configuration.

    Supports two initialization modes:
    1. Direct: instrument_id + bar_type (for manual instantiation)
    2. CLI: instruments + bar_spec + exchange (from backtest.yaml)
    """

    # CLI mode params (from backtest.yaml)
    instruments: tuple[str, ...] = ()
    bar_spec: str = "1m"
    exchange: str = "BINANCE"

    # Direct mode params (optional overrides)
    instrument_id: str = ""
    bar_type: str = ""

    # Strategy params
    sma_period: int = 200
    breakout_period: int = 30
    stop_loss_pct: float = 0.02
    exit_bars_below_sma: int = 5
    order_amount: float = 10000.0
    enable_long: bool = True
    enable_short: bool = True
    interval: str = "1h"

    def get_instrument_id(self) -> str:
        """Get instrument ID (supports both CLI and direct modes)."""
        if self.instrument_id:
            return self.instrument_id
        if self.instruments:
            return f"{self.instruments[0]}.{self.exchange}"
        raise ValueError("No instrument configured")

    def get_bar_type(self) -> str:
        """Get bar type string (supports both CLI and direct modes)."""
        if self.bar_type:
            return self.bar_type
        inst_id = self.get_instrument_id()
        return f"{inst_id}-1-MINUTE-LAST-EXTERNAL"


class PriceVolumeBreakoutStrategy(Strategy):
    """Price-volume breakout strategy with multi-timeframe support."""

    def __init__(self, config: PriceVolumeBreakoutStrategyConfig):
        super().__init__(config)

        # Support both CLI mode (instruments/bar_spec) and direct mode (instrument_id/bar_type)
        self.instrument_id = InstrumentId.from_str(config.get_instrument_id())

        # Base 1-minute bar type (for stop loss checks)
        self.bar_type_base = BarType.from_str(config.get_bar_type())

        # Signal bar type (aggregated if interval > 1m)
        self._use_aggregation = config.interval.lower() != "1m"
        if self._use_aggregation:
            bar_type_str, self._subscribe_str = create_aggregated_bar_type(
                str(self.instrument_id),
                config.interval,
                "1-MINUTE-EXTERNAL",
            )
            self.bar_type_signal = BarType.from_str(bar_type_str)
        else:
            self.bar_type_signal = self.bar_type_base
            self._subscribe_str = None

        # Data storage (only stores signal-timeframe data)
        self.closes: list[float] = []
        self.volumes: list[float] = []

        # Position tracking
        self.entry_price: float | None = None
        self.position_side: str | None = None  # "LONG" or "SHORT"
        self.current_sma: float | None = None
        self._current_close: float | None = None  # Latest close for stop loss

        # Signal calculator
        self.signal = PriceVolumeBreakoutSignal(
            breakout_period=config.breakout_period,
            sma_period=config.sma_period,
            exit_bars_below_sma=config.exit_bars_below_sma,
        )

        self.instrument = None
        self._signal_bar_count = 0

    def on_start(self):
        """Strategy startup."""
        self.instrument = self.cache.instrument(self.instrument_id)

        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.instrument_id}")
            self.stop()
            return

        # Subscribe to base 1-min bars (for stop loss)
        self.subscribe_bars(self.bar_type_base)
        self.log.info(f"Subscribed to base bars: {self.bar_type_base}")

        # Subscribe to aggregated signal bars if using multi-timeframe
        if self._use_aggregation and self._subscribe_str:
            subscribe_bar_type = BarType.from_str(self._subscribe_str)
            self.subscribe_bars(subscribe_bar_type)
            self.log.info(f"Subscribed to signal bars: {self._subscribe_str}")
            self.log.info(f"Interval: {self.config.interval}")

        self.log.info(f"Strategy started: {self.instrument_id}")
        self.log.info(f"Long: {'enabled' if self.config.enable_long else 'disabled'}")
        self.log.info(f"Short: {'enabled' if self.config.enable_short else 'disabled'}")

    def on_bar(self, bar: Bar):
        """Handle bar data with dual-path processing."""
        close = float(bar.close)

        # Route bar to appropriate handler based on bar type
        if bar.bar_type == self.bar_type_base:
            # Fast risk path: every 1-min bar
            self._handle_base_bar(bar, close)
        elif bar.bar_type == self.bar_type_signal:
            # Slow signal path: aggregated bars
            self._handle_signal_bar(bar, close)
        else:
            # Single-timeframe mode: bar_type_base == bar_type_signal
            if not self._use_aggregation:
                self._handle_base_bar(bar, close)
                self._handle_signal_bar(bar, close)

    def _handle_base_bar(self, bar: Bar, close: float):
        """Handle base (1-min) bar - fast risk path."""
        self._current_close = close

        # Check stop loss on every bar (fast response)
        if self.position_side is not None and self.entry_price is not None:
            if self._check_stop_loss(close):
                return  # Position closed, skip further processing

    def _handle_signal_bar(self, bar: Bar, close: float):
        """Handle signal bar - slow signal path."""
        volume = float(bar.volume)
        self._signal_bar_count += 1

        # Update history FIRST to include current bar (matching TradingView behavior)
        self.closes.append(close)
        self.volumes.append(volume)

        # Warmup: need enough data for SMA
        if len(self.closes) < self.config.sma_period:
            if len(self.closes) == self.config.sma_period:
                self.current_sma = float(np.mean(self.closes[-self.config.sma_period:]))
            if len(self.closes) % 10 == 0:
                self.log.info(
                    f"Warmup: {len(self.closes)}/{self.config.sma_period} signal bars"
                )
            return

        # Calculate SMA (including current bar, matching TradingView)
        sma = float(np.mean(self.closes[-self.config.sma_period:]))
        self.current_sma = sma

        # Check positions
        positions = [
            p
            for p in self.cache.positions_open()
            if p.instrument_id == self.instrument_id
        ]
        has_position = len(positions) > 0

        # Debug logging (every 10 signal bars)
        if self._signal_bar_count % 10 == 0:
            highest = max(self.closes[-self.config.breakout_period - 1:-1])  # Exclude current
            highest_vol = max(self.volumes[-self.config.breakout_period - 1:-1])
            self.log.info(
                f"Signal bar #{self._signal_bar_count}: close={close:.1f}, vol={volume:.0f}, "
                f"SMA={sma:.1f}, highest_close={highest:.1f}, highest_vol={highest_vol:.0f}"
            )

        if not has_position:
            # Check entry signals (using history EXCLUDING current bar)
            # Entry condition: current bar breaks above highest of PREVIOUS N bars
            if self.config.enable_long:
                if self.signal.check_long(close, volume, self.closes[:-1], self.volumes[:-1], sma):
                    self._open_long(close, volume)

            if self.config.enable_short:
                if self.signal.check_short(close, volume, self.closes[:-1], self.volumes[:-1], sma):
                    self._open_short(close, volume)
        else:
            # Check exit signals (using history INCLUDING current bar)
            # Exit condition: last N bars (including current) all below SMA
            self._check_trend_exit(close, sma)

    def _check_stop_loss(self, close: float) -> bool:
        """Check stop loss condition. Returns True if position was closed."""
        if self.entry_price is None:
            return False

        if self.position_side == "LONG":
            if close < self.entry_price * (1 - self.config.stop_loss_pct):
                self._close_position("STOP_LOSS")
                return True
        elif self.position_side == "SHORT":
            if close > self.entry_price * (1 + self.config.stop_loss_pct):
                self._close_position("STOP_LOSS")
                return True
        return False

    def _check_trend_exit(self, close: float, sma: float):
        """Check trend reversal exit conditions."""
        if self.position_side == "LONG":
            if self.signal.check_exit_long(
                self.closes, sma, self.config.exit_bars_below_sma
            ):
                self._close_position("SMA_EXIT")
        elif self.position_side == "SHORT":
            if self.signal.check_exit_short(
                self.closes, sma, self.config.exit_bars_below_sma
            ):
                self._close_position("SMA_EXIT")

    def _calculate_quantity(self, price: float) -> Decimal:
        """Calculate order quantity based on fixed order amount and current price."""
        raw_qty = self.config.order_amount / price
        return Decimal(str(raw_qty))

    def _open_long(self, price: float, volume: float):
        """Open long position."""
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
        self.log.info(f"OPEN LONG @ {price:.2f}, qty={qty}, amount={self.config.order_amount} USDT")

    def _open_short(self, price: float, volume: float):
        """Open short position."""
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
        self.log.info(f"OPEN SHORT @ {price:.2f}, qty={qty}, amount={self.config.order_amount} USDT")

    def _close_position(self, reason: str):
        """Close position."""
        side = self.position_side
        self.close_all_positions(self.instrument_id)
        self.log.info(f"CLOSE ({side}): {reason} @ entry {self.entry_price:.2f}")
        self.entry_price = None
        self.position_side = None

    def on_stop(self):
        """Strategy shutdown."""
        self.log.info(f"Strategy stopped: {self.instrument_id}")
        self.log.info(f"Total signal bars processed: {self._signal_bar_count}")
        self.log.info(f"Interval: {self.config.interval}")
