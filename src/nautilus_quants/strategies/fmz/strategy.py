# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
FMZFactorStrategy - Exact replication of FMZ multi-factor selection strategy.

This strategy implements the logic from: https://www.fmz.com/digest-topic/9647

Key differences from CrossSectionalFactorStrategy:
    - No buffer zone: positions are closed immediately when rank changes
    - Simpler logic: direct long bottom N, short top N
    - Fixed position value per instrument

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
    - Receives pre-computed signals, does not compute them (Principle V)
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.factors.types import FactorValues

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


class FMZFactorStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for FMZFactorStrategy.

    Exact replication of FMZ article parameters.

    Parameters
    ----------
    instrument_ids : list[str]
        List of instrument IDs to trade (e.g., ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"]).
    n_long : int, default 40
        Number of instruments to long (lowest composite factor values).
    n_short : int, default 40
        Number of instruments to short (highest composite factor values).
    position_value : float, default 300.0
        Fixed position value per instrument in quote currency (USDT).
    rebalance_period : int, default 1
        Rebalance every N hours.
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    bar_types : list[str], default []
        Bar type strings for data subscription (injected by CLI).
    """

    instrument_ids: list[str]
    n_long: int = 40
    n_short: int = 40
    position_value: float = 300.0
    rebalance_period: int = 1
    composite_factor: str = "composite"
    bar_types: list[str] = []


class FMZFactorStrategy(Strategy):
    """
    FMZ Multi-Factor Selection Strategy.

    Exact replication of FMZ article logic:
    1. Subscribe to all instrument bars
    2. Receive composite factor values from FactorEngineActor
    3. Every N hours, rebalance based on factor ranking
    4. Long bottom N instruments (lowest composite value)
    5. Short top N instruments (highest composite value)

    No buffer zone logic - positions are closed immediately when rank changes.
    """

    def __init__(self, config: FMZFactorStrategyConfig) -> None:
        """Initialize the FMZFactorStrategy."""
        super().__init__(config)

        self._instrument_ids: list[InstrumentId] = [
            InstrumentId.from_str(iid) for iid in config.instrument_ids
        ]
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._bar_types: list[BarType] = []
        self._n_instruments = len(config.instrument_ids)

        # State tracking
        self._current_prices: dict[str, float] = {}
        self._long_positions: set[str] = set()
        self._short_positions: set[str] = set()
        self._hour_count: int = 0
        self._bar_count: int = 0

        # Accumulate composite factor values for current hour
        self._composite_values: dict[str, float] = {}
        # Track current batch timestamp for synchronization
        self._current_batch_ts: int = 0

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self.log.info(
            f"Starting FMZFactorStrategy with {len(self._instrument_ids)} instruments"
        )

        # Cache instruments
        for instrument_id in self._instrument_ids:
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                self.log.warning(f"Instrument not found: {instrument_id}")
            else:
                self._instruments[instrument_id] = instrument

        if not self._instruments:
            self.log.error("No instruments found in cache")
            self.stop()
            return

        self.log.info(f"Found {len(self._instruments)} instruments in cache")

        # Subscribe to bars
        if not self.config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically."
            )
            self.stop()
            return

        for bar_type_str in self.config.bar_types:
            bar_type = BarType.from_str(bar_type_str)
            self._bar_types.append(bar_type)
            self.subscribe_bars(bar_type)
            self.log.debug(f"Subscribed to bars: {bar_type}")

        # Subscribe to factor data
        self.subscribe_data(DataType(FactorValues))
        self.log.info("Subscribed to FactorValues Data")

        self.log.info(
            f"Strategy started: n_long={self.config.n_long}, n_short={self.config.n_short}, "
            f"position_value={self.config.position_value}, "
            f"rebalance_period={self.config.rebalance_period}h"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        self._close_all_positions("STRATEGY_STOP")
        self.log.info(
            f"FMZFactorStrategy stopped: bars={self._bar_count}, hours={self._hour_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates - track current prices."""
        if bar.bar_type not in self._bar_types:
            return

        self._bar_count += 1
        instrument_id = str(bar.bar_type.instrument_id)
        self._current_prices[instrument_id] = float(bar.close)

    def on_data(self, data) -> None:
        """
        Handle FactorValues from FactorEngineActor.

        Uses timestamp-based synchronization:
        - When timestamp changes, process previous batch
        - Accumulate values for current timestamp
        """
        if not isinstance(data, FactorValues):
            return

        ts_ns = data.ts_event
        factors = data.factors
        composite_factor = self.config.composite_factor

        # Timestamp changed - process previous batch
        if self._current_batch_ts > 0 and ts_ns > self._current_batch_ts:
            self._process_batch()

        # Update current batch timestamp
        self._current_batch_ts = ts_ns

        # Extract composite factor values and accumulate
        if composite_factor in factors:
            for instrument_id, value in factors[composite_factor].items():
                self._composite_values[instrument_id] = value

    def _process_batch(self) -> None:
        """
        Process accumulated factor values for the previous timestamp.

        Called when timestamp changes to process the complete batch.
        """
        if not self._composite_values:
            self.log.debug(f"_process_batch: no composite values")
            return

        self._hour_count += 1

        # Get timestamp from batch
        ts_dt = datetime.fromtimestamp(self._current_batch_ts / 1e9, tz=timezone.utc)
        current_hour = ts_dt.hour

        # Only rebalance at specific hours (0, 8, 16 for period=8)
        if current_hour % self.config.rebalance_period != 0:
            self._composite_values = {}
            return

        # Use composite values directly - NaN filtering is handled by CsFactorEngine
        composite = self._composite_values

        if len(composite) < self.config.n_long + self.config.n_short:
            self.log.warning(
                f"Not enough valid instruments: {len(composite)} < "
                f"{self.config.n_long + self.config.n_short}"
            )
            self._composite_values = {}
            return

        # Debug: Log composite factor rankings on first rebalance
        if len(self._long_positions) == 0 and len(self._short_positions) == 0:
            sorted_composite = sorted(composite.items(), key=lambda x: x[1])
            self.log.info(f"FIRST REBALANCE at {ts_dt} - Composite factor rankings:")
            self.log.info(f"  Bottom 5 (LONG): {[(k, f'{v:.4f}') for k, v in sorted_composite[:5]]}")
            self.log.info(f"  Top 5 (SHORT): {[(k, f'{v:.4f}') for k, v in sorted_composite[-5:]]}")

        # Execute rebalance
        self._rebalance(composite)
        self._composite_values = {}

    def _rebalance(self, composite: dict[str, float]) -> None:
        """
        Execute rebalance - FMZ logic adapted for HEDGING mode.

        FMZ original code:
            buy_symbols = row.sort_values().dropna()[0:N].index
            sell_symbols = row.sort_values().dropna()[-N:].index

            for symbol in symbols:
                if symbol in buy_symbols and e.account[symbol]['amount'] <= 0:
                    e.Buy(symbol, prices[symbol], value / prices[symbol] - e.account[symbol]['amount'])
                if symbol in sell_symbols and e.account[symbol]['amount'] >= 0:
                    e.Sell(symbol, prices[symbol], value / prices[symbol] + e.account[symbol]['amount'])

        Key insight: FMZ only opens position if NOT already in that direction.
        FMZ DOES NOT close positions that fall out of top/bottom N.
        Positions are only closed when they FLIP direction.

        Enhancement: Close positions for instruments missing from factor data (e.g., delisted).

        In HEDGING mode, we need 2 trades to flip (close + open).
        """
        # Sort by factor value
        sorted_symbols = sorted(composite.items(), key=lambda x: x[1])

        # Determine long/short targets
        long_targets = set([s for s, _ in sorted_symbols[:self.config.n_long]])
        short_targets = set([s for s, _ in sorted_symbols[-self.config.n_short:]])

        # === Close positions for instruments missing from factor data ===
        # This handles delisted instruments or data gaps
        instruments_with_data = set(composite.keys())

        # Check long positions for missing data
        missing_long = self._long_positions - instruments_with_data
        for inst_id in missing_long:
            self.log.warning(f"Closing LONG {inst_id}: no factor data (possible delisting)")
            self._close_position(inst_id, "NO_FACTOR_DATA")
            self._long_positions.discard(inst_id)

        # Check short positions for missing data
        missing_short = self._short_positions - instruments_with_data
        for inst_id in missing_short:
            self.log.warning(f"Closing SHORT {inst_id}: no factor data (possible delisting)")
            self._close_position(inst_id, "NO_FACTOR_DATA")
            self._short_positions.discard(inst_id)

        if self._hour_count <= 10 or self._hour_count % 24 == 0:
            self.log.info(
                f"Rebalance #{self._hour_count}: "
                f"long={len(long_targets)}, short={len(short_targets)}, "
                f"current_long={len(self._long_positions)}, current_short={len(self._short_positions)}"
            )

        # === FMZ Logic adapted for HEDGING mode ===
        for inst_id in composite.keys():
            is_long_target = inst_id in long_targets
            is_short_target = inst_id in short_targets
            currently_long = inst_id in self._long_positions
            currently_short = inst_id in self._short_positions

            # FMZ: if in buy_symbols AND amount <= 0, then Buy
            if is_long_target and not currently_long:
                # If currently short, close first then open long
                if currently_short:
                    self._close_position(inst_id, "FLIP_TO_LONG")
                    self._short_positions.discard(inst_id)
                if self._open_position(inst_id, OrderSide.BUY):
                    self._long_positions.add(inst_id)

            # FMZ: if in sell_symbols AND amount >= 0, then Sell
            elif is_short_target and not currently_short:
                # If currently long, close first then open short
                if currently_long:
                    self._close_position(inst_id, "FLIP_TO_SHORT")
                    self._long_positions.discard(inst_id)
                if self._open_position(inst_id, OrderSide.SELL):
                    self._short_positions.add(inst_id)

            # FMZ CRITICAL: Do NOT close positions that are not in target
            # Positions are only closed when they FLIP direction.

    def _open_position(self, instrument_id_str: str, side: OrderSide) -> bool:
        """Open position with fixed value."""
        instrument_id = InstrumentId.from_str(instrument_id_str)
        instrument = self._instruments.get(instrument_id)

        if instrument is None:
            return False

        price = self._current_prices.get(instrument_id_str)
        if price is None or price <= 0:
            return False

        raw_qty = Decimal(str(self.config.position_value / price))
        qty = instrument.make_qty(raw_qty)

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)

        return True

    def _close_position(self, instrument_id_str: str, reason: str) -> None:
        """Close a position."""
        instrument_id = InstrumentId.from_str(instrument_id_str)
        self.close_all_positions(instrument_id)
        self.log.debug(f"CLOSE {instrument_id_str}: {reason}")

    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        for inst_id in list(self._long_positions):
            self._close_position(inst_id, reason)
        for inst_id in list(self._short_positions):
            self._close_position(inst_id, reason)

        self._long_positions.clear()
        self._short_positions.clear()
