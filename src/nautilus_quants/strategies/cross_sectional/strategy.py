# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
CrossSectionalFactorStrategy - Multi-factor cross-sectional selection strategy.

This strategy implements a market-neutral approach:
- Long the N instruments with lowest composite factor values
- Short the N instruments with highest composite factor values
- Rebalance periodically (e.g., every 4 hours)

Based on: https://www.fmz.com/digest-topic/9647

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.factors.operators.cross_sectional import cs_zscore
from nautilus_quants.factors.types import FactorValues

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


class CrossSectionalFactorStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for CrossSectionalFactorStrategy.

    Parameters
    ----------
    instrument_ids : list[str]
        List of instrument IDs to trade (e.g., ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"]).
    n_positions : int, default 10
        Number of instruments to hold on each side (long and short).
    position_value : float, default 300.0
        Position value per instrument in quote currency (USDT).
    rebalance_bars : int, default 1
        Rebalance every N bars (1 = every bar).
    factor_weights : dict[str, float], default {}
        Weights for factor combination. If empty, uses default weights.
    bar_types : list[str], default []
        Bar type strings for data subscription (injected by CLI).
    enable_long : bool, default True
        Enable long positions.
    enable_short : bool, default True
        Enable short positions.
    """

    instrument_ids: list[str]
    n_positions: int = 10
    position_value: float = 300.0
    rebalance_bars: int = 1
    factor_weights: dict[str, float] = {}
    bar_types: list[str] = []
    enable_long: bool = True
    enable_short: bool = True


class CrossSectionalFactorStrategy(Strategy):
    """
    Cross-sectional multi-factor selection strategy.

    This strategy:
    1. Receives factor values from FactorEngineActor via CustomData
    2. Normalizes factors cross-sectionally using z-score
    3. Computes composite factor as weighted sum
    4. Longs bottom N instruments (lowest composite value)
    5. Shorts top N instruments (highest composite value)
    6. Rebalances positions periodically

    The strategy is market-neutral with equal long/short exposure.
    """

    def __init__(self, config: CrossSectionalFactorStrategyConfig) -> None:
        """
        Initialize the CrossSectionalFactorStrategy.

        Parameters
        ----------
        config : CrossSectionalFactorStrategyConfig
            Strategy configuration.
        """
        super().__init__(config)

        # Parse instrument IDs
        self._instrument_ids: list[InstrumentId] = [
            InstrumentId.from_str(iid) for iid in config.instrument_ids
        ]
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._bar_types: list[BarType] = []

        # Default factor weights if not specified
        self._factor_weights: dict[str, float] = config.factor_weights or {
            "volume": 0.6,
            "momentum": 0.4,
            "vol_change": 0.2,
            "volatility": 0.3,
            "corr": 0.4,
        }

        # State tracking
        self._current_prices: dict[str, float] = {}
        self._long_positions: set[str] = set()  # instrument_id strings
        self._short_positions: set[str] = set()
        self._bar_count: int = 0
        self._signal_count: int = 0

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self.log.info(
            f"Starting CrossSectionalFactorStrategy with {len(self._instrument_ids)} instruments"
        )

        # Get instruments from cache
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

        # Subscribe to bar types (injected by CLI)
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

        # Subscribe to FactorValues Data
        self.subscribe_data(DataType(FactorValues))
        self.log.info("Subscribed to FactorValues Data")

        self.log.info(
            f"Strategy started: n_positions={self.config.n_positions}, "
            f"position_value={self.config.position_value}, "
            f"rebalance_bars={self.config.rebalance_bars}"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        # Close all positions
        self._close_all_positions("STRATEGY_STOP")

        self.log.info(
            f"CrossSectionalFactorStrategy stopped: "
            f"bars={self._bar_count}, signals={self._signal_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar updates - track current prices.

        Parameters
        ----------
        bar : Bar
            The received bar.
        """
        if bar.bar_type not in self._bar_types:
            return

        self._bar_count += 1
        instrument_id = str(bar.bar_type.instrument_id)
        self._current_prices[instrument_id] = float(bar.close)

    def on_data(self, data) -> None:
        """
        Handle FactorValues from FactorEngineActor.

        This is where the cross-sectional logic is implemented:
        1. Normalize each factor using cs_zscore
        2. Compute weighted composite factor
        3. Rank instruments and select long/short lists
        4. Execute rebalancing trades

        Parameters
        ----------
        data : FactorValues
            The FactorValues Data object.
        """
        if not isinstance(data, FactorValues):
            return

        self._signal_count += 1

        # Only rebalance on schedule
        if self._signal_count % self.config.rebalance_bars != 0:
            return

        # Compute composite factor
        composite = self._compute_composite_factor(data)

        if len(composite) < 2 * self.config.n_positions:
            self.log.warning(
                f"Not enough instruments with valid factor values: "
                f"{len(composite)} < {2 * self.config.n_positions}"
            )
            return

        # Sort by composite value (ascending)
        sorted_instruments = sorted(composite.items(), key=lambda x: x[1])

        # Select long list (lowest values) and short list (highest values)
        n = self.config.n_positions
        long_list = {x[0] for x in sorted_instruments[:n]}
        short_list = {x[0] for x in sorted_instruments[-n:]}

        # Log rebalance info
        if self._signal_count % 10 == 0:
            self.log.info(
                f"Rebalance #{self._signal_count}: "
                f"long={list(long_list)[:3]}..., short={list(short_list)[:3]}..."
            )

        # Execute rebalancing
        self._rebalance_positions(long_list, short_list)

    def _compute_composite_factor(
        self, factor_values: FactorValues
    ) -> dict[str, float]:
        """
        Compute composite factor from individual factor values.

        Steps:
        1. Extract each factor's cross-sectional values
        2. Normalize using cs_zscore
        3. Compute weighted sum

        Parameters
        ----------
        factor_values : FactorValues
            Raw factor values from FactorEngineActor.

        Returns
        -------
        dict[str, float]
            Composite factor values keyed by instrument_id.
        """
        normalized_factors: dict[str, dict[str, float]] = {}

        # Normalize each factor
        for factor_name, weight in self._factor_weights.items():
            raw_values = factor_values.get_factor(factor_name)

            if not raw_values:
                continue

            # Filter out NaN values
            valid_values = {
                k: v for k, v in raw_values.items() 
                if not math.isnan(v)
            }

            if len(valid_values) < 3:
                continue

            # Cross-sectional z-score normalization
            normalized = cs_zscore(valid_values)
            normalized_factors[factor_name] = normalized

        if not normalized_factors:
            return {}

        # Compute weighted composite
        composite: dict[str, float] = {}
        all_instruments = set()

        for factor_dict in normalized_factors.values():
            all_instruments.update(factor_dict.keys())

        for instrument_id in all_instruments:
            total_weight = 0.0
            weighted_sum = 0.0

            for factor_name, factor_dict in normalized_factors.items():
                if instrument_id in factor_dict:
                    value = factor_dict[instrument_id]
                    if not math.isnan(value):
                        weight = self._factor_weights.get(factor_name, 0.0)
                        weighted_sum += weight * value
                        total_weight += weight

            if total_weight > 0:
                composite[instrument_id] = weighted_sum / total_weight

        return composite

    def _rebalance_positions(
        self, long_list: set[str], short_list: set[str]
    ) -> None:
        """
        Rebalance positions to match target long/short lists.

        Logic:
        - Close longs that are no longer in long_list or are now in short_list
        - Close shorts that are no longer in short_list or are now in long_list
        - Open new longs for instruments in long_list
        - Open new shorts for instruments in short_list

        Parameters
        ----------
        long_list : set[str]
            Target instruments to be long.
        short_list : set[str]
            Target instruments to be short.
        """
        # Instruments to close
        longs_to_close = self._long_positions - long_list
        shorts_to_close = self._short_positions - short_list

        # Also close if direction flips
        longs_to_close |= self._long_positions & short_list
        shorts_to_close |= self._short_positions & long_list

        # Close positions
        for instrument_id in longs_to_close:
            self._close_position(instrument_id, "REBALANCE_EXIT_LONG")
            self._long_positions.discard(instrument_id)

        for instrument_id in shorts_to_close:
            self._close_position(instrument_id, "REBALANCE_EXIT_SHORT")
            self._short_positions.discard(instrument_id)

        # Open new positions
        if self.config.enable_long:
            for instrument_id in long_list:
                if instrument_id not in self._long_positions:
                    if self._open_long(instrument_id):
                        self._long_positions.add(instrument_id)

        if self.config.enable_short:
            for instrument_id in short_list:
                if instrument_id not in self._short_positions:
                    if self._open_short(instrument_id):
                        self._short_positions.add(instrument_id)

    def _open_long(self, instrument_id_str: str) -> bool:
        """
        Open a long position.

        Parameters
        ----------
        instrument_id_str : str
            Instrument ID string.

        Returns
        -------
        bool
            True if order was submitted successfully.
        """
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
            order_side=OrderSide.BUY,
            quantity=qty,
        )
        self.submit_order(order)

        self.log.debug(f"OPEN LONG {instrument_id_str} @ {price:.4f}, qty={qty}")
        return True

    def _open_short(self, instrument_id_str: str) -> bool:
        """
        Open a short position.

        Parameters
        ----------
        instrument_id_str : str
            Instrument ID string.

        Returns
        -------
        bool
            True if order was submitted successfully.
        """
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
            order_side=OrderSide.SELL,
            quantity=qty,
        )
        self.submit_order(order)

        self.log.debug(f"OPEN SHORT {instrument_id_str} @ {price:.4f}, qty={qty}")
        return True

    def _close_position(self, instrument_id_str: str, reason: str) -> None:
        """
        Close a position for an instrument.

        Parameters
        ----------
        instrument_id_str : str
            Instrument ID string.
        reason : str
            Reason for closing.
        """
        instrument_id = InstrumentId.from_str(instrument_id_str)
        self.close_all_positions(instrument_id)
        self.log.debug(f"CLOSE {instrument_id_str}: {reason}")

    def _close_all_positions(self, reason: str) -> None:
        """
        Close all open positions.

        Parameters
        ----------
        reason : str
            Reason for closing.
        """
        for instrument_id_str in list(self._long_positions):
            self._close_position(instrument_id_str, reason)
        for instrument_id_str in list(self._short_positions):
            self._close_position(instrument_id_str, reason)

        self._long_positions.clear()
        self._short_positions.clear()
