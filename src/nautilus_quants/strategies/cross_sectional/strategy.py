# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
CrossSectionalFactorStrategy - Multi-factor cross-sectional selection strategy.

This strategy implements a market-neutral approach aligned with FMZ:
- Long the N instruments with lowest composite factor values
- Short the N instruments with highest composite factor values
- Rebalance every 4 hours (period=4)
- Only close positions when direction flips (FMZ style)
- Apply quantile clipping for factor normalization

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
    rebalance_period : int, default 4
        Rebalance every N hours (FMZ uses period=4).
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
    rebalance_period: int = 4  # FMZ: period=4 (every 4 hours)
    factor_weights: dict[str, float] = {}
    bar_types: list[str] = []
    enable_long: bool = True
    enable_short: bool = True
    buffer_ratio: float = 0.5  # Buffer threshold (0.5 = median)


class CrossSectionalFactorStrategy(Strategy):
    """
    Cross-sectional multi-factor selection strategy.

    This strategy:
    1. Receives factor values from FactorEngineActor via CustomData
    2. Normalizes factors with quantile clipping (20%-80%) + z-score
    3. Computes composite factor as weighted sum
    4. Longs bottom N instruments (lowest composite value)
    5. Shorts top N instruments (highest composite value)
    6. Only closes positions when direction flips (Market-Neutral style)

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
        self._n_instruments = len(config.instrument_ids)

        # Default factor weights if not specified
        self._factor_weights: dict[str, float] = config.factor_weights or {
            "volume": 0.6,
            "momentum": 0.4,
            "volatility": 0.3,
            "corr": 0.4,
        }

        # State tracking
        self._current_prices: dict[str, float] = {}
        self._long_positions: set[str] = set()  # instrument_id strings
        self._short_positions: set[str] = set()
        self._bar_count: int = 0
        self._signal_count: int = 0
        self._hour_count: int = 0  # Track hours for rebalance period

        # Accumulate factor values across instruments for cross-sectional computation
        # Structure: {factor_name: {instrument_id: value}}
        self._accumulated_factors: dict[str, dict[str, float]] = {}

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
        self.subscribe_data(DataType(FactorValues), client_id=None)
        self.log.info("Subscribed to FactorValues Data")

        self.log.info(
            f"Strategy started: n_positions={self.config.n_positions}, "
            f"position_value={self.config.position_value}, "
            f"rebalance_period={self.config.rebalance_period}h"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        # Close all positions
        self._close_all_positions("STRATEGY_STOP")

        self.log.info(
            f"CrossSectionalFactorStrategy stopped: "
            f"bars={self._bar_count}, signals={self._signal_count}, hours={self._hour_count}"
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

        FactorValues are published per-bar (one instrument at a time).
        We accumulate them and compute cross-sectional composite when
        we have all instruments for this hour.

        Parameters
        ----------
        data : FactorValues
            The FactorValues Data object.
        """
        if not isinstance(data, FactorValues):
            return

        self._signal_count += 1

        # Accumulate factor values from this FactorValues message
        factors = data.factors
        for factor_name, instrument_values in factors.items():
            if factor_name not in self._accumulated_factors:
                self._accumulated_factors[factor_name] = {}
            # Merge instrument values
            for instrument_id, value in instrument_values.items():
                self._accumulated_factors[factor_name][instrument_id] = value

        # Check if we have all instruments for this hour (one signal per instrument)
        n_accumulated = len(self._accumulated_factors.get("volume", {}))
        if n_accumulated < self._n_instruments:
            return

        # We have a complete set - this is one "hour"
        self._hour_count += 1

        # Only rebalance every rebalance_period hours
        if self._hour_count % self.config.rebalance_period != 0:
            # Clear accumulated factors for next hour but don't rebalance
            self._accumulated_factors = {}
            return

        # Compute composite factor from accumulated values
        composite = self._compute_composite_factor()

        # Debug logging
        if self._hour_count <= 20 or self._hour_count % 100 == 0:
            self.log.info(f"Hour #{self._hour_count}: Computed composite for {len(composite)} instruments")

        if len(composite) < 2 * self.config.n_positions:
            self.log.warning(
                f"Not enough instruments with valid factor values: "
                f"{len(composite)} < {2 * self.config.n_positions}"
            )
            self._accumulated_factors = {}
            return

        # Sort by composite value (ascending)
        sorted_instruments = sorted(composite.items(), key=lambda x: x[1])

        # Execute rebalancing with buffer zone
        self._rebalance_positions_with_buffer(sorted_instruments)

        # Clear accumulated factors for next period
        self._accumulated_factors = {}

    def _compute_composite_factor(self) -> dict[str, float]:
        """
        Compute composite factor from accumulated values using normalization.

        Normalization steps:
        1. Clip to 20%-80% quantile
        2. Subtract mean
        3. Divide by std

        Returns
        -------
        dict[str, float]
            Composite factor values keyed by instrument_id.
        """
        normalized_factors: dict[str, dict[str, float]] = {}

        # Normalize each factor
        for factor_name in self._factor_weights.keys():
            raw_values = self._accumulated_factors.get(factor_name, {})

            if not raw_values:
                continue

            # Filter out NaN values
            valid_values = {
                k: v for k, v in raw_values.items()
                if not math.isnan(v)
            }

            if len(valid_values) < 3:
                continue

            # Normalization: clip to 20%-80% quantile, then z-score
            normalized = self._normalize_factor(valid_values)
            if normalized:
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

    def _normalize_factor(self, values: dict[str, float]) -> dict[str, float]:
        """
        Normalize factor values using quantile clipping and z-score.

        Logic:
            factor_clip = factor.clip(quantile(0.2), quantile(0.8))
            factor_norm = (factor_clip - mean) / std

        Parameters
        ----------
        values : dict[str, float]
            Raw factor values.

        Returns
        -------
        dict[str, float]
            Normalized factor values.
        """
        if len(values) < 3:
            return {}

        vals = list(values.values())
        vals_sorted = sorted(vals)
        n = len(vals_sorted)

        # Calculate 20% and 80% quantiles
        q20_idx = int(n * 0.2)
        q80_idx = int(n * 0.8) - 1 if n > 1 else 0
        q20 = vals_sorted[q20_idx]
        q80 = vals_sorted[max(q80_idx, q20_idx)]

        # Clip values to quantile range
        clipped = {k: max(q20, min(q80, v)) for k, v in values.items()}

        # Calculate mean and std of clipped values
        clipped_vals = list(clipped.values())
        mean_val = sum(clipped_vals) / len(clipped_vals)
        variance = sum((v - mean_val) ** 2 for v in clipped_vals) / len(clipped_vals)
        std_val = math.sqrt(variance) if variance > 0 else 1.0

        if std_val == 0:
            std_val = 1.0

        # Normalize: (x - mean) / std
        normalized = {k: (v - mean_val) / std_val for k, v in clipped.items()}

        return normalized

    def _rebalance_positions_with_buffer(
        self,
        sorted_instruments: list[tuple[str, float]],
    ) -> None:
        """
        Rebalance with buffer zone.

        Logic:
        - Step 1 (Close): Close positions that exceeded buffer threshold
          - Long: close if rank > mid (fell too far, 边界持有)
          - Short: close if rank < mid (rose too far, 边界持有)
        - Step 2 (Open): Open new positions in target zones
          - Long: open if in top N and not already long
          - Short: open if in bottom N and not already short

        This naturally handles "flip" cases:
        - Long at rank 12 → rank 21: Step1 closes long, Step2 opens short

        Example (N=10, total=30, mid=15):
        - Long position at rank 1-10: open/hold
        - Long position at rank 11-15: hold (buffer zone, 边界持有)
        - Long position at rank 16-20: close (exceeded buffer)
        - Long position at rank 21-30: close, then open short (flip)

        Parameters
        ----------
        sorted_instruments : list[tuple[str, float]]
            Instruments sorted by composite factor (ascending, lowest=best for long).
        """
        n = self.config.n_positions
        total = len(sorted_instruments)
        mid = int(total * self.config.buffer_ratio)  # 默认 0.5 = 中位数

        # Build rank lookup: instrument_id -> rank (0-indexed, 0=best for long)
        rank = {inst_id: i for i, (inst_id, _) in enumerate(sorted_instruments)}

        # Target lists (for opening new positions)
        long_targets = {x[0] for x in sorted_instruments[:n]}      # top N (rank 0 to n-1)
        short_targets = {x[0] for x in sorted_instruments[-n:]}    # bottom N (rank total-n to total-1)

        # Log position ranks before rebalance (for debugging)
        self._log_position_ranks(rank, total, mid, long_targets, short_targets)

        # === STEP 1: CLOSE (with buffer) ===

        # Close longs that fell below buffer (rank > mid, 边界持有)
        for inst_id in list(self._long_positions):
            if inst_id in rank and rank[inst_id] > mid:
                self._close_position(inst_id, f"LONG_EXCEEDED_BUFFER_RANK_{rank[inst_id]}")
                self._long_positions.discard(inst_id)

        # Close shorts that rose above buffer (rank < mid, 边界持有)
        for inst_id in list(self._short_positions):
            if inst_id in rank and rank[inst_id] < mid:
                self._close_position(inst_id, f"SHORT_EXCEEDED_BUFFER_RANK_{rank[inst_id]}")
                self._short_positions.discard(inst_id)

        # === STEP 2: OPEN (target zones only) ===

        # Open new longs (top N)
        if self.config.enable_long:
            for inst_id in long_targets:
                if inst_id not in self._long_positions:
                    if self._open_long(inst_id):
                        self._long_positions.add(inst_id)

        # Open new shorts (bottom N)
        if self.config.enable_short:
            for inst_id in short_targets:
                if inst_id not in self._short_positions:
                    if self._open_short(inst_id):
                        self._short_positions.add(inst_id)

    def _log_position_ranks(
        self,
        rank: dict[str, int],
        total: int,
        mid: int,
        long_targets: set[str],
        short_targets: set[str],
    ) -> None:
        """Log current position ranks for debugging."""
        # Only log periodically to avoid spam
        if self._hour_count % 24 != 0 and self._hour_count > 20:
            return

        n = self.config.n_positions

        # Get ranks for current positions
        long_ranks = sorted([rank.get(inst, -1) for inst in self._long_positions])
        short_ranks = sorted([rank.get(inst, -1) for inst in self._short_positions])

        # Count positions in each zone
        long_in_target = len(self._long_positions & long_targets)
        long_in_buffer = len([r for r in long_ranks if n <= r <= mid])
        short_in_target = len(self._short_positions & short_targets)
        short_in_buffer = len([r for r in short_ranks if mid <= r < total - n])

        self.log.info(
            f"Rebalance #{self._hour_count // self.config.rebalance_period}: "
            f"total={total}, mid={mid}, "
            f"long_pos={len(self._long_positions)} (target={long_in_target}, buffer={long_in_buffer}), "
            f"short_pos={len(self._short_positions)} (target={short_in_target}, buffer={short_in_buffer})"
        )

        # Log detailed ranks for first few rebalances
        if self._hour_count <= 20:
            self.log.debug(
                f"Position ranks - Long: {long_ranks}, Short: {short_ranks}"
            )

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

        # Target value per position
        target_value = self.config.position_value
        raw_qty = Decimal(str(target_value / price))
        qty = instrument.make_qty(raw_qty)

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
        )
        self.submit_order(order)

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

        # Target value per position
        target_value = self.config.position_value
        raw_qty = Decimal(str(target_value / price))
        qty = instrument.make_qty(raw_qty)

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
        )
        self.submit_order(order)

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
