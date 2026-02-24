# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
WorldQuantAlphaStrategy - WorldQuant BRAIN 7-step portfolio construction.

Implements the WorldQuant BRAIN investment process:
  Step 1: Extract raw alpha vector from FactorValues
  Step 2: Apply delay (use previous period's data if delay=1)
  Step 3: Market neutralization (subtract mean → sum(alpha) = 0)
  Step 4: Scale (divide by |sum| → sum(|alpha|) = 1)
  Step 5: Apply linear decay to normalized weights (if decay > 0)
  Step 6: Truncation (cap max single-asset weight)
  Step 7: Capital allocation and order submission

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
    - Receives pre-computed signals, does not compute them (Principle V)
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.backtest.protocols import POSITION_METADATA_CACHE_KEY
from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.worldquant.metadata import WorldQuantMetadataProvider

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import Instrument


class WorldQuantAlphaConfig(StrategyConfig, frozen=True):
    """
    Configuration for WorldQuantAlphaStrategy.

    All parameters correspond to WorldQuant BRAIN simulation settings.

    Parameters
    ----------
    instrument_ids : list[str]
        Target instrument IDs (e.g., ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"]).
    factor_name : str, default "alpha_101"
        Name of the factor to use from FactorValues (must match factor config).
    delay : int, default 1
        Data delay. 0 = use current bar data, 1 = use previous bar's data.
        BRAIN default is delay=1 (avoids look-ahead bias).
    decay : int, default 0
        Linear decay window. 0 = no decay, N = weighted average over N+1 periods.
        BRAIN weights: oldest=1, ..., newest=N.
    neutralization : str, default "MARKET"
        "MARKET" = subtract cross-sectional mean (market neutral).
        "NONE" = no neutralization.
    truncation : float, default 0.0
        Maximum weight per instrument. 0.0 = no truncation.
        E.g., 0.05 caps each instrument at 5% of total book.
    rebalance_period : int, default 24
        Rebalance every N signals (1 signal = 1 hour bar). 24 = daily.
    capital : float, default 100_000.0
        Virtual capital for position sizing (in quote currency).
    enable_long : bool, default True
        Whether to open long positions (positive weights).
    enable_short : bool, default True
        Whether to open short positions (negative weights).
    bar_types : list[str], default []
        Bar type strings for price subscription (injected by CLI).
    """

    instrument_ids: list[str]
    factor_name: str = "alpha_101"
    delay: int = 1
    decay: int = 0
    neutralization: str = "MARKET"
    truncation: float = 0.0
    rebalance_period: int = 24
    capital: float = 100_000.0
    enable_long: bool = True
    enable_short: bool = True
    bar_types: list[str] = []


class WorldQuantAlphaStrategy(BarSubscriptionMixin, Strategy):
    """
    WorldQuant BRAIN Alpha Strategy.

    Implements the 7-step BRAIN portfolio construction pipeline:
      1. Extract raw alpha vector from FactorEngineActor signals
      2. Apply delay buffer (prevents look-ahead bias)
      3. Market neutralization (sum(alpha) = 0, long/short balanced)
      4. Scale (sum(|alpha|) = 1, defines capital allocation fractions)
      5. Apply linear decay to normalized weights (smooth over recent history)
      6. Truncation (cap max single-instrument weight)
      7. Capital allocation and rebalancing

    Position sizing:
      target_dollar_value = capital * weight
      positive weight → long position
      negative weight → short position
    """

    def __init__(self, config: WorldQuantAlphaConfig) -> None:
        """Initialize WorldQuantAlphaStrategy."""
        super().__init__(config)

        self._instrument_ids: list[InstrumentId] = [
            InstrumentId.from_str(iid) for iid in config.instrument_ids
        ]
        self._instruments: dict[InstrumentId, Instrument] = {}

        # Price tracking
        self._current_prices: dict[str, float] = {}

        # Position state
        self._long_positions: set[str] = set()
        self._short_positions: set[str] = set()

        # BRAIN pipeline state
        self._prev_alpha: dict[str, float] | None = None   # delay buffer
        self._alpha_history: list[dict[str, float]] = []    # decay window

        # Signal counting
        self._signal_count: int = 0
        self._rebalance_count: int = 0
        self._bar_count: int = 0

        # Metadata tracking
        self._metadata_provider: WorldQuantMetadataProvider = WorldQuantMetadataProvider()
        self._current_alpha101: dict[str, float] = {}
        self._current_ts_event: int = 0

        # Pipeline intermediate values (captured in _process_alpha)
        self._last_neutralized: dict[str, float] = {}
        self._last_scaled: dict[str, float] = {}
        self._last_decayed: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Nautilus lifecycle
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """Subscribe to data on strategy start."""
        self.log.info(
            f"Starting WorldQuantAlphaStrategy with {len(self._instrument_ids)} instruments"
        )

        # Cache instrument objects
        for instrument_id in self._instrument_ids:
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                self.log.warning(f"Instrument not found in cache: {instrument_id}")
            else:
                self._instruments[instrument_id] = instrument

        if not self._instruments:
            self.log.error("No instruments found in cache")
            self.stop()
            return

        self.log.info(f"Cached {len(self._instruments)} instruments")

        # Subscribe to price bars
        if not self.config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically."
            )
            self.stop()
            return

        self._subscribe_bar_types(self.config.bar_types)
        self.log.debug(f"Subscribed to {len(self.config.bar_types)} bar type(s)")

        # Subscribe to factor signals
        self.subscribe_data(DataType(FactorValues))
        self.log.info("Subscribed to FactorValues")

        self.log.info(
            f"Config: factor={self.config.factor_name}, delay={self.config.delay}, "
            f"decay={self.config.decay}, neutralization={self.config.neutralization}, "
            f"truncation={self.config.truncation}, rebalance_period={self.config.rebalance_period}h, "
            f"capital={self.config.capital:,.0f}"
        )

    def on_stop(self) -> None:
        """Close all positions on strategy stop."""
        self._close_all_positions("STRATEGY_STOP")
        all_metadata = self._metadata_provider.get_all_metadata()
        if all_metadata:
            self.cache.add(POSITION_METADATA_CACHE_KEY, self._metadata_provider.serialize())
            self.log.info(f"Position metadata: {len(all_metadata)} positions stored in cache")
        self.log.info(
            f"WorldQuantAlphaStrategy stopped: bars={self._bar_count}, "
            f"signals={self._signal_count}, rebalances={self._rebalance_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates - track current prices for position sizing."""
        instrument_id_str = self._resolve_bar(bar)
        if instrument_id_str is None:
            return
        self._bar_count += 1
        self._current_prices[instrument_id_str] = float(bar.close)

    def on_data(self, data: object) -> None:
        """
        Handle FactorValues from FactorEngineActor.

        Each FactorValues message contains the full cross-sectional alpha
        vector for all instruments at the current timestamp.
        """
        if not isinstance(data, FactorValues):
            return
        self._handle_factor_values(data)

    # -------------------------------------------------------------------------
    # BRAIN 7-step pipeline
    # -------------------------------------------------------------------------

    def _handle_factor_values(self, data: FactorValues) -> None:
        """
        Process incoming FactorValues through the full BRAIN pipeline.

        Steps 1-7 of the WorldQuant BRAIN portfolio construction process.
        """
        # Step 1: Extract raw alpha vector for the configured factor
        raw = data.factors.get(self.config.factor_name, {})
        if not raw:
            self.log.debug(f"No data for factor '{self.config.factor_name}'")
            return

        # Filter out NaN values
        raw = {k: v for k, v in raw.items() if not math.isnan(v)}
        if not raw:
            return

        # Step 2: Apply delay
        alpha = self._apply_delay(raw)
        if alpha is None:
            self.log.debug("Warmup period: delay buffer not yet filled")
            return  # Warmup period - skip rebalancing

        # Process through neutralize/scale/decay/truncate (BRAIN order)
        processed = self._process_alpha(alpha)
        if not processed:
            return

        # Save current signal state for record_open in _rebalance
        self._current_alpha101 = alpha
        self._current_ts_event = data.ts_event

        # Update alpha101 history for all open positions on every signal cycle
        all_open = list(self._long_positions | self._short_positions)
        if all_open:
            self._metadata_provider.update_alpha101_history(
                instrument_ids=all_open,
                alpha101_lookup=alpha,
                weight_lookup=processed,
                neutralized_lookup=self._last_neutralized,
                scaled_lookup=self._last_scaled,
                decayed_lookup=self._last_decayed,
                ts_event=data.ts_event,
            )

        # Step 7: Check rebalance period and execute
        self._signal_count += 1
        if not self._should_rebalance():
            return

        self.log.info(
            f"Rebalancing #{self._rebalance_count + 1}: "
            f"{len(processed)} instruments, signal_count={self._signal_count}"
        )
        self._rebalance(processed)

    def _process_alpha(self, alpha: dict[str, float]) -> dict[str, float]:
        """
        Apply Steps 3-6: neutralize, scale, decay, truncate.

        Returns the final weight vector. Empty dict if all values are NaN.
        """
        # Filter NaN
        alpha = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not alpha:
            return {}

        # Step 3: Market neutralization
        if self.config.neutralization == "MARKET":
            alpha = self._neutralize(alpha)
        self._last_neutralized = dict(alpha)

        # Step 4: Scale (sum(|alpha|) = 1)
        alpha = self._scale(alpha)
        self._last_scaled = dict(alpha)

        # Step 5: Apply decay to normalized weights (BRAIN order)
        # Decay computes a weighted average of past normalized weight vectors.
        # Re-scale afterwards since the weighted average may not sum to 1.
        alpha = self._apply_decay(alpha)
        alpha = self._scale(alpha)
        self._last_decayed = dict(alpha)

        # Step 6: Iterative truncation + re-scale (repeat until convergence)
        # Re-scaling after truncation can push remaining weights above threshold,
        # so we iterate until no weight exceeds the truncation limit.
        if self.config.truncation > 0:
            for _ in range(20):  # max iterations for convergence
                truncated = self._truncate(alpha)
                alpha = self._scale(truncated)
                if max(abs(v) for v in alpha.values()) <= self.config.truncation + 1e-10:
                    break

        return alpha

    # -------------------------------------------------------------------------
    # Step 2: Delay
    # -------------------------------------------------------------------------

    def _apply_delay(self, raw: dict[str, float]) -> dict[str, float] | None:
        """
        Apply delay buffer.

        delay=0: use current period's data directly.
        delay=1: use previous period's data (returns None during warmup).
        """
        if self.config.delay == 0:
            return raw

        # delay=1: slide the buffer
        prev = self._prev_alpha
        self._prev_alpha = raw
        return prev  # None on first call (warmup)

    # -------------------------------------------------------------------------
    # Step 5: Decay (applied to normalized weights, per BRAIN order)
    # -------------------------------------------------------------------------

    def _apply_decay(self, alpha: dict[str, float]) -> dict[str, float]:
        """
        Apply linear decay weighted average to normalized weights.

        Called after neutralize+scale so history contains normalized vectors.

        decay=0: return current alpha unchanged.
        decay=N: weighted average over last N+1 periods.
                 Weights: oldest=1, ..., newest=N (most recent is N).

        History stored in chronological order: [oldest, ..., newest].
        """
        if self.config.decay == 0:
            return alpha

        # Append to history (chronological: oldest first)
        self._alpha_history.append(alpha)
        # Keep only decay+1 periods
        if len(self._alpha_history) > self.config.decay + 1:
            self._alpha_history.pop(0)

        n = len(self._alpha_history)
        # Weights: history[0]=oldest=weight 1, ..., history[-1]=newest=weight n
        weights = list(range(1, n + 1))
        total_weight = sum(weights)

        # Collect all instrument keys
        all_keys: set[str] = set()
        for hist in self._alpha_history:
            all_keys.update(hist.keys())

        result: dict[str, float] = {}
        for key in all_keys:
            weighted_sum = sum(
                w * hist.get(key, 0.0)
                for w, hist in zip(weights, self._alpha_history)
            )
            result[key] = weighted_sum / total_weight

        return result

    # -------------------------------------------------------------------------
    # Step 4: Neutralization
    # -------------------------------------------------------------------------

    def _neutralize(self, alpha: dict[str, float]) -> dict[str, float]:
        """
        Market neutralization: subtract cross-sectional mean.

        Result: sum(alpha) = 0 → market neutral (equal long/short exposure).
        """
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        if not valid:
            return alpha

        mean = sum(valid.values()) / len(valid)
        result = {k: v - mean for k, v in valid.items()}
        # Preserve NaN entries
        for k, v in alpha.items():
            if math.isnan(v):
                result[k] = v
        return result

    # -------------------------------------------------------------------------
    # Step 5: Scale
    # -------------------------------------------------------------------------

    def _scale(self, alpha: dict[str, float]) -> dict[str, float]:
        """
        Scale alpha vector so sum(|alpha|) = 1.

        Defines how capital is allocated across instruments.
        """
        valid = {k: v for k, v in alpha.items() if not math.isnan(v)}
        total_abs = sum(abs(v) for v in valid.values())
        if total_abs == 0:
            return alpha
        return {k: v / total_abs for k, v in valid.items()}

    # -------------------------------------------------------------------------
    # Step 6: Truncation
    # -------------------------------------------------------------------------

    def _truncate(self, alpha: dict[str, float]) -> dict[str, float]:
        """
        Cap individual instrument weights at the configured threshold.

        Prevents concentration risk in single instruments.
        """
        threshold = self.config.truncation
        return {
            k: max(-threshold, min(threshold, v))
            for k, v in alpha.items()
        }

    # -------------------------------------------------------------------------
    # Step 7: Rebalancing
    # -------------------------------------------------------------------------

    def _should_rebalance(self) -> bool:
        """Check if it's time to rebalance based on signal counter."""
        period: int = self.config.rebalance_period
        return self._signal_count % period == 0

    def _rebalance(self, weights: dict[str, float]) -> None:
        """
        Execute portfolio rebalancing based on target weights.

        - Positive weight → long position (if enable_long)
        - Negative weight → short position (if enable_short)
        - Weight ≈ 0 → close position
        """
        self._rebalance_count += 1

        # Determine new target positions
        new_longs: set[str] = set()
        new_shorts: set[str] = set()

        for inst_id, weight in weights.items():
            if weight > 1e-8 and self.config.enable_long:
                new_longs.add(inst_id)
            elif weight < -1e-8 and self.config.enable_short:
                new_shorts.add(inst_id)

        # Close positions no longer in target
        for inst_id in list(self._long_positions):
            if inst_id not in new_longs:
                self._close_position(inst_id, "EXIT_LONG")
                self._long_positions.discard(inst_id)

        for inst_id in list(self._short_positions):
            if inst_id not in new_shorts:
                self._close_position(inst_id, "EXIT_SHORT")
                self._short_positions.discard(inst_id)

        # Open/resize long positions
        for inst_id in new_longs:
            weight = weights[inst_id]
            target_value = self.config.capital * weight
            # Close opposite position first if needed
            if inst_id in self._short_positions:
                self._close_position(inst_id, "FLIP_TO_LONG")
                self._short_positions.discard(inst_id)
            is_new_long = inst_id not in self._long_positions
            if self._open_position(inst_id, OrderSide.BUY, target_value):
                self._long_positions.add(inst_id)
                if is_new_long:
                    self._metadata_provider.record_open(
                        instrument_id=inst_id,
                        side="LONG",
                        alpha101=self._current_alpha101.get(inst_id),
                        weight=weight,
                        neutralized=self._last_neutralized.get(inst_id),
                        scaled=self._last_scaled.get(inst_id),
                        decayed=self._last_decayed.get(inst_id),
                        ts_event=self._current_ts_event,
                    )

        # Open/resize short positions
        for inst_id in new_shorts:
            weight = weights[inst_id]
            target_value = self.config.capital * abs(weight)
            # Close opposite position first if needed
            if inst_id in self._long_positions:
                self._close_position(inst_id, "FLIP_TO_SHORT")
                self._long_positions.discard(inst_id)
            is_new_short = inst_id not in self._short_positions
            if self._open_position(inst_id, OrderSide.SELL, target_value):
                self._short_positions.add(inst_id)
                if is_new_short:
                    self._metadata_provider.record_open(
                        instrument_id=inst_id,
                        side="SHORT",
                        alpha101=self._current_alpha101.get(inst_id),
                        weight=weights[inst_id],
                        neutralized=self._last_neutralized.get(inst_id),
                        scaled=self._last_scaled.get(inst_id),
                        decayed=self._last_decayed.get(inst_id),
                        ts_event=self._current_ts_event,
                    )

        if self._rebalance_count <= 5 or self._rebalance_count % 30 == 0:
            self.log.info(
                f"Rebalance #{self._rebalance_count}: "
                f"long={len(self._long_positions)}, short={len(self._short_positions)}"
            )

    def _open_position(
        self,
        instrument_id_str: str,
        side: OrderSide,
        target_value: float,
    ) -> bool:
        """Open a position with target dollar value."""
        instrument_id = InstrumentId.from_str(instrument_id_str)
        instrument = self._instruments.get(instrument_id)

        if instrument is None:
            return False

        price = self._current_prices.get(instrument_id_str)
        if price is None or price <= 0:
            return False

        raw_qty = Decimal(str(target_value / price))
        try:
            qty = instrument.make_qty(raw_qty)
        except ValueError:
            # Quantity rounds to zero (e.g. high-priced instruments with integer lot size)
            return False

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)
        self.log.debug(
            f"OPEN {side.name} {instrument_id_str}: qty={qty}, value={target_value:.2f}"
        )
        return True

    def _close_position(self, instrument_id_str: str, reason: str) -> None:
        """Close all positions for an instrument."""
        instrument_id = InstrumentId.from_str(instrument_id_str)
        self.close_all_positions(instrument_id)
        self._metadata_provider.record_close(instrument_id_str, reason, self._signal_count)
        self.log.debug(f"CLOSE {instrument_id_str}: {reason}")

    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        for inst_id in list(self._long_positions):
            self._close_position(inst_id, reason)
        for inst_id in list(self._short_positions):
            self._close_position(inst_id, reason)
        self._long_positions.clear()
        self._short_positions.clear()
