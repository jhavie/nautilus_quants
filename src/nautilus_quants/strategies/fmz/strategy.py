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
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders.base import Order
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.backtest.protocols import POSITION_METADATA_CACHE_KEY
from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.common.event_time_pending_execution import (
    EventTimePendingExecutionMixin,
)
from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.fmz.metadata import FMZMetadataProvider

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
    rebalance_interval : int, default 1
        Rebalance every N bars (e.g., 2 means every 2nd valid signal).
        Equivalent to qlib's rebalance_interval parameter.
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    bar_types : list[str], default []
        Bar type strings for data subscription (injected by CLI).
    """

    instrument_ids: list[str]
    n_long: int = 40
    n_short: int = 40
    position_value: float = 300.0
    rebalance_interval: int = 1
    composite_factor: str = "composite"
    bar_types: list[str] = []


class FMZFactorStrategy(
    EventTimePendingExecutionMixin[dict[str, float]],
    BarSubscriptionMixin,
    Strategy,
):
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
        self._n_instruments = len(config.instrument_ids)

        # State tracking
        self._init_event_time_pending(max_timestamps=6, max_pending_wait_timestamps=2)
        self._long_positions: dict[str, Quantity] = {}
        self._short_positions: dict[str, Quantity] = {}
        self._hour_count: int = 0
        self._bar_count: int = 0
        self._bars_until_rebalance: int = 0  # 0 = rebalance on next valid signal

        # Metadata tracking for position visualization
        self._metadata_provider = FMZMetadataProvider()

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self._stopping = False
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

        self._subscribe_bar_types(self.config.bar_types)

        # Subscribe to factor data
        self.subscribe_data(DataType(FactorValues))
        self.log.info("Subscribed to FactorValues Data")

        self.log.info(
            f"Strategy started: n_long={self.config.n_long}, n_short={self.config.n_short}, "
            f"position_value={self.config.position_value}, "
            f"rebalance_interval={self.config.rebalance_interval} bars"
        )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        self._stopping = True
        self._close_all_positions("STRATEGY_STOP")

        # Store metadata in cache for report generation
        all_metadata = self._metadata_provider.get_all_metadata()
        if all_metadata:
            self.cache.add(POSITION_METADATA_CACHE_KEY, self._metadata_provider.serialize())
            self.log.info(f"Position metadata: {len(all_metadata)} positions stored in cache")

        self.log.info(
            f"FMZFactorStrategy stopped: bars={self._bar_count}, hours={self._hour_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates - track closes by event timestamp."""
        instrument_id = self._resolve_bar(bar)
        if instrument_id is None:
            return

        self._bar_count += 1
        self._record_close_and_try_execute(bar.ts_event, instrument_id, float(bar.close))

    def on_data(self, data) -> None:
        """Handle FactorValues and defer execution until signal-ts prices are complete."""
        if not isinstance(data, FactorValues):
            return

        signal_ts = data.ts_event
        factors = data.factors
        composite_factor = self.config.composite_factor

        # Diagnostic: log factor reception for first 5 calls
        if self._bar_count <= 5 * self._n_instruments or self._hour_count == 0:
            factor_summary = {k: len(v) for k, v in factors.items()}
            has_composite = composite_factor in factors and len(factors[composite_factor]) > 0
            self.log.info(
                f"on_data received: ts={signal_ts}, "
                f"composite_present={has_composite}, "
                f"factor_sizes={factor_summary}"
            )

        composite = factors.get(composite_factor, {})
        if not composite:
            self.log.debug(f"Skip signal_ts={signal_ts}: no '{composite_factor}' values")
            return

        if len(composite) < self.config.n_long + self.config.n_short:
            self.log.warning(
                f"Skip signal_ts={signal_ts}: not enough valid instruments "
                f"{len(composite)} < {self.config.n_long + self.config.n_short}"
            )
            return

        self._enqueue_pending(signal_ts, dict(composite))

    def _required_instruments(self, payload: dict[str, float]) -> list[str]:
        """Return required instruments for this pending composite payload."""
        return list(payload.keys())

    def _on_pending_ready(
        self,
        signal_ts: int,
        payload: dict[str, float],
        execution_prices: dict[str, float],
    ) -> None:
        """Apply FMZ interval gate then execute rebalance on ready snapshot."""
        self._hour_count += 1
        if self._bars_until_rebalance > 0:
            self.log.info(
                f"Skip rebalance: signal_ts={signal_ts}, "
                f"rebalance_gate_state=countdown({self._bars_until_rebalance})"
            )
            self._bars_until_rebalance -= 1
            return

        self._bars_until_rebalance = self.config.rebalance_interval - 1
        ts_dt = datetime.fromtimestamp(signal_ts / 1e9, tz=timezone.utc)
        if not self._long_positions and not self._short_positions:
            sorted_composite = sorted(payload.items(), key=lambda x: (x[1], x[0]))
            self.log.info(f"FIRST REBALANCE at {ts_dt} - Composite factor rankings:")
            self.log.info(
                f"  Bottom 5 (LONG): {[(k, f'{v:.4f}') for k, v in sorted_composite[:5]]}"
            )
            self.log.info(
                f"  Top 5 (SHORT): {[(k, f'{v:.4f}') for k, v in sorted_composite[-5:]]}"
            )

        self._rebalance(
            composite=payload,
            execution_prices=execution_prices,
            signal_ts=signal_ts,
        )

    def _rebalance(
        self,
        composite: dict[str, float],
        execution_prices: dict[str, float],
        signal_ts: int,
    ) -> None:
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
        sorted_symbols = sorted(composite.items(), key=lambda x: (x[1], x[0]))

        # Build rank and composite lookups for metadata tracking
        rank_lookup = {s: i for i, (s, _) in enumerate(sorted_symbols)}
        composite_lookup = dict(sorted_symbols)

        # Determine long/short targets
        long_targets = set([s for s, _ in sorted_symbols[:self.config.n_long]])
        short_targets = set([s for s, _ in sorted_symbols[-self.config.n_short:]])

        # === Close positions for instruments missing from factor data ===
        # This handles delisted instruments or data gaps
        instruments_with_data = set(composite.keys())

        # Check long positions for missing data
        missing_long = sorted(set(self._long_positions) - instruments_with_data)
        # Check short positions for missing data
        missing_short = sorted(set(self._short_positions) - instruments_with_data)

        # Fetch latest closes only when needed (deterministic anchor for fill model)
        if missing_long or missing_short:
            latest_closes = self._price_book.get_latest_closes()

        for inst_id in missing_long:
            self.log.warning(f"Closing LONG {inst_id}: no factor data (possible delisting)")
            self._close_position(
                inst_id, "NO_FACTOR_DATA",
                exec_price=latest_closes.get(inst_id),
            )
            self._long_positions.pop(inst_id, None)

        for inst_id in missing_short:
            self.log.warning(f"Closing SHORT {inst_id}: no factor data (possible delisting)")
            self._close_position(
                inst_id, "NO_FACTOR_DATA",
                exec_price=latest_closes.get(inst_id),
            )
            self._short_positions.pop(inst_id, None)

        if self._hour_count <= 10 or self._hour_count % 24 == 0:
            self.log.info(
                f"Rebalance #{self._hour_count}: "
                f"long={len(long_targets)}, short={len(short_targets)}, "
                f"current_long={len(self._long_positions)}, current_short={len(self._short_positions)}"
            )

        # === FMZ Logic adapted for HEDGING mode ===
        for inst_id in sorted(composite.keys()):
            is_long_target = inst_id in long_targets
            is_short_target = inst_id in short_targets
            currently_long = inst_id in self._long_positions
            currently_short = inst_id in self._short_positions
            exec_price = execution_prices.get(inst_id)
            if exec_price is None or exec_price <= 0:
                self.log.warning(
                    f"Skip {inst_id}: missing/invalid execution price "
                    f"for signal_ts={signal_ts}"
                )
                continue

            # FMZ: if in buy_symbols AND amount <= 0, then Buy
            if is_long_target and not currently_long:
                # If currently short, close first then open long
                if currently_short:
                    self._close_position(inst_id, "FLIP_TO_LONG", exec_price=exec_price)
                    self._short_positions.pop(inst_id, None)
                qty_opened = self._open_position(
                    inst_id,
                    OrderSide.BUY,
                    exec_price=exec_price,
                    rank=rank_lookup.get(inst_id),
                    composite_value=composite_lookup.get(inst_id),
                    ts_event=signal_ts,
                )
                if qty_opened is not None:
                    self._long_positions[inst_id] = qty_opened

            # FMZ: if in sell_symbols AND amount >= 0, then Sell
            elif is_short_target and not currently_short:
                # If currently long, close first then open short
                if currently_long:
                    self._close_position(inst_id, "FLIP_TO_SHORT", exec_price=exec_price)
                    self._long_positions.pop(inst_id, None)
                qty_opened = self._open_position(
                    inst_id,
                    OrderSide.SELL,
                    exec_price=exec_price,
                    rank=rank_lookup.get(inst_id),
                    composite_value=composite_lookup.get(inst_id),
                    ts_event=signal_ts,
                )
                if qty_opened is not None:
                    self._short_positions[inst_id] = qty_opened

            # FMZ CRITICAL: Do NOT close positions that are not in target
            # Positions are only closed when they FLIP direction.

        # Update rank history for all current positions
        all_positions = sorted(set(self._long_positions) | set(self._short_positions))
        self._metadata_provider.update_rank_history(
            instrument_ids=all_positions,
            rank_lookup=rank_lookup,
            composite_lookup=composite_lookup,
            ts_event=signal_ts,
        )

    def _open_position(
        self,
        instrument_id_str: str,
        side: OrderSide,
        exec_price: float,
        rank: int | None = None,
        composite_value: float | None = None,
        ts_event: int | None = None,
    ) -> Quantity | None:
        """Open position with fixed value and record metadata.

        Returns the submitted Quantity on success, or None on failure.
        """
        instrument_id = InstrumentId.from_str(instrument_id_str)
        instrument = self._instruments.get(instrument_id)

        if instrument is None:
            return None

        if exec_price <= 0:
            return None

        raw_qty = Decimal(str(self.config.position_value / exec_price))
        qty = instrument.make_qty(raw_qty)

        # Record position metadata
        self._metadata_provider.record_open(
            instrument_id=instrument_id_str,
            side="LONG" if side == OrderSide.BUY else "SHORT",
            rank=rank,
            composite=composite_value,
            ts_event=ts_event or 0,
        )

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=qty,
            exec_algorithm_params={"anchor_px": str(exec_price)},
        )
        self.submit_order(order)

        return qty

    def _close_position(
        self,
        instrument_id_str: str,
        reason: str,
        exec_price: float | None = None,
    ) -> None:
        """Close a position and record metadata.

        When exec_price is provided, creates explicit MarketOrders with anchor_px
        so the SignalCloseFillModel can fill at the deterministic signal-time price.
        """
        instrument_id = InstrumentId.from_str(instrument_id_str)
        self._metadata_provider.record_close(
            instrument_id=instrument_id_str,
            reason=reason,
            hour_count=self._hour_count,
        )
        for position in self.cache.positions_open(
            instrument_id=instrument_id, strategy_id=self.id
        ):
            if position.is_closed:
                continue
            if exec_price is not None and exec_price > 0:
                params = {"anchor_px": str(exec_price)}
            else:
                self.log.warning(
                    f"CLOSE {instrument_id_str}: no anchor_px, using default matching"
                )
                params = None
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=Order.closing_side(position.side),
                quantity=position.quantity,
                reduce_only=True,
                exec_algorithm_params=params,
            )
            self.submit_order(order, position_id=position.id)
        self.log.debug(f"CLOSE {instrument_id_str}: {reason}")

    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions from cache with deterministic fill pricing.

        Uses cache.positions_open() as source of truth instead of tracking dicts,
        because pending rebalance orders may have updated tracking dicts without
        their fills being reflected in cache yet.
        """
        latest_closes = self._price_book.get_latest_closes()

        for position in sorted(
            self.cache.positions_open(strategy_id=self.id),
            key=lambda p: str(p.instrument_id),
        ):
            if position.is_closed:
                continue
            inst_id = str(position.instrument_id)
            close_side = Order.closing_side(position.side)
            self._metadata_provider.record_close(
                instrument_id=inst_id, reason=reason, hour_count=self._hour_count,
            )
            anchor_px = latest_closes.get(inst_id)
            if anchor_px is None:
                self.log.warning(f"CLOSE {inst_id}: no anchor_px in latest_closes, using default matching")
            params = {"anchor_px": str(anchor_px)} if anchor_px is not None else None
            order = self.order_factory.market(
                instrument_id=position.instrument_id,
                order_side=close_side,
                quantity=position.quantity,
                reduce_only=True,
                exec_algorithm_params=params,
            )
            self.submit_order(order, position_id=position.id)
            self.log.debug(f"CLOSE {inst_id}: {reason}")

        self._long_positions.clear()
        self._short_positions.clear()

    def on_position_opened(self, event) -> None:
        """Auto-close positions opened after strategy stop (late rebalance fills)."""
        if not self._stopping:
            return

        position = self.cache.position(event.position_id)
        if position is None or position.is_closed:
            return

        inst_id = str(position.instrument_id)
        close_side = Order.closing_side(position.side)
        latest_closes = self._price_book.get_latest_closes()
        anchor_px = latest_closes.get(inst_id)
        params = {"anchor_px": str(anchor_px)} if anchor_px is not None else None

        order = self.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=close_side,
            quantity=position.quantity,
            reduce_only=True,
            exec_algorithm_params=params,
        )
        self.submit_order(order, position_id=position.id)
        self.log.info(f"LATE_CLOSE {inst_id}: position opened after stop, closing immediately")
