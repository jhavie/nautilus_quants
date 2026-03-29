# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
CrossSectionalFactorStrategy - Multi-factor cross-sectional selection strategy.

This strategy implements a market-neutral approach:
- Long the N instruments with lowest composite factor values
- Short the N instruments with highest composite factor values
- Rebalance every N hours (configurable via rebalance_period)
- Only close positions when direction flips (buffer zone logic)

Constitution Compliance:
    - Extends Nautilus Strategy base class (Principle I)
    - Uses Nautilus CustomData subscription (Principle I)
    - Configuration-driven via StrategyConfig (Principle II)
    - Receives pre-computed signals, does not compute them (Principle V)
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from nautilus_trader.model.objects import Currency
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.utils.cache_keys import POSITION_METADATA_CACHE_KEY
from nautilus_quants.common.anchor_price_execution import AnchorPriceExecutionMixin
from nautilus_quants.common.bar_subscription import BarSubscriptionMixin
from nautilus_quants.common.event_time_pending_execution import (
    EventTimePendingExecutionMixin,
)
from nautilus_quants.common.limit_order_execution import LimitOrderExecutionMixin
from nautilus_quants.factors.types import FactorValues
from nautilus_quants.strategies.cross_sectional.metadata import CrossSectionalMetadataProvider

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
    position_value : float | None, default None
        Fixed position value per instrument in quote currency (USDT).
        If None and monthly_position_update is True, calculated dynamically.
    rebalance_period : int, default 4
        Rebalance every N hours.
    composite_factor : str, default "composite"
        Name of the composite factor to use for ranking.
    bar_types : list[str], default []
        Bar type strings for data subscription (injected by CLI).
    enable_long : bool, default True
        Enable long positions.
    enable_short : bool, default True
        Enable short positions.
    buffer_ratio : float, default 0.5
        Buffer threshold ratio (0.5 = median).
    target_leverage : float, default 4.0
        Target leverage for dynamic position sizing.
        Formula: equity * target_leverage / total_positions = position_value
    monthly_position_update : bool, default True
        If True, update position value on the 1st of each month based on account equity.
    """

    instrument_ids: list[str]
    n_positions: int = 10
    position_value: float | None = None
    rebalance_period: int = 4
    composite_factor: str = "composite"
    bar_types: list[str] = []
    enable_long: bool = True
    enable_short: bool = True
    buffer_ratio: float = 0.5
    target_leverage: float = 4.0
    monthly_position_update: bool = True
    execution_mode: str = "anchor"  # "anchor" | "post_limit"


class CrossSectionalFactorStrategy(
    AnchorPriceExecutionMixin,
    LimitOrderExecutionMixin,
    EventTimePendingExecutionMixin[list[tuple[str, float]]],
    BarSubscriptionMixin,
    Strategy,
):
    """
    Cross-sectional multi-factor selection strategy.

    This strategy:
    1. Receives pre-computed factor values from FactorEngineActor via CustomData
    2. Uses the composite factor directly (normalization done in factor engine)
    3. Longs bottom N instruments (lowest composite value)
    4. Shorts top N instruments (highest composite value)
    5. Only closes positions when direction flips (buffer zone logic)

    The strategy is market-neutral with equal long/short exposure.
    """

    def __init__(self, config: CrossSectionalFactorStrategyConfig) -> None:
        """Initialize the CrossSectionalFactorStrategy."""
        super().__init__(config)

        self._instrument_ids: list[InstrumentId] = [
            InstrumentId.from_str(iid) for iid in config.instrument_ids
        ]
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._n_instruments = len(config.instrument_ids)

        # State tracking
        self._init_event_time_pending(max_timestamps=6, max_pending_wait_timestamps=2)
        self._long_positions: set[str] = set()
        self._short_positions: set[str] = set()
        self._bar_count: int = 0
        self._signal_count: int = 0
        self._hour_count: int = 0

        # Monthly position update state
        self._current_month: int | None = None
        self._monthly_position_value: float | None = None

        # Position metadata provider for reporting
        self._metadata_provider = CrossSectionalMetadataProvider()

    def on_start(self) -> None:
        """Actions to perform on strategy start."""
        self.log.info(
            f"Starting CrossSectionalFactorStrategy with {len(self._instrument_ids)} instruments"
        )

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

        if not self.config.bar_types:
            self.log.error(
                "bar_types not configured. Ensure backtest is run via CLI which injects "
                "bar_types from data config automatically."
            )
            self.stop()
            return

        self._subscribe_bar_types(self.config.bar_types)

        self.subscribe_data(DataType(FactorValues), client_id=ClientId(self.id.value))
        self.log.info("Subscribed to FactorValues Data")

        if self.config.monthly_position_update:
            self.log.info(
                f"Strategy started: n_positions={self.config.n_positions}, "
                f"target_leverage={self.config.target_leverage}, "
                f"monthly_position_update=True, "
                f"rebalance_period={self.config.rebalance_period}h"
            )
        else:
            self.log.info(
                f"Strategy started: n_positions={self.config.n_positions}, "
                f"position_value={self.config.position_value}, "
                f"rebalance_period={self.config.rebalance_period}h, "
                f"composite_factor={self.config.composite_factor}"
            )

    def on_stop(self) -> None:
        """Actions to perform on strategy stop."""
        self._close_all_positions("STRATEGY_STOP")

        # Store metadata in cache for report generation
        all_metadata = self._metadata_provider.get_all_metadata()
        if all_metadata:
            self.cache.add(POSITION_METADATA_CACHE_KEY, self._metadata_provider.serialize())
            self.log.info(f"Position metadata: {len(all_metadata)} positions stored in cache")

        self.log.info(
            f"CrossSectionalFactorStrategy stopped: "
            f"bars={self._bar_count}, signals={self._signal_count}, hours={self._hour_count}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates - cache closes by event timestamp."""
        instrument_id = self._resolve_bar(bar)
        if instrument_id is None:
            return

        self._bar_count += 1
        self._record_close_and_try_execute(bar.ts_event, instrument_id, float(bar.close))

    def on_data(self, data) -> None:
        """Handle FactorValues and defer execution until signal-ts prices are ready."""
        if not isinstance(data, FactorValues):
            return

        signal_ts = data.ts_event
        # Check for monthly position value update
        self._update_monthly_position_value(signal_ts)

        self._signal_count += 1
        factors = data.factors
        composite_factor = self.config.composite_factor

        # Extract composite factor values
        raw_composite = factors.get(composite_factor, {})
        if not raw_composite:
            return

        composite = {k: v for k, v in raw_composite.items() if not math.isnan(v)}
        self._hour_count += 1

        # Only rebalance every rebalance_period hours
        if self._hour_count % self.config.rebalance_period != 0:
            return

        if self._hour_count <= 20 or self._hour_count % 100 == 0:
            self.log.info(f"Hour #{self._hour_count}: Computed composite for {len(composite)} instruments")

        if len(composite) < 2 * self.config.n_positions:
            self.log.warning(
                f"Not enough instruments with valid factor values: "
                f"{len(composite)} < {2 * self.config.n_positions}"
            )
            return

        sorted_instruments = sorted(composite.items(), key=lambda x: (x[1], x[0]))
        self._enqueue_pending(signal_ts, sorted_instruments)

    def _required_instruments(self, payload: list[tuple[str, float]]) -> list[str]:
        return [inst_id for inst_id, _ in payload]

    def _on_pending_ready(
        self,
        signal_ts: int,
        payload: list[tuple[str, float]],
        execution_prices: dict[str, float],
    ) -> None:
        self._rebalance_positions_with_buffer(
            sorted_instruments=payload,
            ts_event=signal_ts,
            execution_prices=execution_prices,
        )

    def _rebalance_positions_with_buffer(
        self,
        sorted_instruments: list[tuple[str, float]],
        ts_event: int,
        execution_prices: dict[str, float],
    ) -> None:
        """
        Rebalance with buffer zone.

        Logic:
        - Step 1 (Close): Close positions that exceeded buffer threshold
          - Long: close if rank > mid
          - Short: close if rank < mid
        - Step 2 (Open): Open new positions in target zones
          - Long: open if in top N and not already long
          - Short: open if in bottom N and not already short
        """
        n = self.config.n_positions
        total = len(sorted_instruments)
        mid = int(total * self.config.buffer_ratio)

        rank = {inst_id: i for i, (inst_id, _) in enumerate(sorted_instruments)}

        long_targets = {x[0] for x in sorted_instruments[:n]}
        short_targets = {x[0] for x in sorted_instruments[-n:]}

        self._log_position_ranks(rank, total, mid, long_targets, short_targets)

        # === STEP 1: CLOSE (with buffer) ===
        for inst_id in sorted(self._long_positions):
            if inst_id in rank and rank[inst_id] > mid:
                exec_price = execution_prices.get(inst_id)
                self._close_position(inst_id, f"LONG_EXCEEDED_BUFFER_RANK_{rank[inst_id]}", exec_price=exec_price)
                self._long_positions.discard(inst_id)

        for inst_id in sorted(self._short_positions):
            if inst_id in rank and rank[inst_id] < mid:
                exec_price = execution_prices.get(inst_id)
                self._close_position(inst_id, f"SHORT_EXCEEDED_BUFFER_RANK_{rank[inst_id]}", exec_price=exec_price)
                self._short_positions.discard(inst_id)

        # === STEP 2: OPEN (target zones only) ===
        # Create composite value lookup from sorted_instruments
        composite_lookup = dict(sorted_instruments)

        if self.config.enable_long:
            for inst_id in sorted(long_targets):
                if inst_id not in self._long_positions:
                    inst_rank = rank.get(inst_id)
                    composite_val = composite_lookup.get(inst_id)
                    exec_price = execution_prices.get(inst_id)
                    if exec_price is None or exec_price <= 0:
                        self.log.warning(
                            f"Skip LONG {inst_id}: missing/invalid execution price "
                            f"for signal_ts={ts_event}"
                        )
                        continue
                    if self._open_position(
                        inst_id,
                        OrderSide.BUY,
                        exec_price=exec_price,
                        rank=inst_rank,
                        composite_value=composite_val,
                        ts_event=ts_event,
                    ):
                        self._long_positions.add(inst_id)

        if self.config.enable_short:
            for inst_id in sorted(short_targets):
                if inst_id not in self._short_positions:
                    inst_rank = rank.get(inst_id)
                    composite_val = composite_lookup.get(inst_id)
                    exec_price = execution_prices.get(inst_id)
                    if exec_price is None or exec_price <= 0:
                        self.log.warning(
                            f"Skip SHORT {inst_id}: missing/invalid execution price "
                            f"for signal_ts={ts_event}"
                        )
                        continue
                    if self._open_position(
                        inst_id,
                        OrderSide.SELL,
                        exec_price=exec_price,
                        rank=inst_rank,
                        composite_value=composite_val,
                        ts_event=ts_event,
                    ):
                        self._short_positions.add(inst_id)

        # === STEP 3: UPDATE RANK HISTORY for all current positions ===
        self._update_rank_history(rank, composite_lookup, ts_event)

    def _log_position_ranks(
        self,
        rank: dict[str, int],
        total: int,
        mid: int,
        long_targets: set[str],
        short_targets: set[str],
    ) -> None:
        """Log current position ranks for debugging."""
        if self._hour_count % 24 != 0 and self._hour_count > 20:
            return

        n = self.config.n_positions

        long_ranks = sorted([rank.get(inst, -1) for inst in self._long_positions])
        short_ranks = sorted([rank.get(inst, -1) for inst in self._short_positions])

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

        if self._hour_count <= 20:
            self.log.debug(
                f"Position ranks - Long: {long_ranks}, Short: {short_ranks}"
            )

    def _update_monthly_position_value(self, ts_event: int) -> None:
        """Update position value on the 1st of each month based on account equity."""
        if not self.config.monthly_position_update:
            return

        # Convert nanoseconds to datetime
        current_time = datetime.utcfromtimestamp(ts_event / 1_000_000_000)
        current_month = current_time.month

        # Skip if month hasn't changed
        if self._current_month == current_month:
            return

        # Get venue from first instrument
        if not self._instrument_ids:
            return
        venue = self._instrument_ids[0].venue

        # Get current account equity
        account = self.portfolio.account(venue)
        if account is None:
            return

        balance = account.balance_total(Currency.from_str("USDT"))
        if balance is None:
            return
        equity = float(balance.as_double())

        # Safety check: skip update if equity is non-positive
        if equity <= 0:
            self.log.warning(
                f"Monthly position update skipped [{current_time.strftime('%Y-%m')}]: "
                f"equity={equity:.2f} (non-positive)"
            )
            self._current_month = current_month
            return

        # Calculate new position value: equity * leverage / total_positions
        total_positions = self.config.n_positions * 2  # long + short
        self._monthly_position_value = (equity * self.config.target_leverage) / total_positions

        self._current_month = current_month
        self.log.info(
            f"Monthly position update [{current_time.strftime('%Y-%m')}]: "
            f"equity={equity:.2f}, leverage={self.config.target_leverage}, "
            f"new_position_value={self._monthly_position_value:.2f}"
        )

    def _get_position_value(self) -> float:
        """Get the current position value for opening new positions."""
        min_position_value = 100.0  # Minimum safe position value

        # Priority: monthly calculated > config fixed > default calculation
        if self.config.monthly_position_update and self._monthly_position_value is not None:
            return max(self._monthly_position_value, min_position_value)

        if self.config.position_value is not None:
            return max(self.config.position_value, min_position_value)

        # Fallback: calculate on-the-fly
        if not self._instrument_ids:
            return 2000.0  # Default fallback

        venue = self._instrument_ids[0].venue
        account = self.portfolio.account(venue)
        if account is None:
            return 2000.0

        balance = account.balance_total(Currency.from_str("USDT"))
        if balance is None:
            return 2000.0
        equity = float(balance.as_double())

        if equity <= 0:
            return min_position_value

        total_positions = self.config.n_positions * 2
        calculated = (equity * self.config.target_leverage) / total_positions
        return max(calculated, min_position_value)

    def _open_position(
        self,
        instrument_id_str: str,
        side: OrderSide,
        exec_price: float,
        rank: int | None = None,
        composite_value: float | None = None,
        ts_event: int | None = None,
    ) -> bool:
        """Open position with specified side and store metadata for reporting."""
        instrument_id = InstrumentId.from_str(instrument_id_str)
        instrument = self._instruments.get(instrument_id)

        if instrument is None:
            return False

        if exec_price <= 0:
            return False

        position_value = self._get_position_value()
        raw_qty = Decimal(str(position_value / exec_price))
        qty = instrument.make_qty(raw_qty)

        # Record position metadata via provider
        self._metadata_provider.record_open(
            instrument_id=instrument_id_str,
            side="LONG" if side == OrderSide.BUY else "SHORT",
            rank=rank,
            composite=composite_value,
            buffer_ratio=self.config.buffer_ratio,
            ts_event=ts_event or 0,
        )

        if self.config.execution_mode == "post_limit":
            self._submit_limit_open(instrument_id, side, qty, exec_price)
        else:
            self._submit_anchor_open(instrument_id, side, qty, exec_price)

        return True

    def _update_rank_history(
        self,
        rank: dict[str, int],
        composite_lookup: dict[str, float],
        ts_event: int,
    ) -> None:
        """Update rank history for all current positions."""
        all_positions = sorted(self._long_positions | self._short_positions)
        self._metadata_provider.update_rank_history(
            instrument_ids=all_positions,
            rank_lookup=rank,
            composite_lookup=composite_lookup,
            ts_event=ts_event,
        )

    def _close_position(
        self,
        instrument_id_str: str,
        reason: str,
        exec_price: float | None = None,
    ) -> None:
        """Close a position for an instrument with deterministic anchor pricing."""
        instrument_id = InstrumentId.from_str(instrument_id_str)

        # Record close in metadata provider
        self._metadata_provider.record_close(
            instrument_id=instrument_id_str,
            reason=reason,
            hour_count=self._hour_count,
        )

        if self.config.execution_mode == "post_limit":
            self._close_instrument_positions_limit(instrument_id, exec_price)
        else:
            self._close_instrument_positions(instrument_id, exec_price)
        self.log.debug(f"CLOSE {instrument_id_str}: {reason}")

    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions with deterministic fill pricing."""
        latest_closes = self._price_book.get_latest_closes()

        for position in sorted(
            self.cache.positions_open(strategy_id=self.id),
            key=lambda p: str(p.instrument_id),
        ):
            if position.is_closed:
                continue
            inst_id = str(position.instrument_id)
            self._metadata_provider.record_close(
                instrument_id=inst_id, reason=reason, hour_count=self._hour_count,
            )
            if self.config.execution_mode == "post_limit":
                self._submit_limit_close(position, latest_closes.get(inst_id))
            else:
                self._submit_anchor_close(position, latest_closes.get(inst_id))
            self.log.debug(f"CLOSE {inst_id}: {reason}")

        self._long_positions.clear()
        self._short_positions.clear()
