"""EquitySnapshotActor for periodic mark-to-market equity sampling in backtests.

Uses the Portfolio.unrealized_pnls() bar-close fallback chain
to build an equity series with unrealized PnL and store it in cache
for downstream reporting.

Usage (YAML ``engine.actors`` config):
    - actor_path: "nautilus_quants.actors.equity_snapshot:EquitySnapshotActor"
      config_path: "nautilus_quants.actors.equity_snapshot:EquitySnapshotActorConfig"
      config:
        interval: "8h"
        venue_name: "BINANCE"
        currency: "USDT"
"""

from __future__ import annotations

import pickle

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency

from nautilus_quants.backtest.protocols import EQUITY_SNAPSHOTS_CACHE_KEY
from nautilus_quants.backtest.utils.bar_spec import parse_interval_to_timedelta
from nautilus_quants.backtest.utils.equity import compute_mtm_equity


class EquitySnapshotActorConfig(ActorConfig, frozen=True):
    """Configuration for EquitySnapshotActor.

    Parameters
    ----------
    interval : str, default "8h"
        Sampling interval (e.g., "1h", "4h", "8h", "1d").
    venue_name : str, default "SIM"
        Venue name.
    currency : str, default "USD"
        Settlement currency code.
    """

    interval: str = "8h"
    venue_name: str = "SIM"
    currency: str = "USD"


class EquitySnapshotActor(Actor):
    """Periodically sample mark-to-market equity during backtests.

    Uses the Portfolio.unrealized_pnls() bar-close fallback chain
    (QuoteTick -> TradeTick -> bar close) to build an equity series
    with unrealized PnL and store it in cache for report generation.
    """

    def __init__(self, config: EquitySnapshotActorConfig) -> None:
        super().__init__(config)
        self._equity_points: list[tuple[int, float]] = []
        self._venue = Venue(config.venue_name)
        self._currency = Currency.from_str(config.currency)

    def on_start(self) -> None:
        """Register a timer that samples equity at ``interval``."""
        interval_td = parse_interval_to_timedelta(self.config.interval)
        self.clock.set_timer(
            name="equity_snapshot",
            interval=interval_td,
            callback=self._on_snapshot,
        )
        self.log.info(
            f"EquitySnapshotActor started: interval={self.config.interval}, "
            f"venue={self._venue}, currency={self._currency}"
        )

    def _on_snapshot(self, event: object) -> None:
        """Timer callback: compute balance + unrealized PnL and record a point."""
        equity = self._compute_equity()
        if equity is None:
            return
        ts_event = event.ts_event if hasattr(event, "ts_event") else 0
        self._equity_points.append((ts_event, equity))

    def on_stop(self) -> None:
        """Collect final snapshot and persist captured points to cache."""
        # Collect a final equity point at actor stop time.
        equity = self._compute_equity()
        if equity is not None:
            self._equity_points.append((self.clock.timestamp_ns(), equity))

        if self._equity_points:
            self.cache.add(
                EQUITY_SNAPSHOTS_CACHE_KEY,
                pickle.dumps(self._equity_points),
            )
            self.log.info(
                f"EquitySnapshotActor: {len(self._equity_points)} equity points "
                f"saved to cache"
            )
        else:
            self.log.warning("EquitySnapshotActor: no equity points captured")

    def _compute_equity(self) -> float | None:
        """Compute current MTM equity.

        Returns ``None`` for non-positive values to avoid extreme ``pct_change``.
        """
        equity = compute_mtm_equity(self.portfolio, self._venue, self._currency)
        if equity is None or equity <= 0:
            return None
        return equity
