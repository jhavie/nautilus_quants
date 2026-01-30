#!/usr/bin/env python
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Demo backtest with funding rate data.

This script demonstrates how to integrate Tardis funding rate data
into a Nautilus Trader backtest.

Usage:
    python scripts/demo_backtest_with_funding.py
"""

from decimal import Decimal
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common.component import TimeEvent
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.data.transform.funding import load_funding_rates


def create_btcusdt_perpetual() -> CryptoPerpetual:
    """Create BTCUSDT perpetual instrument definition."""
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=3,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(10, USDT),
        max_price=Price.from_str("1000000"),
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("0.01"),
        margin_maint=Decimal("0.005"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
    )


class FundingRateMonitorConfig(StrategyConfig, frozen=True):
    """Configuration for FundingRateMonitor strategy.

    Parameters
    ----------
    instrument_id : str
        The instrument ID to monitor.
    """

    instrument_id: str = "BTCUSDT-PERP.BINANCE"


class FundingRateMonitor(Strategy):
    """Simple strategy that monitors funding rate updates.

    This is a demo strategy that just logs funding rate changes.
    In a real strategy, you would use this to:
    - Adjust positions before funding settlements
    - Trade based on funding rate arbitrage
    - Calculate expected funding costs

    Note: FundingRateUpdate data is automatically received via on_data
    when added to the backtest engine.
    """

    def __init__(self, config: FundingRateMonitorConfig) -> None:
        super().__init__(config)
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._funding_updates_count = 0
        self._last_rate: Decimal | None = None

    def on_start(self) -> None:
        """Called when strategy starts."""
        self.log.info("FundingRateMonitor started")
        self.log.info(f"Monitoring instrument: {self._instrument_id}")
        # FundingRateUpdate data is received automatically when added to engine

    def on_data(self, data) -> None:
        """Called when any custom data is received."""
        if isinstance(data, FundingRateUpdate):
            self._funding_updates_count += 1

            # Only log significant rate changes or first update
            if self._last_rate is None or data.rate != self._last_rate:
                self.log.info(
                    f"Funding rate update: {data.rate:.8f} "
                    f"(ts_event={data.ts_event})"
                )
                self._last_rate = data.rate

    def on_stop(self) -> None:
        """Called when strategy stops."""
        self.log.info(
            f"FundingRateMonitor stopped. "
            f"Total updates received: {self._funding_updates_count}"
        )


def main():
    print("=" * 60)
    print("Demo: Backtest with Funding Rate Data")
    print("=" * 60)
    print()

    # Check if funding rate data exists
    funding_file = Path("data/funding/btcusdt_funding_202501.csv")
    if not funding_file.exists():
        print(f"Error: Funding rate data not found at {funding_file}")
        print("Please run scripts/download_funding_http.py first")
        return

    # Create engine
    print("Creating backtest engine...")
    config = BacktestEngineConfig(
        trader_id="FUNDING-DEMO-001",
    )
    engine = BacktestEngine(config=config)

    # Add venue
    print("Adding venue...")
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(100_000, USDT)],
    )

    # Add instrument
    print("Adding instrument...")
    instrument = create_btcusdt_perpetual()
    engine.add_instrument(instrument)

    # Load and add funding rate data
    print("Loading funding rate data...")
    funding_rates = load_funding_rates(
        funding_file,
        instrument_id=instrument.id,
    )
    print(f"Loaded {len(funding_rates)} funding rate updates")

    print("Adding funding rate data to engine...")
    engine.add_data(funding_rates)

    # Add strategy
    print("Adding strategy...")
    strategy_config = FundingRateMonitorConfig(
        instrument_id=str(instrument.id),
    )
    strategy = FundingRateMonitor(config=strategy_config)
    engine.add_strategy(strategy)

    # Run backtest
    print()
    print("=" * 60)
    print("Running backtest...")
    print("=" * 60)
    print()

    engine.run()

    # Print results
    print()
    print("=" * 60)
    print("Backtest complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
