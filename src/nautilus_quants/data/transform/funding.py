# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Binance funding rate CSV loader for Nautilus Trader backtesting.

This module provides a loader for Binance funding rate CSV files
(downloaded via BinanceFundingDownloader) to convert them into
Nautilus Trader's FundingRateUpdate data type.

The Binance CSV format:
    symbol,funding_time,funding_rate,mark_price
    ETHUSDT,1735689600000,0.0001,3500.25

Note: Nautilus Trader's built-in TardisCSVDataLoader.load_funding_rates()
requires Tardis derivative_ticker format, which is incompatible with
Binance's funding rate API format.
"""

from decimal import Decimal
from pathlib import Path

from nautilus_trader.core.datetime import millis_to_nanos
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId


def load_funding_rates(
    file_path: Path | str,
    instrument_id: InstrumentId | str,
) -> list[FundingRateUpdate]:
    """Load funding rates from Binance funding rate CSV.

    Args:
        file_path: Path to CSV file (format: symbol,funding_time,funding_rate,mark_price)
        instrument_id: Nautilus instrument ID for the funding rates

    Returns:
        List of FundingRateUpdate objects sorted by timestamp
    """
    if isinstance(instrument_id, str):
        instrument_id = InstrumentId.from_str(instrument_id)

    file_path = Path(file_path)
    results = []

    with open(file_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue

            funding_time_ms = int(parts[1])
            rate = Decimal(parts[2])

            # Convert milliseconds to nanoseconds
            ts_ns = millis_to_nanos(funding_time_ms)

            update = FundingRateUpdate(
                instrument_id=instrument_id,
                rate=rate,
                next_funding_ns=None,  # Not provided by Binance historical API
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            results.append(update)

    return sorted(results, key=lambda x: x.ts_event)
