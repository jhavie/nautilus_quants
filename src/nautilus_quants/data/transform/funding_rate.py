# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Transform module for converting Funding Rate CSV to NautilusTrader Parquet."""

from __future__ import annotations

import logging
import traceback
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog

logger = logging.getLogger(__name__)


def csv_to_funding_rate_updates(
    csv_path: Path | str,
    symbol: str,
    venue: str = "BINANCE",
) -> list[FundingRateUpdate]:
    """Convert funding rate CSV to NautilusTrader FundingRateUpdate objects.

    CSV format: timestamp (ms), funding_rate

    Each row becomes a FundingRateUpdate with:
        - instrument_id from symbol.venue
        - rate as Decimal
        - ts_event / ts_init converted from ms to ns

    Note: next_funding_ns is not available from historical data,
    so it is omitted (defaults to None).

    Args:
        csv_path: Path to the funding rate CSV file.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        venue: Exchange venue name (default "BINANCE").

    Returns:
        List of FundingRateUpdate objects.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if df.empty:
        return []

    instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")
    updates: list[FundingRateUpdate] = []

    for _, row in df.iterrows():
        ts_ns = int(row["timestamp"]) * 1_000_000  # ms -> ns

        fru = FundingRateUpdate(
            instrument_id=instrument_id,
            rate=Decimal(str(row["funding_rate"])),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        updates.append(fru)

    return updates


def transform_funding_rates(
    raw_dir: Path | str,
    catalog_path: Path | str,
    symbols: list[str],
    venue: str = "BINANCE",
) -> list[dict]:
    """Transform funding rate CSVs for all symbols and write to Parquet catalog.

    For each symbol:
    1. Find FR CSV in {raw_dir}/{symbol}/funding_rate/
    2. Convert to FundingRateUpdate objects
    3. Write to catalog via catalog.write_data()

    Args:
        raw_dir: Root raw data directory (e.g., data/raw/binance).
        catalog_path: NautilusTrader Parquet catalog directory.
        symbols: List of trading pair symbols.
        venue: Exchange venue name (default "BINANCE").

    Returns:
        List of dicts with keys: symbol, count, success, and optionally error.
    """
    raw_dir = Path(raw_dir)
    catalog_path = Path(catalog_path)
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path))

    results: list[dict] = []

    for symbol in symbols:
        fr_dir = raw_dir / symbol / "funding_rate"

        if not fr_dir.exists():
            logger.warning("No funding_rate directory for %s at %s", symbol, fr_dir)
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": f"Directory not found: {fr_dir}",
                }
            )
            continue

        csv_files = sorted(fr_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found for %s in %s", symbol, fr_dir)
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": f"No CSV files in {fr_dir}",
                }
            )
            continue

        try:
            all_updates: list[FundingRateUpdate] = []
            for csv_file in csv_files:
                updates = csv_to_funding_rate_updates(csv_file, symbol, venue)
                all_updates.extend(updates)

            if not all_updates:
                results.append(
                    {
                        "symbol": symbol,
                        "count": 0,
                        "success": False,
                        "error": "No data rows in CSV files",
                    }
                )
                continue

            catalog.write_data(all_updates)
            logger.info(
                "Wrote %d FundingRateUpdate records for %s",
                len(all_updates),
                symbol,
            )
            results.append(
                {
                    "symbol": symbol,
                    "count": len(all_updates),
                    "success": True,
                }
            )

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            logger.error("Failed to transform funding rates for %s: %s", symbol, msg)
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": msg,
                }
            )

    return results
