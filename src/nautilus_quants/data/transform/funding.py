# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tardis funding rate data loader for Nautilus Trader.

Wraps Nautilus Trader's TardisCSVDataLoader for loading funding rate data.
"""

from pathlib import Path

from nautilus_trader.adapters.tardis.loaders import TardisCSVDataLoader
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId


def load_funding_rates(
    file_path: Path | str,
    instrument_id: InstrumentId | str | None = None,
    limit: int | None = None,
) -> list[FundingRateUpdate]:
    """Load funding rate data from Tardis derivative_ticker CSV.

    Uses Nautilus Trader's TardisCSVDataLoader to parse the CSV file.

    Args:
        file_path: Path to the CSV file (derivative_ticker format).
        instrument_id: Optional InstrumentId to override CSV values.
        limit: Optional limit on number of records to load.

    Returns:
        List of FundingRateUpdate objects.

    Example:
        >>> from nautilus_quants.data.transform.funding import load_funding_rates
        >>> rates = load_funding_rates(
        ...     "data/funding/btcusdt_funding.csv",
        ...     instrument_id="BTCUSDT-PERP.BINANCE"
        ... )
        >>> print(f"Loaded {len(rates)} funding rate updates")
    """
    # Handle string instrument_id
    if isinstance(instrument_id, str):
        instrument_id = InstrumentId.from_str(instrument_id)

    loader = TardisCSVDataLoader(instrument_id=instrument_id)

    if limit:
        return loader.load_funding_rates(str(file_path), limit=limit)
    return loader.load_funding_rates(str(file_path))


def stream_funding_rates(
    file_path: Path | str,
    instrument_id: InstrumentId | str | None = None,
    chunk_size: int = 1000,
):
    """Stream funding rate data from Tardis CSV in chunks.

    Memory-efficient streaming for large files.

    Args:
        file_path: Path to the CSV file.
        instrument_id: Optional InstrumentId to override CSV values.
        chunk_size: Number of records per chunk.

    Yields:
        Lists of FundingRateUpdate objects.
    """
    if isinstance(instrument_id, str):
        instrument_id = InstrumentId.from_str(instrument_id)

    loader = TardisCSVDataLoader(instrument_id=instrument_id)

    yield from loader.stream_funding_rates(str(file_path), chunk_size=chunk_size)
