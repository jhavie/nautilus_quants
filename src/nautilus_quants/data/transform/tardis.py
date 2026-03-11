"""
Tardis CSV data to NautilusTrader Parquet converter.

Uses TardisCSVDataLoader (Rust-backed) to load tick-level trade and quote data
and writes to ParquetDataCatalog for backtesting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from nautilus_trader.adapters.tardis.loaders import TardisCSVDataLoader
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog

logger = logging.getLogger(__name__)


@dataclass
class TardisTransformResult:
    """Result of a Tardis transform operation."""

    success: bool
    symbol: str
    files_processed: int
    total_ticks: int
    errors: list[str] = field(default_factory=list)


def transform_tardis_trades(
    input_dir: Path,
    catalog_path: Path,
    symbol: str,
) -> TardisTransformResult:
    """Convert Tardis CSV.gz trade files to NautilusTrader Parquet catalog.

    Precision is automatically inferred by TardisCSVDataLoader's Rust layer
    (infer_precision() scans each CSV row for max decimal places).

    Args:
        input_dir: Directory containing Tardis CSV.gz files
                   (e.g., data/raw/tardis/binance-futures/trades/)
        catalog_path: NautilusTrader Parquet catalog directory
        symbol: Trading symbol (e.g., "BTCUSDT")

    Returns:
        TardisTransformResult with counts and status.
    """
    instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
    loader = TardisCSVDataLoader(instrument_id=instrument_id)
    catalog = ParquetDataCatalog(str(catalog_path))

    files = sorted(input_dir.glob(f"*_{symbol}.csv.gz"))
    if not files:
        return TardisTransformResult(
            success=False,
            symbol=symbol,
            files_processed=0,
            total_ticks=0,
            errors=[f"No CSV.gz files found for {symbol} in {input_dir}"],
        )

    total_ticks = 0
    errors: list[str] = []

    for f in files:
        try:
            trades = loader.load_trades(filepath=str(f))
            if trades:
                catalog.write_data(trades)
                total_ticks += len(trades)
                logger.info(f"Loaded {len(trades)} trades from {f.name}")
        except Exception as e:
            msg = f"Failed to load {f.name}: {e}"
            logger.error(msg)
            errors.append(msg)

    return TardisTransformResult(
        success=len(errors) == 0,
        symbol=symbol,
        files_processed=len(files),
        total_ticks=total_ticks,
        errors=errors,
    )


def transform_tardis_quotes(
    input_dir: Path,
    catalog_path: Path,
    symbol: str,
) -> TardisTransformResult:
    """Convert Tardis CSV.gz quote files to NautilusTrader Parquet catalog.

    Args:
        input_dir: Directory containing Tardis CSV.gz quote files
                   (e.g., data/raw/tardis/binance-futures/quotes/)
        catalog_path: NautilusTrader Parquet catalog directory
        symbol: Trading symbol (e.g., "ETHUSDT")

    Returns:
        TardisTransformResult with counts and status.
    """
    instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
    loader = TardisCSVDataLoader(instrument_id=instrument_id)
    catalog = ParquetDataCatalog(str(catalog_path))

    files = sorted(input_dir.glob(f"*_{symbol}.csv.gz"))
    if not files:
        return TardisTransformResult(
            success=False,
            symbol=symbol,
            files_processed=0,
            total_ticks=0,
            errors=[f"No CSV.gz files found for {symbol} in {input_dir}"],
        )

    total_ticks = 0
    errors: list[str] = []

    for f in files:
        try:
            quotes = loader.load_quotes(filepath=str(f))
            if quotes:
                catalog.write_data(quotes)
                total_ticks += len(quotes)
                logger.info(f"Loaded {len(quotes)} quotes from {f.name}")
        except Exception as e:
            msg = f"Failed to load {f.name}: {e}"
            logger.error(msg)
            errors.append(msg)

    return TardisTransformResult(
        success=len(errors) == 0,
        symbol=symbol,
        files_processed=len(files),
        total_ticks=total_ticks,
        errors=errors,
    )
