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

from nautilus_quants.data.transform.parquet import (
    DEFAULT_PRICE_PRECISION,
    DEFAULT_QUANTITY_PRECISION,
    _create_instrument,
)

logger = logging.getLogger(__name__)


@dataclass
class TardisTransformResult:
    """Result of a Tardis transform operation."""

    success: bool
    symbol: str
    files_processed: int
    total_ticks: int
    errors: list[str] = field(default_factory=list)


def _safe_int(value: object, default: int) -> int:
    """Best-effort integer coercion with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def transform_tardis_trades(
    input_dir: Path,
    catalog_path: Path,
    symbol: str,
    maker_fee: str = "0.0002",
    taker_fee: str = "0.0004",
    margin_init: str = "0.05",
    margin_maint: str = "0.025",
) -> TardisTransformResult:
    """Convert Tardis CSV.gz trade files to NautilusTrader Parquet catalog.

    Precision is automatically inferred by TardisCSVDataLoader's Rust layer
    (infer_precision() scans each CSV row for max decimal places).

    Args:
        input_dir: Directory containing Tardis CSV.gz files
                   (e.g., data/raw/tardis/binance-futures/trades/)
        catalog_path: NautilusTrader Parquet catalog directory
        symbol: Trading symbol (e.g., "BTCUSDT")
        maker_fee: Maker fee rate as string
        taker_fee: Taker fee rate as string
        margin_init: Initial margin rate as string
        margin_maint: Maintenance margin rate as string

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
    first_trade = None

    for f in files:
        try:
            trades = loader.load_trades(filepath=str(f))
            if trades:
                if first_trade is None:
                    first_trade = trades[0]
                catalog.write_data(trades)
                total_ticks += len(trades)
                logger.info(f"Loaded {len(trades)} trades from {f.name}")
        except Exception as e:
            msg = f"Failed to load {f.name}: {e}"
            logger.error(msg)
            errors.append(msg)

    # Write instrument definition (required for BacktestNode)
    if first_trade is not None:
        instrument = _create_instrument(
            symbol=f"{symbol}-PERP",
            venue="BINANCE",
            ts_init=_safe_int(getattr(first_trade, "ts_init", 0), 0),
            price_precision=_safe_int(
                getattr(getattr(first_trade, "price", None), "precision", None),
                DEFAULT_PRICE_PRECISION,
            ),
            size_precision=_safe_int(
                getattr(getattr(first_trade, "size", None), "precision", None),
                DEFAULT_QUANTITY_PRECISION,
            ),
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            margin_init=margin_init,
            margin_maint=margin_maint,
        )
        catalog.write_data([instrument])
        logger.info(f"Wrote instrument {instrument.id} to catalog")

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
    maker_fee: str = "0.0002",
    taker_fee: str = "0.0004",
    margin_init: str = "0.05",
    margin_maint: str = "0.025",
) -> TardisTransformResult:
    """Convert Tardis CSV.gz quote files to NautilusTrader Parquet catalog.

    Args:
        input_dir: Directory containing Tardis CSV.gz quote files
                   (e.g., data/raw/tardis/binance-futures/quotes/)
        catalog_path: NautilusTrader Parquet catalog directory
        symbol: Trading symbol (e.g., "ETHUSDT")
        maker_fee: Maker fee rate as string
        taker_fee: Taker fee rate as string
        margin_init: Initial margin rate as string
        margin_maint: Maintenance margin rate as string

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
    first_quote = None

    for f in files:
        try:
            quotes = loader.load_quotes(filepath=str(f))
            if quotes:
                if first_quote is None:
                    first_quote = quotes[0]
                catalog.write_data(quotes)
                total_ticks += len(quotes)
                logger.info(f"Loaded {len(quotes)} quotes from {f.name}")
        except Exception as e:
            msg = f"Failed to load {f.name}: {e}"
            logger.error(msg)
            errors.append(msg)

    # Write instrument definition (idempotent — overwrites if trades already wrote it)
    if first_quote is not None:
        instrument = _create_instrument(
            symbol=f"{symbol}-PERP",
            venue="BINANCE",
            ts_init=_safe_int(getattr(first_quote, "ts_init", 0), 0),
            price_precision=_safe_int(
                getattr(getattr(first_quote, "bid_price", None), "precision", None),
                DEFAULT_PRICE_PRECISION,
            ),
            size_precision=_safe_int(
                getattr(getattr(first_quote, "bid_size", None), "precision", None),
                DEFAULT_QUANTITY_PRECISION,
            ),
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            margin_init=margin_init,
            margin_maint=margin_maint,
        )
        catalog.write_data([instrument])
        logger.info(f"Wrote instrument {instrument.id} to catalog (from quotes)")

    return TardisTransformResult(
        success=len(errors) == 0,
        symbol=symbol,
        files_processed=len(files),
        total_ticks=total_ticks,
        errors=errors,
    )
