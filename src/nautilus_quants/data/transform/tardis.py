"""
Tardis CSV data to NautilusTrader Parquet converter.

Uses TardisCSVDataLoader (Rust-backed) to load tick-level trade and quote data
and writes to ParquetDataCatalog for backtesting.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from nautilus_trader.adapters.tardis.loaders import TardisCSVDataLoader
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from tqdm import tqdm

from nautilus_quants.data.transform.parquet import _create_instrument

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
            ts_init=first_trade.ts_init,
            price_precision=first_trade.price.precision,
            size_precision=first_trade.size.precision,
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
            ts_init=first_quote.ts_init,
            price_precision=first_quote.bid_price.precision,
            size_precision=first_quote.bid_size.precision,
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


def transform_all(
    raw_data_dir: Path,
    catalog_path: Path,
    exchange: str,
    symbols: list[str],
    data_types: list[str],
    maker_fee: str = "0.0002",
    taker_fee: str = "0.0004",
    margin_init: str = "0.05",
    margin_maint: str = "0.025",
    max_workers: int = 3,
) -> list[TardisTransformResult]:
    """Transform all (symbol, data_type) pairs with concurrent workers.

    Each worker gets its own TardisCSVDataLoader and ParquetDataCatalog
    instance.  Different symbols write to different Parquet partitions
    so no locking is needed.

    Args:
        raw_data_dir: Root raw data directory (e.g., data/raw/tardis)
        catalog_path: NautilusTrader Parquet catalog directory
        exchange: Exchange identifier (e.g., "binance-futures")
        symbols: List of symbols to transform
        data_types: List of data types ("trades", "quotes")
        maker_fee: Maker fee rate
        taker_fee: Taker fee rate
        margin_init: Initial margin rate
        margin_maint: Maintenance margin rate
        max_workers: Maximum concurrent transform threads

    Returns:
        List of TardisTransformResult, one per (symbol, data_type).
    """
    transform_map = {
        "trades": transform_tardis_trades,
        "quotes": transform_tardis_quotes,
    }

    work_units = [
        (sym, dt) for sym in symbols for dt in data_types if dt in transform_map
    ]

    bar = tqdm(total=len(work_units), desc="Transforming", unit="task")
    results: list[TardisTransformResult] = []

    def _run(sym: str, data_type: str) -> TardisTransformResult:
        input_dir = Path(raw_data_dir) / exchange / data_type
        return transform_map[data_type](
            input_dir=input_dir,
            catalog_path=catalog_path,
            symbol=sym,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            margin_init=margin_init,
            margin_maint=margin_maint,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run, sym, dt): (sym, dt) for sym, dt in work_units
        }

        for future in as_completed(futures):
            sym, dt = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = TardisTransformResult(
                    success=False,
                    symbol=f"{sym}/{dt}",
                    files_processed=0,
                    total_ticks=0,
                    errors=[str(e)],
                )

            results.append(result)
            status = "\u2713" if result.success else "\u2717"
            detail = f"{result.total_ticks} ticks" if result.success else result.errors
            tqdm.write(f"  {status} {sym}/{dt}: {detail}")
            bar.update(1)

    bar.close()
    return results
