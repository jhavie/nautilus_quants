"""
Transform module for converting processed CSV to Nautilus Parquet format.

Uses Nautilus Trader's native ParquetDataCatalog for compatibility.
"""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog


# Default precision values when exchange info is not available
DEFAULT_PRICE_PRECISION = 8
DEFAULT_QUANTITY_PRECISION = 8


@dataclass
class TransformResult:
    """Result of a transform operation."""

    success: bool
    symbol: str
    timeframe: str
    input_file: str
    output_path: str
    rows_transformed: int
    errors: list[str] = field(default_factory=list)


# Timeframe to Nautilus bar aggregation mapping
TIMEFRAME_TO_STEP = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def _load_symbol_precision(
    symbol: str,
    raw_data_path: Path | str | None,
) -> tuple[int, int]:
    """Load price and quantity precision from exchange info JSON.

    Args:
        symbol: Trading pair symbol (e.g., "1000PEPEUSDT")
        raw_data_path: Path to raw data directory containing .exchange_info/

    Returns:
        Tuple of (price_precision, quantity_precision)
    """
    if raw_data_path is None:
        return DEFAULT_PRICE_PRECISION, DEFAULT_QUANTITY_PRECISION

    raw_data_path = Path(raw_data_path)
    json_path = raw_data_path / ".exchange_info" / f"{symbol}_precision.json"

    if not json_path.exists():
        return DEFAULT_PRICE_PRECISION, DEFAULT_QUANTITY_PRECISION

    try:
        with open(json_path) as f:
            data = json.load(f)
        return (
            data.get("pricePrecision", DEFAULT_PRICE_PRECISION),
            data.get("quantityPrecision", DEFAULT_QUANTITY_PRECISION),
        )
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PRICE_PRECISION, DEFAULT_QUANTITY_PRECISION


def _create_instrument(
    symbol: str,
    venue: str = "BINANCE",
    ts_init: int = 0,
    price_precision: int = DEFAULT_PRICE_PRECISION,
    size_precision: int = DEFAULT_QUANTITY_PRECISION,
) -> CryptoPerpetual:
    """Create a CryptoPerpetual instrument for the catalog.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        venue: Exchange venue name
        ts_init: Initialization timestamp in nanoseconds
        price_precision: Number of decimal places for price
        size_precision: Number of decimal places for quantity

    Returns:
        CryptoPerpetual instrument
    """
    # Dynamically compute increment strings based on precision
    if price_precision > 0:
        price_increment_str = "0." + "0" * (price_precision - 1) + "1"
    else:
        price_increment_str = "1"

    if size_precision > 0:
        size_increment_str = "0." + "0" * (size_precision - 1) + "1"
    else:
        size_increment_str = "1"

    return CryptoPerpetual(
        instrument_id=InstrumentId(symbol=Symbol(symbol), venue=Venue(venue)),
        raw_symbol=Symbol(symbol),
        base_currency=USDT,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_increment_str),
        size_increment=Quantity.from_str(size_increment_str),
        max_quantity=Quantity.from_str("10000000"),
        min_quantity=Quantity.from_str(size_increment_str),
        max_notional=Money(1_000_000_000, USDT),
        min_notional=Money(1, USDT),
        max_price=Price.from_str("10000000"),
        min_price=Price.from_str(price_increment_str),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=ts_init,
        ts_init=ts_init,
    )


def _get_bar_type(symbol: str, timeframe: str) -> BarType:
    """Create BarType for the given symbol and timeframe.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: K-line interval (e.g., "1h")

    Returns:
        BarType for the symbol and timeframe
    """
    instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")

    # Determine aggregation type and step
    if timeframe.endswith("m"):
        aggregation = BarAggregation.MINUTE
        step = int(timeframe[:-1])
    elif timeframe.endswith("h"):
        aggregation = BarAggregation.HOUR
        step = int(timeframe[:-1])
    elif timeframe.endswith("d"):
        aggregation = BarAggregation.DAY
        step = int(timeframe[:-1])
    else:
        aggregation = BarAggregation.MINUTE
        step = 60

    # Create BarSpecification
    bar_spec = BarSpecification(
        step=step,
        aggregation=aggregation,
        price_type=PriceType.LAST,
    )

    # Create BarType
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
    )

    return bar_type


def csv_to_bars(
    csv_path: Path | str,
    symbol: str,
    timeframe: str,
    price_precision: int = DEFAULT_PRICE_PRECISION,
    quantity_precision: int = DEFAULT_QUANTITY_PRECISION,
) -> list[Bar]:
    """Convert CSV data to Nautilus Bar objects.

    Args:
        csv_path: Path to processed CSV file
        symbol: Trading pair symbol
        timeframe: K-line interval
        price_precision: Number of decimal places for price formatting
        quantity_precision: Number of decimal places for quantity formatting

    Returns:
        List of Bar objects
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    bar_type = _get_bar_type(symbol, timeframe)
    bars = []

    for _, row in df.iterrows():
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['open']:.{price_precision}f}"),
            high=Price.from_str(f"{row['high']:.{price_precision}f}"),
            low=Price.from_str(f"{row['low']:.{price_precision}f}"),
            close=Price.from_str(f"{row['close']:.{price_precision}f}"),
            volume=Quantity.from_str(f"{row['volume']:.{quantity_precision}f}"),
            ts_event=int(row["timestamp"]) * 1_000_000,  # Convert ms to ns
            ts_init=int(row["timestamp"]) * 1_000_000,
        )
        bars.append(bar)

    return bars


def transform_to_parquet(
    input_path: Path | str,
    catalog_path: Path | str,
    symbol: str,
    timeframe: str,
    merge: bool = True,
    raw_data_path: Path | str | None = None,
) -> TransformResult:
    """Transform processed CSV to Nautilus Parquet format.

    Args:
        input_path: Path to processed CSV file
        catalog_path: Parquet catalog directory
        symbol: Trading pair symbol
        timeframe: K-line interval
        merge: Merge with existing data if present
        raw_data_path: Path to raw data directory for precision lookup

    Returns:
        TransformResult with output path and row count
    """
    input_path = Path(input_path)
    catalog_path = Path(catalog_path)

    try:
        # Load precision from exchange info
        price_precision, quantity_precision = _load_symbol_precision(
            symbol, raw_data_path
        )

        # Load and convert to bars
        bars = csv_to_bars(
            input_path, symbol, timeframe, price_precision, quantity_precision
        )

        if not bars:
            return TransformResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                input_file=str(input_path),
                output_path=str(catalog_path),
                rows_transformed=0,
                errors=["No data to transform"],
            )

        # Create catalog and write data
        catalog_path.mkdir(parents=True, exist_ok=True)
        catalog = ParquetDataCatalog(str(catalog_path))

        # Write instrument definition first (required for BacktestNode)
        instrument = _create_instrument(
            symbol=symbol,
            venue="BINANCE",
            ts_init=bars[0].ts_init,
            price_precision=price_precision,
            size_precision=quantity_precision,
        )
        catalog.write_data([instrument])

        # Write bars to catalog
        catalog.write_data(bars)

        return TransformResult(
            success=True,
            symbol=symbol,
            timeframe=timeframe,
            input_file=str(input_path),
            output_path=str(catalog_path),
            rows_transformed=len(bars),
        )

    except Exception as e:
        return TransformResult(
            success=False,
            symbol=symbol,
            timeframe=timeframe,
            input_file=str(input_path),
            output_path=str(catalog_path),
            rows_transformed=0,
            errors=[str(e)],
        )
