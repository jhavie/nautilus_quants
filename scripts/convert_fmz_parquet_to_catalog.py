#!/usr/bin/env python
"""
Convert FMZ parquet data (2023-2025) to Nautilus Trader catalog format.

FMZ parquet structure:
- volume = USDT turnover (quote_volume) - this is what we want
- amount = base quantity

This script reads the FMZ parquet and creates Nautilus catalog bars.
"""

import sys
from pathlib import Path

import pandas as pd

# Add nautilus_quants to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decimal import Decimal

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

# Paths
FMZ_PARQUET = Path("/Users/joe/Sync/strategy_research/16_fmz_research/data/binance_futures_1h_2023_2025.parquet")
CATALOG_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/fmz_data/catalog_2023_2025")
PRECISION_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/fmz_data/raw/.exchange_info")


def load_precision(symbol: str) -> tuple[int, int]:
    """Load precision from exchange info."""
    json_path = PRECISION_PATH / f"{symbol}_precision.json"
    if json_path.exists():
        import json
        with open(json_path) as f:
            data = json.load(f)
        return data.get("pricePrecision", 8), data.get("quantityPrecision", 8)
    return 8, 8


def create_instrument(symbol: str, ts_init: int, price_precision: int, size_precision: int) -> CryptoPerpetual:
    """Create a CryptoPerpetual instrument."""
    if price_precision > 0:
        price_increment_str = "0." + "0" * (price_precision - 1) + "1"
    else:
        price_increment_str = "1"

    if size_precision > 0:
        size_increment_str = "0." + "0" * (size_precision - 1) + "1"
    else:
        size_increment_str = "1"

    return CryptoPerpetual(
        instrument_id=InstrumentId(symbol=Symbol(symbol), venue=Venue("BINANCE")),
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
        taker_fee=Decimal("0.0002"),
        ts_event=ts_init,
        ts_init=ts_init,
    )


def main():
    """Convert FMZ parquet to Nautilus catalog."""
    print("=" * 80)
    print("CONVERTING FMZ PARQUET TO NAUTILUS CATALOG")
    print("=" * 80)
    print(f"Source: {FMZ_PARQUET}")
    print(f"Target: {CATALOG_PATH}")

    # Load FMZ data
    print("\nLoading FMZ parquet data...")
    df_all = pd.read_parquet(FMZ_PARQUET)
    symbols = df_all["symbol"].unique().tolist()
    print(f"Found {len(symbols)} symbols")
    print(f"Date range: {df_all.index.min()} ~ {df_all.index.max()}")

    # Create catalog
    CATALOG_PATH.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(CATALOG_PATH))

    success_count = 0
    error_count = 0
    errors = []

    for i, symbol in enumerate(symbols):
        try:
            # Extract symbol data
            df_sym = df_all[df_all["symbol"] == symbol].copy()
            df_sym = df_sym.drop("symbol", axis=1)

            if df_sym.empty:
                continue

            # FMZ parquet: volume = USDT turnover (quote_volume)
            # Prepare DataFrame for BarDataWrangler
            df_bars = pd.DataFrame({
                "open": df_sym["open"].astype(float),
                "high": df_sym["high"].astype(float),
                "low": df_sym["low"].astype(float),
                "close": df_sym["close"].astype(float),
                "volume": df_sym["volume"].astype(float),  # This is USDT turnover
            }, index=df_sym.index)

            # Ensure UTC timezone
            if df_bars.index.tz is None:
                df_bars.index = df_bars.index.tz_localize("UTC")

            # Load precision
            price_prec, size_prec = load_precision(symbol)

            # Create bar type
            instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")
            bar_spec = BarSpecification(
                step=1,
                aggregation=BarAggregation.HOUR,
                price_type=PriceType.LAST,
            )
            bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)

            # Create instrument
            ts_init = int(df_bars.index[0].value)
            instrument = create_instrument(symbol, ts_init, price_prec, size_prec)

            # Use BarDataWrangler
            wrangler = BarDataWrangler(bar_type, instrument)
            bars = wrangler.process(df_bars)

            if not bars:
                errors.append(f"{symbol}: No bars generated")
                error_count += 1
                continue

            # Write to catalog
            catalog.write_data([instrument])
            catalog.write_data(bars)

            print(f"[{i+1}/{len(symbols)}] {symbol}: OK ({len(bars)} bars)")
            success_count += 1

        except Exception as e:
            print(f"[{i+1}/{len(symbols)}] {symbol}: ERROR - {e}")
            errors.append(f"{symbol}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {success_count}/{len(symbols)}")
    print(f"Errors: {error_count}/{len(symbols)}")
    print(f"Catalog path: {CATALOG_PATH}")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
