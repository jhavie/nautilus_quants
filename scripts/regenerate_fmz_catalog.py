#!/usr/bin/env python
"""
Regenerate FMZ catalog data using quote_volume (USDT turnover) instead of base volume.

This script rebuilds the Nautilus catalog for FMZ strategy with the correct volume data.
FMZ original strategy uses USDT turnover (quote_volume) not base quantity (volume).
"""

import os
import sys
from pathlib import Path

# Add nautilus_quants to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nautilus_quants.data.transform.parquet import transform_to_parquet

# Paths
RAW_DATA_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/fmz_data/raw")
PROCESSED_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/fmz_data/processed")
CATALOG_PATH = Path("/Users/joe/Sync/nautilus_quants2/data/fmz_data/catalog_quote_volume")

# Symbols to process (from backtest config)
SYMBOLS = [
    "1000SHIBUSDT", "1000XECUSDT", "1INCHUSDT", "AAVEUSDT", "ADAUSDT",
    "ALGOUSDT", "ALICEUSDT", "ALPHAUSDT", "ANKRUSDT", "ANTUSDT",
    "ARPAUSDT", "ARUSDT", "ATAUSDT", "ATOMUSDT", "AUDIOUSDT",
    "AVAXUSDT", "AXSUSDT", "BAKEUSDT", "BALUSDT", "BANDUSDT",
    "BATUSDT", "BCHUSDT", "BELUSDT", "BLZUSDT", "BNBUSDT",
    "BTCDOMUSDT", "BTCUSDT", "BTSUSDT", "C98USDT", "CELOUSDT",
    "CELRUSDT", "CHRUSDT", "CHZUSDT", "COMPUSDT", "COTIUSDT",
    "CRVUSDT", "CTSIUSDT", "DASHUSDT", "DEFIUSDT", "DENTUSDT",
    "DGBUSDT", "DOGEUSDT", "DOTUSDT", "DUSKUSDT", "DYDXUSDT",
    "EGLDUSDT", "ENJUSDT", "ENSUSDT", "EOSUSDT", "ETCUSDT",
    "ETHUSDT", "FILUSDT", "FLMUSDT", "FTMUSDT", "GALAUSDT",
    "GRTUSDT", "GTCUSDT", "HBARUSDT", "HNTUSDT", "HOTUSDT",
    "ICXUSDT", "IOSTUSDT", "IOTAUSDT", "IOTXUSDT", "KAVAUSDT",
    "KLAYUSDT", "KNCUSDT", "KSMUSDT", "LINAUSDT", "LINKUSDT",
    "LPTUSDT", "LRCUSDT", "LTCUSDT", "MANAUSDT", "MASKUSDT",
    "MATICUSDT", "MKRUSDT", "MTLUSDT", "NEARUSDT", "NEOUSDT",
    "NKNUSDT", "OCEANUSDT", "OGNUSDT", "OMGUSDT", "ONEUSDT",
    "ONTUSDT", "PEOPLEUSDT", "QTUMUSDT", "RAYUSDT", "REEFUSDT",
    "RENUSDT", "RLCUSDT", "ROSEUSDT", "RSRUSDT", "RUNEUSDT",
    "RVNUSDT", "SANDUSDT", "SCUSDT", "SFPUSDT", "SKLUSDT",
    "SNXUSDT", "SOLUSDT", "SRMUSDT", "STMXUSDT", "STORJUSDT",
    "SUSHIUSDT", "SXPUSDT", "THETAUSDT", "TOMOUSDT", "TRBUSDT",
    "TRXUSDT", "UNFIUSDT", "UNIUSDT", "VETUSDT", "WAVESUSDT",
    "XEMUSDT", "XLMUSDT", "XMRUSDT", "XRPUSDT", "XTZUSDT",
    "YFIUSDT", "ZECUSDT", "ZENUSDT", "ZILUSDT", "ZRXUSDT",
]

TIMEFRAME = "1h"


def main():
    """Regenerate catalog with quote_volume."""
    print("=" * 80)
    print("REGENERATING FMZ CATALOG WITH QUOTE_VOLUME")
    print("=" * 80)

    # Create catalog directory
    CATALOG_PATH.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    errors = []

    for i, symbol in enumerate(SYMBOLS):
        # Find processed CSV file (try different naming patterns)
        csv_path = PROCESSED_PATH / "binance" / symbol / TIMEFRAME / f"{symbol}_{TIMEFRAME}_processed.csv"

        if not csv_path.exists():
            # Try alternative naming
            csv_path = PROCESSED_PATH / "binance" / symbol / TIMEFRAME / f"{symbol}_{TIMEFRAME}_20220101_20220914.csv"

        if not csv_path.exists():
            # Try raw data directly
            csv_path = RAW_DATA_PATH / "binance" / symbol / TIMEFRAME / f"{symbol}_{TIMEFRAME}_20220101_20220914.csv"

        if not csv_path.exists():
            print(f"[{i+1}/{len(SYMBOLS)}] {symbol}: CSV not found")
            errors.append(f"{symbol}: CSV not found at {csv_path}")
            error_count += 1
            continue

        try:
            result = transform_to_parquet(
                input_path=csv_path,
                catalog_path=CATALOG_PATH,
                symbol=symbol,
                timeframe=TIMEFRAME,
                merge=False,  # Fresh catalog
                raw_data_path=RAW_DATA_PATH,
                maker_fee="0.0002",
                taker_fee="0.0002",
                use_quote_volume=True,  # Use USDT turnover
            )

            if result.success:
                print(f"[{i+1}/{len(SYMBOLS)}] {symbol}: OK ({result.rows_transformed} rows)")
                success_count += 1
            else:
                print(f"[{i+1}/{len(SYMBOLS)}] {symbol}: FAILED - {result.errors}")
                errors.append(f"{symbol}: {result.errors}")
                error_count += 1

        except Exception as e:
            print(f"[{i+1}/{len(SYMBOLS)}] {symbol}: ERROR - {e}")
            errors.append(f"{symbol}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {success_count}/{len(SYMBOLS)}")
    print(f"Errors: {error_count}/{len(SYMBOLS)}")
    print(f"Catalog path: {CATALOG_PATH}")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
