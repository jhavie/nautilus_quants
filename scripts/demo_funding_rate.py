#!/usr/bin/env python
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Demo script for downloading and loading Tardis funding rate data.

This script demonstrates the complete workflow:
1. Download derivative_ticker data from Tardis
2. Merge CSV files
3. Load into Nautilus FundingRateUpdate format
4. Verify data is ready for backtesting

Usage:
    python scripts/demo_funding_rate.py
"""

import os
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nautilus_quants.data.download.tardis import (
    TardisFundingConfig,
    download_derivative_ticker,
    merge_funding_csvs,
)
from nautilus_quants.data.transform.funding import load_funding_rates


def main():
    # Configuration
    API_KEY = "TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"

    config = TardisFundingConfig(
        exchange="binance-futures",
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-01-31",
        api_key=API_KEY,
    )

    data_dir = Path("data/funding")
    merged_file = data_dir / "btcusdt_funding_202501.csv"

    # Step 1: Download data
    print("=" * 60)
    print("Step 1: Downloading derivative_ticker data from Tardis")
    print("=" * 60)
    print(f"  Exchange: {config.exchange}")
    print(f"  Symbols: {config.symbols}")
    print(f"  Date range: {config.start_date} to {config.end_date}")
    print()

    result = download_derivative_ticker(config, data_dir)

    if not result.success:
        print(f"Download failed: {result.error}")
        return

    print(f"Downloaded {result.file_count} files to {result.output_dir}")
    print()

    # Step 2: Merge CSV files
    print("=" * 60)
    print("Step 2: Merging CSV files")
    print("=" * 60)

    try:
        row_count = merge_funding_csvs(data_dir, merged_file, symbol="BTCUSDT")
        print(f"Merged to: {merged_file}")
        print(f"Total rows: {row_count}")
    except FileNotFoundError as e:
        print(f"Merge failed: {e}")
        return

    print()

    # Step 3: Load into Nautilus format
    print("=" * 60)
    print("Step 3: Loading into Nautilus FundingRateUpdate format")
    print("=" * 60)

    funding_rates = load_funding_rates(
        merged_file,
        instrument_id="BTCUSDT-PERP.BINANCE",
    )

    print(f"Loaded {len(funding_rates)} FundingRateUpdate records")
    print()

    if funding_rates:
        print("Sample records:")
        print(f"  First: {funding_rates[0]}")
        print(f"  Last:  {funding_rates[-1]}")
        print()

        # Show rate statistics
        rates = [fr.rate for fr in funding_rates]
        print("Funding rate statistics:")
        print(f"  Min:  {min(rates):.8f}")
        print(f"  Max:  {max(rates):.8f}")
        print(f"  Mean: {sum(rates) / len(rates):.8f}")

    print()
    print("=" * 60)
    print("Data is ready for backtesting!")
    print("=" * 60)
    print()
    print("Usage in backtest:")
    print("  from nautilus_quants.data.transform import load_funding_rates")
    print("  rates = load_funding_rates('data/funding/btcusdt_funding_202501.csv')")
    print("  engine.add_data(rates)")


if __name__ == "__main__":
    main()
