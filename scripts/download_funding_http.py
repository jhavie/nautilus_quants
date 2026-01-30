#!/usr/bin/env python
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Download Tardis derivative_ticker data using direct HTTP API.

Uses requests library instead of tardis-dev client for more control.

Usage:
    python scripts/download_funding_http.py
"""

import gzip
from datetime import datetime, timedelta
from pathlib import Path

import requests


def download_derivative_ticker_http(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    api_key: str | None = None,
) -> list[Path]:
    """Download derivative_ticker data using direct HTTP API.

    Args:
        exchange: Tardis exchange identifier (e.g., "binance-futures")
        symbol: Symbol to download (e.g., "BTCUSDT")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        output_dir: Directory to save files
        api_key: Tardis API key

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    downloaded_files = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        year = current.strftime("%Y")
        month = current.strftime("%m")
        day = current.strftime("%d")

        url = (
            f"https://datasets.tardis.dev/v1/{exchange}/derivative_ticker/"
            f"{year}/{month}/{day}/{symbol}.csv.gz"
        )

        output_file = output_dir / f"{exchange}_{symbol}_{date_str}.csv"

        print(f"Downloading {date_str}...", end=" ", flush=True)

        try:
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code == 200:
                content = gzip.decompress(response.content)
                output_file.write_bytes(content)
                rows = content.count(b"\n") - 1
                size_kb = len(content) / 1024
                print(f"OK ({rows} rows, {size_kb:.1f} KB)")
                downloaded_files.append(output_file)
            elif response.status_code == 404:
                print("No data available")
            elif response.status_code == 401:
                error = response.json()
                print(f"Auth error: {error.get('message', 'Unauthorized')[:50]}...")
                # For 401, try to continue - might work for 1st of month
            else:
                print(f"Error: HTTP {response.status_code}")

        except Exception as e:
            print(f"Error: {e}")

        current += timedelta(days=1)

    return downloaded_files


def merge_csvs(files: list[Path], output_file: Path) -> int:
    """Merge multiple CSV files into one."""
    if not files:
        return 0

    total_rows = 0

    with open(output_file, "wb") as outf:
        for i, f in enumerate(sorted(files)):
            content = f.read_bytes()
            if i == 0:
                outf.write(content)
                total_rows += content.count(b"\n")
            else:
                # Skip header for subsequent files
                lines = content.split(b"\n", 1)
                if len(lines) > 1:
                    outf.write(lines[1])
                    total_rows += lines[1].count(b"\n")

    return total_rows


def main():
    # Configuration
    API_KEY = "TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"

    EXCHANGE = "binance-futures"
    SYMBOL = "BTCUSDT"
    START_DATE = "2025-01-01"
    END_DATE = "2025-01-31"
    OUTPUT_DIR = Path("data/funding")
    MERGED_FILE = OUTPUT_DIR / "btcusdt_funding_202501.csv"

    print("=" * 60)
    print("Downloading Tardis derivative_ticker (funding rate) data")
    print("=" * 60)
    print(f"  Exchange: {EXCHANGE}")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print()

    # Download
    files = download_derivative_ticker_http(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=OUTPUT_DIR,
        api_key=API_KEY,
    )

    print()
    print(f"Downloaded {len(files)} files")

    if not files:
        print("No files downloaded, exiting")
        return

    # Merge
    print()
    print("Merging files...")
    row_count = merge_csvs(files, MERGED_FILE)
    print(f"Merged to: {MERGED_FILE}")
    print(f"Total rows: {row_count}")

    # Verify loading
    print()
    print("=" * 60)
    print("Verifying data loading with Nautilus")
    print("=" * 60)

    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from nautilus_quants.data.transform.funding import load_funding_rates

    funding_rates = load_funding_rates(
        MERGED_FILE,
        instrument_id="BTCUSDT-PERP.BINANCE",
    )

    print(f"Loaded {len(funding_rates)} FundingRateUpdate records")

    if funding_rates:
        print()
        print("Sample records:")
        print(f"  First: {funding_rates[0]}")
        if len(funding_rates) > 1:
            print(f"  Last:  {funding_rates[-1]}")


if __name__ == "__main__":
    main()
