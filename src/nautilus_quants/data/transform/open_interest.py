# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Transform module for Open Interest data -- write/load as standalone Parquet.

Since Open Interest has no native NautilusTrader data type, we store it as
plain Parquet files alongside the catalog using pandas + pyarrow.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def write_oi_parquet(
    csv_path: Path | str,
    output_path: Path | str,
    symbol: str,
    venue: str = "BINANCE",
) -> int:
    """Convert OI CSV to Parquet file.

    CSV format: timestamp (ms), open_interest

    Output Parquet columns:
        - timestamp_ns: int64 (ms * 1_000_000)
        - instrument_id: string (e.g., "BTCUSDT.BINANCE")
        - open_interest: float64

    Args:
        csv_path: Path to the open interest CSV file.
        output_path: Destination Parquet file path.
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        venue: Exchange venue name (default "BINANCE").

    Returns:
        Number of rows written.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path)

    if df.empty:
        return 0

    instrument_id = f"{symbol}.{venue}"

    out_df = pd.DataFrame(
        {
            "timestamp_ns": (df["timestamp"].astype("int64") * 1_000_000),
            "instrument_id": instrument_id,
            "open_interest": df["open_interest"].astype("float64"),
        }
    )

    schema = pa.schema(
        [
            pa.field("timestamp_ns", pa.int64()),
            pa.field("instrument_id", pa.string()),
            pa.field("open_interest", pa.float64()),
        ]
    )

    table = pa.Table.from_pandas(out_df, schema=schema, preserve_index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))

    return len(out_df)


def transform_open_interest(
    raw_dir: Path | str,
    catalog_path: Path | str,
    symbols: list[str],
    timeframe: str = "4h",
    venue: str = "BINANCE",
) -> list[dict]:
    """Transform OI CSVs for all symbols to Parquet.

    For each symbol:
    1. Find OI CSV in {raw_dir}/{symbol}/open_interest/
    2. Write to {catalog_path}/open_interest/{symbol}_{timeframe}_oi.parquet

    Args:
        raw_dir: Root raw data directory (e.g., data/raw/binance).
        catalog_path: NautilusTrader Parquet catalog directory.
        symbols: List of trading pair symbols.
        timeframe: Timeframe label used in the output filename (default "4h").
        venue: Exchange venue name (default "BINANCE").

    Returns:
        List of dicts with keys: symbol, count, success, and optionally error.
    """
    raw_dir = Path(raw_dir)
    catalog_path = Path(catalog_path)

    oi_output_dir = catalog_path / "open_interest"
    oi_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for symbol in symbols:
        oi_dir = raw_dir / symbol / "open_interest"

        if not oi_dir.exists():
            logger.warning("No open_interest directory for %s at %s", symbol, oi_dir)
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": f"Directory not found: {oi_dir}",
                }
            )
            continue

        csv_files = sorted(oi_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found for %s in %s", symbol, oi_dir)
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": f"No CSV files in {oi_dir}",
                }
            )
            continue

        try:
            # Concatenate all CSV files for this symbol
            frames: list[pd.DataFrame] = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    frames.append(df)

            if not frames:
                results.append(
                    {
                        "symbol": symbol,
                        "count": 0,
                        "success": False,
                        "error": "No data rows in CSV files",
                    }
                )
                continue

            merged_df = pd.concat(frames, ignore_index=True)
            merged_df = merged_df.drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

            # Write merged data to a temporary CSV-like DataFrame,
            # then convert via write_oi_parquet's logic inline
            output_file = oi_output_dir / f"{symbol}_{timeframe}_oi.parquet"

            instrument_id = f"{symbol}.{venue}"

            out_df = pd.DataFrame(
                {
                    "timestamp_ns": (
                        merged_df["timestamp"].astype("int64") * 1_000_000
                    ),
                    "instrument_id": instrument_id,
                    "open_interest": merged_df["open_interest"].astype("float64"),
                }
            )

            schema = pa.schema(
                [
                    pa.field("timestamp_ns", pa.int64()),
                    pa.field("instrument_id", pa.string()),
                    pa.field("open_interest", pa.float64()),
                ]
            )

            table = pa.Table.from_pandas(
                out_df, schema=schema, preserve_index=False
            )
            pq.write_table(table, str(output_file))

            count = len(out_df)
            logger.info(
                "Wrote %d OI records for %s to %s", count, symbol, output_file
            )
            results.append(
                {
                    "symbol": symbol,
                    "count": count,
                    "success": True,
                }
            )

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            logger.error(
                "Failed to transform open interest for %s: %s", symbol, msg
            )
            results.append(
                {
                    "symbol": symbol,
                    "count": 0,
                    "success": False,
                    "error": msg,
                }
            )

    return results


def load_oi_lookup(
    catalog_path: Path | str,
    instrument_ids: list[str],
    timeframe: str = "4h",
) -> dict[str, dict[int, dict[str, float]]]:
    """Load OI Parquet files as nested lookup dict.

    Scans {catalog_path}/open_interest/ for matching files by extracting
    the symbol from each instrument_id (e.g., "BTCUSDT" from "BTCUSDT.BINANCE").

    Args:
        catalog_path: NautilusTrader Parquet catalog directory.
        instrument_ids: List of instrument ID strings (e.g., ["BTCUSDT.BINANCE"]).
        timeframe: Timeframe label used in the filename (default "4h").

    Returns:
        Nested dict: {instrument_id: {timestamp_ns: {"open_interest": float}}}

    For backtesting, the FactorEngineActor calls this at startup and uses
    the lookup to inject OI values into bar_data on each bar.
    """
    catalog_path = Path(catalog_path)
    oi_dir = catalog_path / "open_interest"

    lookup: dict[str, dict[int, dict[str, float]]] = {}

    for iid in instrument_ids:
        # Extract symbol from instrument_id: "BTCUSDT.BINANCE" -> "BTCUSDT"
        symbol = iid.split(".")[0]
        parquet_file = oi_dir / f"{symbol}_{timeframe}_oi.parquet"

        if not parquet_file.exists():
            logger.info("OI parquet not found for %s (skipped)", iid)
            lookup[iid] = {}
            continue

        table = pq.read_table(str(parquet_file))
        df = table.to_pandas()

        ts_lookup: dict[int, dict[str, float]] = {}
        for _, row in df.iterrows():
            ts_lookup[int(row["timestamp_ns"])] = {
                "open_interest": float(row["open_interest"]),
            }

        lookup[iid] = ts_lookup
        logger.info("Loaded %d OI records for %s", len(ts_lookup), iid)

    return lookup
