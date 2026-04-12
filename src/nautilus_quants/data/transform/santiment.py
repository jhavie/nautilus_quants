# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Transform Santiment CSV data to standalone Parquet files.

CSV input format (from SantimentDownloader): ``timestamp_ms, value``
Parquet output: ``{timestamp_ns, instrument_id, <field_name>}``

Output directory: ``{catalog_path}/{field_name}/{SYMBOL}_{timeframe}_{suffix}.parquet``
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nautilus_quants.data.santiment.slug_map import AVAILABLE, instrument_to_ticker

logger = logging.getLogger(__name__)


def write_santiment_parquet(
    csv_path: Path | str,
    output_path: Path | str,
    instrument_id: str,
    field_name: str,
) -> int:
    """Convert a Santiment CSV to a single Parquet file.

    Args:
        csv_path: Source CSV with columns ``timestamp_ms, value``.
        output_path: Destination Parquet file path.
        instrument_id: Instrument ID (e.g. ``"BTCUSDT.BINANCE"``).
        field_name: Column name in the output Parquet (e.g. ``"san_funding_rate"``).

    Returns:
        Number of rows written.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    out_df = pd.DataFrame(
        {
            "timestamp_ns": df["timestamp_ms"].astype("int64") * 1_000_000,
            "instrument_id": instrument_id,
            field_name: df["value"].astype("float64"),
        }
    )

    schema = pa.schema(
        [
            pa.field("timestamp_ns", pa.int64()),
            pa.field("instrument_id", pa.string()),
            pa.field(field_name, pa.float64()),
        ]
    )

    table = pa.Table.from_pandas(out_df, schema=schema, preserve_index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))

    return len(out_df)


def transform_santiment(
    raw_dir: Path | str,
    catalog_path: Path | str,
    tickers: list[str],
    metric: str,
    field_name: str,
    file_suffix: str,
    venue: str = "BINANCE",
    timeframe: str = "4h",
) -> list[dict]:
    """Transform all Santiment CSVs for a metric into Parquet files.

    Reads: ``{raw_dir}/{metric}/{ticker}.csv``
    Writes: ``{catalog_path}/{field_name}/{SYMBOL}_{timeframe}_{suffix}.parquet``

    Args:
        raw_dir: Root raw data directory (e.g. ``data/raw/santiment``).
        catalog_path: Output catalog directory (e.g. ``data/catalog_sanapi``).
        tickers: List of tickers to transform (e.g. ``["BTC", "ETH"]``).
        metric: SanAPI metric name (e.g. ``"funding_rate"``).
        field_name: Parquet column name (e.g. ``"san_funding_rate"``).
        file_suffix: Parquet filename suffix (e.g. ``"san_fr"``).
        venue: Venue name for instrument IDs (default ``"BINANCE"``).
        timeframe: Timeframe label in filename (default ``"4h"``).

    Returns:
        List of dicts with keys: ``ticker, symbol, count, success`` and
        optionally ``error``.
    """
    raw_dir = Path(raw_dir)
    catalog_path = Path(catalog_path)
    output_dir = catalog_path / field_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for ticker in tickers:
        csv_path = raw_dir / metric / f"{ticker}.csv"

        if not csv_path.exists():
            results.append(
                {
                    "ticker": ticker,
                    "symbol": f"{ticker}USDT",
                    "count": 0,
                    "success": False,
                    "error": f"CSV not found: {csv_path}",
                }
            )
            continue

        try:
            symbol = f"{ticker}USDT"
            instrument_id = f"{symbol}.{venue}"
            output_file = output_dir / f"{symbol}_{timeframe}_{file_suffix}.parquet"

            count = write_santiment_parquet(
                csv_path=csv_path,
                output_path=output_file,
                instrument_id=instrument_id,
                field_name=field_name,
            )

            logger.info(
                "Wrote %d %s records for %s to %s",
                count,
                field_name,
                ticker,
                output_file,
            )
            results.append(
                {
                    "ticker": ticker,
                    "symbol": symbol,
                    "count": count,
                    "success": True,
                }
            )

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            logger.error("Failed to transform %s for %s: %s", metric, ticker, msg)
            results.append(
                {
                    "ticker": ticker,
                    "symbol": f"{ticker}USDT",
                    "count": 0,
                    "success": False,
                    "error": msg,
                }
            )

    return results
