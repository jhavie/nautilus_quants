# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for open interest CSV → Parquet transform and load_oi_lookup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nautilus_quants.data.transform.open_interest import (
    load_oi_lookup,
    write_oi_parquet,
)


class TestWriteOiParquetRoundtrip:
    """Write OI CSV to Parquet, read back, verify columns and values."""

    def test_write_oi_parquet_roundtrip(self, tmp_path: Path) -> None:
        """Create tmp CSV, write to parquet, read back, verify columns
        and values match.
        """
        csv_file = tmp_path / "oi.csv"
        csv_file.write_text(
            "timestamp,open_interest,open_interest_value\n"
            "1700000000000,12345.67,456789000.0\n"
            "1700028800000,12500.00,462500000.0\n"
        )

        output_parquet = tmp_path / "output" / "oi.parquet"
        count = write_oi_parquet(csv_file, output_parquet, "BTCUSDT", "BINANCE")

        assert count == 2
        assert output_parquet.exists()

        # Read back and verify
        table = pq.read_table(str(output_parquet))
        df = table.to_pandas()

        assert list(df.columns) == [
            "timestamp_ns",
            "instrument_id",
            "open_interest",
            "open_interest_value",
        ]
        assert len(df) == 2

        # Verify ms → ns conversion
        assert df.iloc[0]["timestamp_ns"] == 1700000000000 * 1_000_000
        assert df.iloc[1]["timestamp_ns"] == 1700028800000 * 1_000_000

        # Verify instrument_id
        assert df.iloc[0]["instrument_id"] == "BTCUSDT.BINANCE"

        # Verify numeric values
        assert df.iloc[0]["open_interest"] == pytest.approx(12345.67)
        assert df.iloc[0]["open_interest_value"] == pytest.approx(456789000.0)


class TestWriteOiParquetEmptyCsv:
    """Empty CSV should return 0 rows and not create a file."""

    def test_write_oi_parquet_empty_csv(self, tmp_path: Path) -> None:
        """Empty CSV returns 0 rows."""
        csv_file = tmp_path / "empty_oi.csv"
        csv_file.write_text("timestamp,open_interest,open_interest_value\n")

        output_parquet = tmp_path / "output" / "empty_oi.parquet"
        count = write_oi_parquet(csv_file, output_parquet, "ETHUSDT")

        assert count == 0


class TestLoadOiLookupBasic:
    """Write OI parquet, load via load_oi_lookup, verify nested dict structure."""

    def test_load_oi_lookup_basic(self, tmp_path: Path) -> None:
        """Verify nested dict structure {inst_id: {ts_ns: {oi, oi_value}}}."""
        # Prepare a parquet file in the expected directory layout
        csv_file = tmp_path / "source.csv"
        csv_file.write_text(
            "timestamp,open_interest,open_interest_value\n"
            "1700000000000,5000.0,175000000.0\n"
            "1700014400000,5100.0,178500000.0\n"
        )

        oi_dir = tmp_path / "catalog" / "open_interest"
        oi_dir.mkdir(parents=True)
        parquet_file = oi_dir / "BTCUSDT_4h_oi.parquet"
        write_oi_parquet(csv_file, parquet_file, "BTCUSDT", "BINANCE")

        catalog_path = tmp_path / "catalog"
        lookup = load_oi_lookup(
            catalog_path,
            ["BTCUSDT.BINANCE"],
            timeframe="4h",
        )

        assert "BTCUSDT.BINANCE" in lookup
        ts_map = lookup["BTCUSDT.BINANCE"]
        assert len(ts_map) == 2

        ts_ns_0 = 1700000000000 * 1_000_000
        assert ts_ns_0 in ts_map
        assert ts_map[ts_ns_0]["open_interest"] == pytest.approx(5000.0)
        assert ts_map[ts_ns_0]["open_interest_value"] == pytest.approx(
            175000000.0,
        )

    def test_load_oi_lookup_multiple_instruments(self, tmp_path: Path) -> None:
        """Verify lookup loads data for multiple instruments."""
        oi_dir = tmp_path / "catalog" / "open_interest"
        oi_dir.mkdir(parents=True)

        for symbol, oi_val in [("BTCUSDT", 5000.0), ("ETHUSDT", 80000.0)]:
            csv_file = tmp_path / f"{symbol}.csv"
            csv_file.write_text(
                "timestamp,open_interest,open_interest_value\n"
                f"1700000000000,{oi_val},{oi_val * 35}\n"
            )
            parquet_file = oi_dir / f"{symbol}_4h_oi.parquet"
            write_oi_parquet(csv_file, parquet_file, symbol, "BINANCE")

        lookup = load_oi_lookup(
            tmp_path / "catalog",
            ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE"],
            timeframe="4h",
        )

        assert len(lookup) == 2
        assert "BTCUSDT.BINANCE" in lookup
        assert "ETHUSDT.BINANCE" in lookup


class TestLoadOiLookupMissingFile:
    """Missing parquet file should return empty dict for that instrument."""

    def test_load_oi_lookup_missing_file(self, tmp_path: Path) -> None:
        """Verify returns empty dict when file doesn't exist."""
        catalog_path = tmp_path / "catalog"
        # Don't create any files — directory may not even exist

        lookup = load_oi_lookup(
            catalog_path,
            ["BTCUSDT.BINANCE"],
            timeframe="4h",
        )

        assert "BTCUSDT.BINANCE" in lookup
        assert lookup["BTCUSDT.BINANCE"] == {}
