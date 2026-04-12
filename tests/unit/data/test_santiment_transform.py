# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for Santiment data transformer."""
import pytest

import pandas as pd
import pyarrow.parquet as pq


from nautilus_quants.data.transform.santiment import (
    transform_santiment,
    write_santiment_parquet,
)


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a sample Santiment CSV file."""
    csv_path = tmp_path / "funding_rate" / "BTC.csv"
    csv_path.parent.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "timestamp_ms": [1700000000000, 1700014400000, 1700028800000],
            "value": [0.0001, 0.00015, -0.0002],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


class TestWriteSantimentParquet:
    def test_basic(self, sample_csv, tmp_path) -> None:
        output = tmp_path / "out.parquet"
        rows = write_santiment_parquet(
            csv_path=sample_csv,
            output_path=output,
            instrument_id="BTCUSDT.BINANCE",
            field_name="san_funding_rate",
        )
        assert rows == 3
        assert output.exists()

        table = pq.read_table(str(output))
        df = table.to_pandas()
        assert "timestamp_ns" in df.columns
        assert "instrument_id" in df.columns
        assert "san_funding_rate" in df.columns
        assert len(df) == 3
        assert df["instrument_id"].iloc[0] == "BTCUSDT.BINANCE"

    def test_empty_csv(self, tmp_path) -> None:
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["timestamp_ms", "value"]).to_csv(csv_path, index=False)
        output = tmp_path / "out.parquet"
        rows = write_santiment_parquet(csv_path, output, "BTCUSDT.BINANCE", "san_fr")
        assert rows == 0


class TestTransformSantiment:
    def test_transform_single(self, tmp_path) -> None:
        # Setup raw CSV
        raw_dir = tmp_path / "raw"
        csv_dir = raw_dir / "funding_rate"
        csv_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp_ms": [1700000000000, 1700014400000],
                "value": [0.0001, 0.00015],
            }
        ).to_csv(csv_dir / "BTC.csv", index=False)

        catalog = tmp_path / "catalog"
        results = transform_santiment(
            raw_dir=raw_dir,
            catalog_path=catalog,
            tickers=["BTC"],
            metric="funding_rate",
            field_name="san_funding_rate",
            file_suffix="san_fr",
        )

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["count"] == 2

        # Verify output file
        pq_path = catalog / "san_funding_rate" / "BTCUSDT_4h_san_fr.parquet"
        assert pq_path.exists()

    def test_missing_csv(self, tmp_path) -> None:
        results = transform_santiment(
            raw_dir=tmp_path / "nonexistent",
            catalog_path=tmp_path / "catalog",
            tickers=["BTC"],
            metric="funding_rate",
            field_name="san_fr",
            file_suffix="san_fr",
        )
        assert len(results) == 1
        assert results[0]["success"] is False
