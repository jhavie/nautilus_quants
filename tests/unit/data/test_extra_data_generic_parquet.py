# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for generic parquet loader (Phase 0 generalization)."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nautilus_quants.data.transform.open_interest import (
    load_oi_lookup,
    load_parquet_field_lookup,
)
from nautilus_quants.factors.engine.extra_data import (
    ExtraDataConfig,
    parse_extra_data_raw,
)


@pytest.fixture()
def catalog_with_oi(tmp_path):
    """Create a catalog with OI parquet files (existing format)."""
    oi_dir = tmp_path / "open_interest"
    oi_dir.mkdir()

    schema = pa.schema(
        [
            pa.field("timestamp_ns", pa.int64()),
            pa.field("instrument_id", pa.string()),
            pa.field("open_interest", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "timestamp_ns": [1000000000000, 2000000000000, 3000000000000],
            "instrument_id": ["BTCUSDT.BINANCE"] * 3,
            "open_interest": [100.0, 200.0, 300.0],
        },
        schema=schema,
    )
    pq.write_table(table, str(oi_dir / "BTCUSDT_4h_oi.parquet"))
    return tmp_path


@pytest.fixture()
def catalog_with_san_fr(tmp_path):
    """Create a catalog with SanAPI funding rate parquet files."""
    fr_dir = tmp_path / "san_funding_rate"
    fr_dir.mkdir()

    schema = pa.schema(
        [
            pa.field("timestamp_ns", pa.int64()),
            pa.field("instrument_id", pa.string()),
            pa.field("san_funding_rate", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "timestamp_ns": [1000000000000, 2000000000000],
            "instrument_id": ["ETHUSDT.BINANCE"] * 2,
            "san_funding_rate": [0.0001, -0.0002],
        },
        schema=schema,
    )
    pq.write_table(table, str(fr_dir / "ETHUSDT_4h_san_fr.parquet"))
    return tmp_path


class TestLoadOiLookupBackwardCompat:
    def test_loads_existing_oi(self, catalog_with_oi) -> None:
        lookup = load_oi_lookup(
            catalog_with_oi,
            ["BTCUSDT.BINANCE"],
            "4h",
        )
        assert "BTCUSDT.BINANCE" in lookup
        ts_map = lookup["BTCUSDT.BINANCE"]
        assert len(ts_map) == 3
        assert ts_map[1000000000000]["open_interest"] == 100.0

    def test_missing_symbol(self, catalog_with_oi) -> None:
        lookup = load_oi_lookup(
            catalog_with_oi,
            ["FOOBAR.BINANCE"],
            "4h",
        )
        assert lookup["FOOBAR.BINANCE"] == {}


class TestLoadParquetFieldLookup:
    def test_generic_field(self, catalog_with_san_fr) -> None:
        lookup = load_parquet_field_lookup(
            catalog_path=catalog_with_san_fr,
            instrument_ids=["ETHUSDT.BINANCE"],
            field_name="san_funding_rate",
            timeframe="4h",
            file_suffix="san_fr",
            subdirectory="san_funding_rate",
        )
        assert "ETHUSDT.BINANCE" in lookup
        ts_map = lookup["ETHUSDT.BINANCE"]
        assert len(ts_map) == 2
        assert ts_map[1000000000000]["san_funding_rate"] == 0.0001

    def test_missing_column(self, tmp_path) -> None:
        wrong_dir = tmp_path / "wrong_field"
        wrong_dir.mkdir()
        schema = pa.schema(
            [
                pa.field("timestamp_ns", pa.int64()),
                pa.field("instrument_id", pa.string()),
                pa.field("other_col", pa.float64()),
            ]
        )
        table = pa.table(
            {
                "timestamp_ns": [1000000000000],
                "instrument_id": ["BTCUSDT.BINANCE"],
                "other_col": [42.0],
            },
            schema=schema,
        )
        pq.write_table(table, str(wrong_dir / "BTCUSDT_4h_wrong_field.parquet"))

        lookup = load_parquet_field_lookup(
            tmp_path,
            ["BTCUSDT.BINANCE"],
            field_name="wrong_field",
            timeframe="4h",
        )
        assert lookup["BTCUSDT.BINANCE"] == {}


class TestExtraDataConfigNewFields:
    def test_parse_with_field_name(self) -> None:
        raw = {
            "san_fr": {
                "source": "parquet",
                "path": "/tmp/catalog",
                "field_name": "san_funding_rate",
                "file_suffix": "san_fr",
                "timeframe": "4h",
            }
        }
        configs = parse_extra_data_raw(raw)
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "san_fr"
        assert cfg.field_name == "san_funding_rate"
        assert cfg.file_suffix == "san_fr"

    def test_backward_compat_no_field_name(self) -> None:
        raw = {
            "open_interest": {
                "source": "parquet",
                "path": "/tmp/catalog",
                "timeframe": "4h",
            }
        }
        configs = parse_extra_data_raw(raw)
        cfg = configs[0]
        assert cfg.field_name == ""
        assert cfg.file_suffix == ""
