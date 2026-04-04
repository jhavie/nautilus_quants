# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for funding rate CSV → FundingRateUpdate transform."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from nautilus_quants.data.transform.funding_rate import csv_to_funding_rate_updates


class TestCsvToFundingRateUpdatesBasic:
    """Basic conversion of a multi-row CSV to FundingRateUpdate objects."""

    def test_csv_to_funding_rate_updates_basic(self, tmp_path: Path) -> None:
        """Create a tmp CSV with FR data, convert, verify FundingRateUpdate
        objects have correct instrument_id, rate (Decimal), and ts_event (ns).
        """
        csv_file = tmp_path / "funding_rate.csv"
        csv_file.write_text(
            "timestamp,funding_rate\n"
            "1700000000000,0.0001\n"
            "1700028800000,-0.00005\n"
        )

        updates = csv_to_funding_rate_updates(csv_file, "BTCUSDT", "BINANCE")

        assert len(updates) == 2

        # First row
        fru0 = updates[0]
        assert str(fru0.instrument_id) == "BTCUSDT.BINANCE"
        assert fru0.rate == Decimal("0.0001")
        assert fru0.ts_event == 1700000000000 * 1_000_000  # ms → ns

        # Second row
        fru1 = updates[1]
        assert str(fru1.instrument_id) == "BTCUSDT.BINANCE"
        assert fru1.rate == Decimal("-0.00005")
        assert fru1.ts_event == 1700028800000 * 1_000_000


class TestCsvToFundingRateUpdatesEmptyCsv:
    """Empty CSV should return an empty list."""

    def test_csv_to_funding_rate_updates_empty_csv(self, tmp_path: Path) -> None:
        """Empty CSV (header only) returns empty list."""
        csv_file = tmp_path / "empty_fr.csv"
        csv_file.write_text("timestamp,funding_rate\n")

        updates = csv_to_funding_rate_updates(csv_file, "ETHUSDT", "BINANCE")

        assert updates == []


class TestCsvToFundingRateUpdatesTimestamp:
    """Verify timestamp conversion from milliseconds to nanoseconds."""

    def test_csv_to_funding_rate_updates_timestamp_ms_to_ns(
        self, tmp_path: Path,
    ) -> None:
        """Timestamp is converted from ms to ns (multiply by 1_000_000)."""
        ts_ms = 1609459200000  # 2021-01-01 00:00:00 UTC in ms
        expected_ns = ts_ms * 1_000_000

        csv_file = tmp_path / "ts_test.csv"
        csv_file.write_text(
            "timestamp,funding_rate\n"
            f"{ts_ms},0.0003\n"
        )

        updates = csv_to_funding_rate_updates(csv_file, "BTCUSDT")

        assert len(updates) == 1
        assert updates[0].ts_event == expected_ns
        assert updates[0].ts_init == expected_ns

    def test_default_venue_is_binance(self, tmp_path: Path) -> None:
        """Default venue parameter should be BINANCE."""
        csv_file = tmp_path / "venue_test.csv"
        csv_file.write_text(
            "timestamp,funding_rate\n"
            "1700000000000,0.0001\n"
        )

        updates = csv_to_funding_rate_updates(csv_file, "BTCUSDT")

        assert str(updates[0].instrument_id) == "BTCUSDT.BINANCE"
