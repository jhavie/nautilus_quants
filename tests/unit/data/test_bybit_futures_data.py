# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Unit tests for BybitFundingRateDownloader and BybitOpenInterestDownloader.

Tests download functionality with mocked pybit HTTP responses.
"""

import csv
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.data.types import RawFundingRate, RawOpenInterest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    out = tmp_path / "raw" / "bybit"
    out.mkdir(parents=True)
    return out


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    cp_dir = tmp_path / "raw" / "bybit" / ".checkpoints"
    cp_dir.mkdir(parents=True)
    return cp_dir


def _make_fr_response(count: int, base_time: int = 1704067200000) -> dict:
    """Build sample Bybit funding-rate API response."""
    interval_ms = 28800000  # 8 hours
    records = [
        {
            "symbol": "BTCUSDT",
            "fundingRateTimestamp": str(base_time + i * interval_ms),
            "fundingRate": "0.00010000",
        }
        for i in range(count)
    ]
    # Bybit returns newest first
    records.reverse()
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {"category": "linear", "list": records},
    }


def _make_oi_response(
    count: int,
    base_time: int = 1704067200000,
    cursor: str = "",
) -> dict:
    """Build sample Bybit open-interest API response."""
    interval_ms = 14400000  # 4 hours
    records = [
        {
            "openInterest": "50000.12300000",
            "timestamp": str(base_time + i * interval_ms),
        }
        for i in range(count)
    ]
    records.reverse()
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "symbol": "BTCUSDT",
            "category": "linear",
            "list": records,
            "nextPageCursor": cursor,
        },
    }


# ===================================================================
# TestBybitFundingRateDownloader
# ===================================================================


class TestBybitFundingRateDownloader:
    """Tests for BybitFundingRateDownloader class."""

    def test_download_creates_csv_with_correct_columns(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock get_funding_rate_history, verify CSV columns."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitFundingRateDownloader,
        )

        resp = _make_fr_response(5)

        with patch(
            "nautilus_quants.data.download.bybit_futures_data.HTTP",
        ) as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_session.get_funding_rate_history.return_value = resp

            downloader = BybitFundingRateDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = downloader.download(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

        assert result.success is True
        assert result.rows_downloaded == 5
        assert result.file_path.exists()

        with open(result.file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["timestamp", "funding_rate"]

    def test_download_paginates_correctly(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock API to return 200 rows then fewer, verify both pages."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitFundingRateDownloader,
        )

        # Page 1: 200 records starting well after start_date
        # so current_end_ms stays >= start_ms after page1
        base = 1706000000000  # ~2024-01-23
        page1 = _make_fr_response(200, base_time=base)
        page2 = _make_fr_response(3, base_time=1704067200000)  # 2024-01-01

        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            return page2

        with patch(
            "nautilus_quants.data.download.bybit_futures_data.HTTP",
        ) as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_session.get_funding_rate_history.side_effect = _side_effect

            downloader = BybitFundingRateDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = downloader.download(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2025-01-01",
            )

        assert result.success is True
        assert result.rows_downloaded == 203

    def test_download_validates_symbol(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass invalid symbol, verify failure."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitFundingRateDownloader,
        )

        downloader = BybitFundingRateDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = downloader.download(
            symbol="",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

        assert result.success is False
        assert any("Invalid symbol" in e for e in result.errors)

    def test_download_validates_date_range(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass end_date before start_date, verify failure."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitFundingRateDownloader,
        )

        downloader = BybitFundingRateDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = downloader.download(
            symbol="BTCUSDT",
            start_date="2024-12-31",
            end_date="2024-01-01",
        )

        assert result.success is False
        assert any("Invalid date range" in e for e in result.errors)

    def test_download_output_path_format(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Verify output path format."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitFundingRateDownloader,
        )

        resp = _make_fr_response(1)

        with patch(
            "nautilus_quants.data.download.bybit_futures_data.HTTP",
        ) as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_session.get_funding_rate_history.return_value = resp

            downloader = BybitFundingRateDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = downloader.download(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

        expected = (
            output_dir / "BTCUSDT" / "funding_rate"
            / "BTCUSDT_fr_20240101_20240131.csv"
        )
        assert result.file_path == expected


# ===================================================================
# TestBybitOpenInterestDownloader
# ===================================================================


class TestBybitOpenInterestDownloader:
    """Tests for BybitOpenInterestDownloader class."""

    def test_download_creates_csv_with_correct_columns(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock get_open_interest, verify CSV columns."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitOpenInterestDownloader,
        )

        resp = _make_oi_response(5)

        with patch(
            "nautilus_quants.data.download.bybit_futures_data.HTTP",
        ) as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_session.get_open_interest.return_value = resp

            downloader = BybitOpenInterestDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = downloader.download(
                symbol="BTCUSDT",
                period="4h",
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

        assert result.success is True
        assert result.rows_downloaded == 5
        assert result.file_path.exists()

        with open(result.file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["timestamp", "open_interest"]

    def test_download_validates_period(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass invalid period, verify failure."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitOpenInterestDownloader,
        )

        downloader = BybitOpenInterestDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = downloader.download(
            symbol="BTCUSDT",
            period="3m",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

        assert result.success is False
        assert any("Invalid period" in e for e in result.errors)

    def test_download_output_path_format(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Verify output path includes period in filename."""
        from nautilus_quants.data.download.bybit_futures_data import (
            BybitOpenInterestDownloader,
        )

        resp = _make_oi_response(1)

        with patch(
            "nautilus_quants.data.download.bybit_futures_data.HTTP",
        ) as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_session.get_open_interest.return_value = resp

            downloader = BybitOpenInterestDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = downloader.download(
                symbol="ETHUSDT",
                period="4h",
                start_date="2024-06-01",
                end_date="2024-06-30",
            )

        expected = (
            output_dir / "ETHUSDT" / "open_interest"
            / "ETHUSDT_oi_4h_20240601_20240630.csv"
        )
        assert result.file_path == expected


# ===================================================================
# TestRawDataTypes
# ===================================================================


class TestRawDataTypes:
    """Tests for RawFundingRate and RawOpenInterest frozen dataclasses."""

    def test_raw_funding_rate_frozen(self) -> None:
        """Verify RawFundingRate is immutable."""
        rate = RawFundingRate(
            timestamp=1704067200000,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
        )

        with pytest.raises(AttributeError):
            rate.funding_rate = Decimal("0.0002")  # type: ignore[misc]

    def test_raw_open_interest_frozen(self) -> None:
        """Verify RawOpenInterest is immutable."""
        oi = RawOpenInterest(
            timestamp=1704067200000,
            symbol="BTCUSDT",
            open_interest=Decimal("50000.123"),
        )

        with pytest.raises(AttributeError):
            oi.open_interest = Decimal("99999")  # type: ignore[misc]
