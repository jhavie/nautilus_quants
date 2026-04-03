# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Unit tests for FundingRateDownloader and OpenInterestDownloader.

Tests download functionality with mocked Binance API responses.
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nautilus_quants.data.types import RawFundingRate, RawOpenInterest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    out = tmp_path / "raw" / "binance"
    out.mkdir(parents=True)
    return out


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    cp_dir = tmp_path / "raw" / "binance" / ".checkpoints"
    cp_dir.mkdir(parents=True)
    return cp_dir


def _make_funding_records(count: int, base_time: int = 1704067200000) -> list[dict]:
    """Build sample Binance funding-rate API response dicts."""
    interval_ms = 28800000  # 8 hours
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": base_time + i * interval_ms,
            "fundingRate": "0.00010000",
            "markPrice": "42000.50",
        }
        for i in range(count)
    ]


def _make_oi_records(count: int, base_time: int = 1704067200000) -> list[dict]:
    """Build sample Binance open-interest-hist API response dicts."""
    interval_ms = 3600000  # 1 hour
    return [
        {
            "symbol": "BTCUSDT",
            "timestamp": base_time + i * interval_ms,
            "sumOpenInterest": "50000.123",
            "sumOpenInterestValue": "2100000000.50",
        }
        for i in range(count)
    ]


# ===================================================================
# TestFundingRateDownloader
# ===================================================================


class TestFundingRateDownloader:
    """Tests for FundingRateDownloader class."""

    @pytest.mark.asyncio
    async def test_download_creates_csv_with_correct_columns(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock futures_funding_rate, verify CSV columns."""
        from nautilus_quants.data.download.binance_futures_data import (
            FundingRateDownloader,
        )

        records = _make_funding_records(5)

        with patch(
            "nautilus_quants.data.download.binance_futures_data.AsyncClient",
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()
            # Single page, fewer than 1000 → no pagination
            mock_client.futures_funding_rate = AsyncMock(
                return_value=records,
            )

            downloader = FundingRateDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = await downloader.download(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

        assert result.success is True
        assert result.rows_downloaded == 5
        assert result.file_path.exists()

        import csv

        with open(result.file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["timestamp", "funding_rate", "mark_price"]

    @pytest.mark.asyncio
    async def test_download_paginates_correctly(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock API to return 1000 rows then fewer, verify both pages written."""
        from nautilus_quants.data.download.binance_futures_data import (
            FundingRateDownloader,
        )

        page1 = _make_funding_records(1000, base_time=1704067200000)
        # Page 2 starts after last record of page 1
        last_ts_page1 = int(page1[-1]["fundingTime"])
        page2 = _make_funding_records(
            3, base_time=last_ts_page1 + 1,
        )

        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            return page2

        with patch(
            "nautilus_quants.data.download.binance_futures_data.AsyncClient",
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()
            mock_client.futures_funding_rate = AsyncMock(
                side_effect=_side_effect,
            )

            downloader = FundingRateDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = await downloader.download(
                symbol="BTCUSDT",
                start_date="2024-01-01",
                end_date="2025-01-01",
            )

        assert result.success is True
        assert result.rows_downloaded == 1003

        # Verify CSV row count (header + 1003 data rows)
        import csv

        with open(result.file_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1004  # header + data

    def test_download_validates_symbol(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass invalid symbol, verify DownloadResult.success is False."""
        import asyncio

        from nautilus_quants.data.download.binance_futures_data import (
            FundingRateDownloader,
        )

        downloader = FundingRateDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = asyncio.get_event_loop().run_until_complete(
            downloader.download(
                symbol="",
                start_date="2024-01-01",
                end_date="2024-01-02",
            ),
        )

        assert result.success is False
        assert any("Invalid symbol" in e for e in result.errors)

    def test_download_validates_date_range(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass end_date before start_date, verify failure."""
        import asyncio

        from nautilus_quants.data.download.binance_futures_data import (
            FundingRateDownloader,
        )

        downloader = FundingRateDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = asyncio.get_event_loop().run_until_complete(
            downloader.download(
                symbol="BTCUSDT",
                start_date="2024-12-31",
                end_date="2024-01-01",
            ),
        )

        assert result.success is False
        assert any("Invalid date range" in e for e in result.errors)

    def test_download_output_path_format(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Verify output path is {output_dir}/{SYMBOL}/funding_rate/{SYMBOL}_fr_{start}_{end}.csv."""
        import asyncio

        from nautilus_quants.data.download.binance_futures_data import (
            FundingRateDownloader,
        )

        downloader = FundingRateDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

        # Use invalid date range so download returns early but path is still set
        # Instead, use valid inputs but mock the client so we can inspect the path
        records = _make_funding_records(1)

        with patch(
            "nautilus_quants.data.download.binance_futures_data.AsyncClient",
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()
            mock_client.futures_funding_rate = AsyncMock(
                return_value=records,
            )

            result = asyncio.get_event_loop().run_until_complete(
                downloader.download(
                    symbol="BTCUSDT",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                ),
            )

        expected = (
            output_dir / "BTCUSDT" / "funding_rate"
            / "BTCUSDT_fr_20240101_20240131.csv"
        )
        assert result.file_path == expected


# ===================================================================
# TestOpenInterestDownloader
# ===================================================================


class TestOpenInterestDownloader:
    """Tests for OpenInterestDownloader class."""

    @pytest.mark.asyncio
    async def test_download_creates_csv_with_correct_columns(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Mock futures_open_interest_hist, verify CSV columns."""
        from nautilus_quants.data.download.binance_futures_data import (
            OpenInterestDownloader,
        )

        records = _make_oi_records(5)

        with patch(
            "nautilus_quants.data.download.binance_futures_data.AsyncClient",
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()
            mock_client.futures_open_interest_hist = AsyncMock(
                return_value=records,
            )

            downloader = OpenInterestDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )
            result = await downloader.download(
                symbol="BTCUSDT",
                period="1h",
                start_date="2024-01-01",
                end_date="2024-01-02",
            )

        assert result.success is True
        assert result.rows_downloaded == 5
        assert result.file_path.exists()

        import csv

        with open(result.file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == [
            "timestamp",
            "open_interest",
            "open_interest_value",
        ]

    def test_download_validates_period(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Pass invalid period, verify failure."""
        import asyncio

        from nautilus_quants.data.download.binance_futures_data import (
            OpenInterestDownloader,
        )

        downloader = OpenInterestDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        result = asyncio.get_event_loop().run_until_complete(
            downloader.download(
                symbol="BTCUSDT",
                period="3m",
                start_date="2024-01-01",
                end_date="2024-01-02",
            ),
        )

        assert result.success is False
        assert any("Invalid period" in e for e in result.errors)

    def test_download_output_path_format(
        self, output_dir: Path, checkpoint_dir: Path,
    ) -> None:
        """Verify output path includes period in filename."""
        import asyncio

        from nautilus_quants.data.download.binance_futures_data import (
            OpenInterestDownloader,
        )

        records = _make_oi_records(1)

        downloader = OpenInterestDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

        with patch(
            "nautilus_quants.data.download.binance_futures_data.AsyncClient",
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()
            mock_client.futures_open_interest_hist = AsyncMock(
                return_value=records,
            )

            result = asyncio.get_event_loop().run_until_complete(
                downloader.download(
                    symbol="ETHUSDT",
                    period="4h",
                    start_date="2024-06-01",
                    end_date="2024-06-30",
                ),
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
            mark_price=Decimal("42000.50"),
        )

        with pytest.raises(AttributeError):
            rate.funding_rate = Decimal("0.0002")  # type: ignore[misc]

    def test_raw_open_interest_frozen(self) -> None:
        """Verify RawOpenInterest is immutable."""
        oi = RawOpenInterest(
            timestamp=1704067200000,
            symbol="BTCUSDT",
            open_interest=Decimal("50000.123"),
            open_interest_value=Decimal("2100000000.50"),
        )

        with pytest.raises(AttributeError):
            oi.open_interest = Decimal("99999")  # type: ignore[misc]
