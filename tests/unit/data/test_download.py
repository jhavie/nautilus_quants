"""
Unit tests for the BinanceDownloader.

Tests download functionality with mocked API responses.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nautilus_quants.data.checkpoint import CheckpointManager
from nautilus_quants.data.types import DownloadCheckpoint


class TestBinanceDownloader:
    """Tests for BinanceDownloader class."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        out = tmp_path / "raw" / "binance"
        out.mkdir(parents=True)
        return out

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        cp_dir = tmp_path / "raw" / "binance" / ".checkpoints"
        cp_dir.mkdir(parents=True)
        return cp_dir

    @pytest.fixture
    def mock_klines(self) -> list[list]:
        """Sample kline data from Binance API format."""
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
        base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
        interval_ms = 3600000  # 1 hour in milliseconds

        klines = []
        for i in range(10):
            klines.append([
                base_time + i * interval_ms,  # Open time
                "42000.00",  # Open
                "42500.00",  # High
                "41800.00",  # Low
                "42200.00",  # Close
                "100.5",     # Volume
                base_time + (i + 1) * interval_ms - 1,  # Close time
                "4220000.00",  # Quote volume
                1500,         # Number of trades
                "50.25",      # Taker buy base volume
                "2110000.00",  # Taker buy quote volume
                "0"           # Ignore
            ])
        return klines

    def test_parse_kline_to_raw_kline(self, mock_klines: list[list]) -> None:
        """Test parsing Binance API kline format to RawKline."""
        from nautilus_quants.data.download.binance import _parse_kline

        kline = mock_klines[0]
        raw_kline = _parse_kline(kline, "BTCUSDT", "1h")

        assert raw_kline.timestamp == 1704067200000
        assert raw_kline.open == Decimal("42000.00")
        assert raw_kline.high == Decimal("42500.00")
        assert raw_kline.low == Decimal("41800.00")
        assert raw_kline.close == Decimal("42200.00")
        assert raw_kline.volume == Decimal("100.5")
        assert raw_kline.quote_volume == Decimal("4220000.00")
        assert raw_kline.trades_count == 1500
        assert raw_kline.symbol == "BTCUSDT"
        assert raw_kline.timeframe == "1h"

    def test_interval_to_milliseconds(self) -> None:
        """Test conversion of interval strings to milliseconds."""
        from nautilus_quants.data.download.binance import _interval_to_ms

        assert _interval_to_ms("1m") == 60000
        assert _interval_to_ms("5m") == 300000
        assert _interval_to_ms("15m") == 900000
        assert _interval_to_ms("1h") == 3600000
        assert _interval_to_ms("4h") == 14400000
        assert _interval_to_ms("1d") == 86400000

    def test_date_to_milliseconds(self) -> None:
        """Test conversion of date string to milliseconds."""
        from nautilus_quants.data.download.binance import _date_to_ms

        ms = _date_to_ms("2024-01-01")
        # Should be start of day in UTC
        assert ms == 1704067200000

    def test_get_output_path(self, output_dir: Path) -> None:
        """Test output file path generation."""
        from nautilus_quants.data.download.binance import _get_output_path

        path = _get_output_path(
            output_dir, "BTCUSDT", "1h", "2024-01-01", "2024-12-31"
        )

        assert path.parent == output_dir / "BTCUSDT" / "1h"
        assert path.name == "BTCUSDT_1h_20240101_20241231.csv"

    @pytest.mark.asyncio
    async def test_download_klines_creates_csv(
        self, output_dir: Path, checkpoint_dir: Path, mock_klines: list[list]
    ) -> None:
        """Test that download creates CSV file with correct format."""
        from nautilus_quants.data.download.binance import BinanceDownloader

        # Mock the async client
        with patch("nautilus_quants.data.download.binance.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()

            # Mock the klines generator - now needs to be awaitable
            async def mock_generator():
                for kline in mock_klines:
                    yield kline

            # The generator method is a coroutine that returns an async generator
            mock_client.futures_historical_klines_generator = AsyncMock(
                return_value=mock_generator()
            )

            downloader = BinanceDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )

            result = await downloader.download(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-02",
                market_type="futures",
            )

            assert result.success is True
            assert result.rows_downloaded == 10
            assert result.file_path.exists()

            # Verify CSV content
            import pandas as pd
            df = pd.read_csv(result.file_path)
            assert len(df) == 10
            assert list(df.columns) == [
                "timestamp", "open", "high", "low", "close",
                "volume", "quote_volume", "trades_count"
            ]

    @pytest.mark.asyncio
    async def test_download_klines_with_checkpoint_resume(
        self, output_dir: Path, checkpoint_dir: Path, mock_klines: list[list]
    ) -> None:
        """Test that download resumes from checkpoint."""
        from nautilus_quants.data.download.binance import BinanceDownloader

        # Create existing checkpoint
        checkpoint_mgr = CheckpointManager(checkpoint_dir)
        existing_checkpoint = DownloadCheckpoint(
            symbol="BTCUSDT",
            timeframe="1h",
            exchange="binance",
            market_type="futures",
            last_timestamp=1704067200000 + 5 * 3600000,  # 5 hours in
            last_updated=datetime.now(),
            total_rows=5,
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
        checkpoint_mgr.save(existing_checkpoint)

        # Mock returns only remaining klines (last 5)
        remaining_klines = mock_klines[5:]

        with patch("nautilus_quants.data.download.binance.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()

            async def mock_generator():
                for kline in remaining_klines:
                    yield kline

            mock_client.futures_historical_klines_generator = AsyncMock(
                return_value=mock_generator()
            )

            downloader = BinanceDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )

            result = await downloader.download(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-02",
                market_type="futures",
                resume=True,
            )

            assert result.success is True
            assert result.resumed_from_checkpoint is True
            # Should have downloaded only the remaining 5
            assert result.rows_downloaded == 5

    @pytest.mark.asyncio
    async def test_download_klines_spot_market(
        self, output_dir: Path, checkpoint_dir: Path, mock_klines: list[list]
    ) -> None:
        """Test download from spot market uses correct API."""
        from nautilus_quants.data.download.binance import BinanceDownloader

        with patch("nautilus_quants.data.download.binance.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.create = AsyncMock(return_value=mock_client)
            mock_client.close_connection = AsyncMock()

            async def mock_generator():
                for kline in mock_klines:
                    yield kline

            mock_client.get_historical_klines_generator = AsyncMock(
                return_value=mock_generator()
            )

            downloader = BinanceDownloader(
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
            )

            result = await downloader.download(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-02",
                market_type="spot",
            )

            assert result.success is True
            # Verify spot API was called
            mock_client.get_historical_klines_generator.assert_called_once()

    def test_download_result_dataclass(self) -> None:
        """Test DownloadResult dataclass."""
        from nautilus_quants.data.download.binance import DownloadResult

        result = DownloadResult(
            success=True,
            symbol="BTCUSDT",
            timeframe="1h",
            file_path=Path("/tmp/test.csv"),
            rows_downloaded=1000,
            start_timestamp=1704067200000,
            end_timestamp=1704153600000,
            resumed_from_checkpoint=False,
            errors=[],
        )

        assert result.success is True
        assert result.symbol == "BTCUSDT"
        assert result.rows_downloaded == 1000
        assert result.errors == []


class TestDownloadHelpers:
    """Tests for download helper functions."""

    def test_validate_symbol(self) -> None:
        """Test symbol validation."""
        from nautilus_quants.data.download.binance import _validate_symbol

        assert _validate_symbol("BTCUSDT") is True
        assert _validate_symbol("ETHUSDT") is True
        assert _validate_symbol("") is False
        assert _validate_symbol("BTC-USDT") is False  # Invalid format

    def test_validate_timeframe(self) -> None:
        """Test timeframe validation."""
        from nautilus_quants.data.download.binance import _validate_timeframe

        assert _validate_timeframe("1m") is True
        assert _validate_timeframe("5m") is True
        assert _validate_timeframe("15m") is True
        assert _validate_timeframe("1h") is True
        assert _validate_timeframe("4h") is True
        assert _validate_timeframe("1d") is True
        assert _validate_timeframe("2h") is True  # Binance supports 2h
        assert _validate_timeframe("invalid") is False

    def test_validate_date_range(self) -> None:
        """Test date range validation."""
        from nautilus_quants.data.download.binance import _validate_date_range

        assert _validate_date_range("2024-01-01", "2024-12-31") is True
        assert _validate_date_range("2024-12-31", "2024-01-01") is False  # End before start
        assert _validate_date_range("invalid", "2024-12-31") is False
        assert _validate_date_range("2024-01-01", "invalid") is False
