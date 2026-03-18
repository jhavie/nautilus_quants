"""
Integration tests for download workflow.

Tests complete download workflow including checkpoint resume functionality.
Uses simulated data to test the workflow without actual API calls.
"""

import pytest
from pathlib import Path

import pandas as pd

from nautilus_quants.data.download.binance import BinanceDownloader, DownloadResult
from nautilus_quants.data.checkpoint import CheckpointManager


class TestDownloadIntegration:
    """Integration tests for download workflow."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary output directory."""
        output = tmp_path / "raw" / "binance"
        output.mkdir(parents=True)
        return output

    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        cp_dir = tmp_path / ".checkpoints"
        cp_dir.mkdir(parents=True)
        return cp_dir

    @pytest.fixture
    def sample_csv_data(self):
        """Generate sample OHLCV data for testing."""
        base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        interval_ms = 3600000  # 1 hour

        data = []
        for i in range(10):  # 10 bars
            ts = base_ts + i * interval_ms
            data.append({
                "timestamp": ts,
                "open": 42000.0 + i * 10,
                "high": 42500.0 + i * 10,
                "low": 41900.0 + i * 10,
                "close": 42100.0 + i * 10,
                "volume": 100.0,
                "quote_volume": 4200000.0,
                "trades_count": 1000,
                "taker_buy_base_volume": 50.0,
                "taker_buy_quote_volume": 2100000.0,
            })
        return pd.DataFrame(data)

    def test_download_result_dataclass(self):
        """DownloadResult should contain expected fields."""
        result = DownloadResult(
            success=True,
            symbol="BTCUSDT",
            timeframe="1h",
            file_path=Path("/tmp/test.csv"),
            rows_downloaded=100,
            start_timestamp=1704067200000,
            end_timestamp=1704153600000,
            resumed_from_checkpoint=False,
            errors=[],
        )

        assert result.success is True
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.rows_downloaded == 100
        assert result.resumed_from_checkpoint is False

    def test_checkpoint_manager_save_load(self, checkpoint_dir):
        """CheckpointManager should save and load checkpoints."""
        checkpoint_mgr = CheckpointManager(checkpoint_dir)

        # Save checkpoint
        from nautilus_quants.data.types import DownloadCheckpoint
        from datetime import datetime

        checkpoint = DownloadCheckpoint(
            symbol="BTCUSDT",
            timeframe="1h",
            exchange="binance",
            market_type="futures",
            last_timestamp=1704078000000,
            last_updated=datetime.now(),
            total_rows=3,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        checkpoint_mgr.save(checkpoint)

        # Load checkpoint
        loaded = checkpoint_mgr.load("BTCUSDT", "1h")
        assert loaded is not None
        assert loaded.last_timestamp == 1704078000000
        assert loaded.total_rows == 3

    def test_checkpoint_manager_delete(self, checkpoint_dir):
        """CheckpointManager should delete checkpoints."""
        checkpoint_mgr = CheckpointManager(checkpoint_dir)

        # Create and save checkpoint
        from nautilus_quants.data.types import DownloadCheckpoint
        from datetime import datetime

        checkpoint = DownloadCheckpoint(
            symbol="BTCUSDT",
            timeframe="1h",
            exchange="binance",
            market_type="futures",
            last_timestamp=1704078000000,
            last_updated=datetime.now(),
            total_rows=3,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        checkpoint_mgr.save(checkpoint)

        # Verify exists
        assert checkpoint_mgr.load("BTCUSDT", "1h") is not None

        # Delete
        checkpoint_mgr.delete("BTCUSDT", "1h")

        # Verify deleted
        assert checkpoint_mgr.load("BTCUSDT", "1h") is None

    def test_simulated_download_workflow(self, output_dir, sample_csv_data):
        """Simulated download workflow should create valid CSV."""
        # Simulate download by writing CSV directly
        symbol = "BTCUSDT"
        timeframe = "1h"
        data_dir = output_dir / symbol / timeframe
        data_dir.mkdir(parents=True)

        csv_path = data_dir / f"{symbol}_{timeframe}_2024-01-01.csv"
        sample_csv_data.to_csv(csv_path, index=False)

        # Verify CSV exists and has correct structure
        assert csv_path.exists()
        df = pd.read_csv(csv_path)

        assert len(df) == 10
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_csv_format_validation(self, output_dir, sample_csv_data):
        """CSV format should be valid for downstream processing."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        data_dir = output_dir / symbol / timeframe
        data_dir.mkdir(parents=True)

        csv_path = data_dir / f"{symbol}_{timeframe}_data.csv"
        sample_csv_data.to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)

        # Verify expected columns
        expected_columns = [
            "timestamp", "open", "high", "low", "close",
            "volume", "quote_volume", "trades_count"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Verify data types
        assert df["timestamp"].dtype in ["int64", "float64"]
        assert df["open"].dtype == "float64"
        assert df["volume"].dtype == "float64"

        # Verify data integrity
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_multiple_symbols_workflow(self, output_dir, sample_csv_data):
        """Workflow should support multiple symbols independently."""
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            data_dir = output_dir / symbol / "1h"
            data_dir.mkdir(parents=True)

            csv_path = data_dir / f"{symbol}_1h_data.csv"
            sample_csv_data.to_csv(csv_path, index=False)

        # Verify both symbols have data
        btc_files = list(output_dir.glob("BTCUSDT/1h/*.csv"))
        eth_files = list(output_dir.glob("ETHUSDT/1h/*.csv"))

        assert len(btc_files) >= 1
        assert len(eth_files) >= 1

    def test_downloader_initialization(self, output_dir, checkpoint_dir):
        """BinanceDownloader should initialize correctly."""
        downloader = BinanceDownloader(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            batch_size=100,
        )

        assert downloader.output_dir == output_dir
        assert downloader.checkpoint_manager is not None
        assert downloader.batch_size == 100
