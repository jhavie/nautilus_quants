"""
Unit tests for the CheckpointManager.

Tests checkpoint save, load, delete, and atomic write operations.
"""

from datetime import datetime
from pathlib import Path

import pytest

from nautilus_quants.data.checkpoint import CheckpointManager
from nautilus_quants.data.types import DownloadCheckpoint


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create a temporary checkpoint directory."""
        checkpoint_path = tmp_path / ".checkpoints"
        checkpoint_path.mkdir()
        return checkpoint_path

    @pytest.fixture
    def manager(self, checkpoint_dir: Path) -> CheckpointManager:
        """Create a CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir)

    @pytest.fixture
    def sample_checkpoint(self) -> DownloadCheckpoint:
        """Create a sample checkpoint."""
        return DownloadCheckpoint(
            symbol="BTCUSDT",
            timeframe="1h",
            exchange="binance",
            market_type="futures",
            last_timestamp=1704067200000,
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
            total_rows=1000,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

    def test_save_and_load_checkpoint(
        self, manager: CheckpointManager, sample_checkpoint: DownloadCheckpoint
    ) -> None:
        """Test saving and loading a checkpoint."""
        manager.save(sample_checkpoint)

        loaded = manager.load("BTCUSDT", "1h")

        assert loaded is not None
        assert loaded.symbol == "BTCUSDT"
        assert loaded.timeframe == "1h"
        assert loaded.exchange == "binance"
        assert loaded.market_type == "futures"
        assert loaded.last_timestamp == 1704067200000
        assert loaded.total_rows == 1000
        assert loaded.start_date == "2024-01-01"
        assert loaded.end_date == "2024-12-31"

    def test_load_nonexistent_checkpoint(self, manager: CheckpointManager) -> None:
        """Test loading a checkpoint that doesn't exist."""
        loaded = manager.load("NONEXISTENT", "1h")
        assert loaded is None

    def test_delete_checkpoint(
        self, manager: CheckpointManager, sample_checkpoint: DownloadCheckpoint
    ) -> None:
        """Test deleting a checkpoint."""
        manager.save(sample_checkpoint)

        # Verify it exists
        assert manager.load("BTCUSDT", "1h") is not None

        # Delete it
        result = manager.delete("BTCUSDT", "1h")
        assert result is True

        # Verify it's gone
        assert manager.load("BTCUSDT", "1h") is None

    def test_delete_nonexistent_checkpoint(self, manager: CheckpointManager) -> None:
        """Test deleting a checkpoint that doesn't exist."""
        result = manager.delete("NONEXISTENT", "1h")
        assert result is False

    def test_list_checkpoints(self, manager: CheckpointManager) -> None:
        """Test listing all checkpoints."""
        # Create multiple checkpoints
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            checkpoint = DownloadCheckpoint(
                symbol=symbol,
                timeframe="1h",
                exchange="binance",
                market_type="futures",
                last_timestamp=1704067200000,
                last_updated=datetime.now(),
                total_rows=1000,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            manager.save(checkpoint)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        symbols = {cp.symbol for cp in checkpoints}
        assert symbols == {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

    def test_clear_all_checkpoints(self, manager: CheckpointManager) -> None:
        """Test clearing all checkpoints."""
        # Create multiple checkpoints
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            checkpoint = DownloadCheckpoint(
                symbol=symbol,
                timeframe="1h",
                exchange="binance",
                market_type="futures",
                last_timestamp=1704067200000,
                last_updated=datetime.now(),
                total_rows=1000,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            manager.save(checkpoint)

        # Clear all
        count = manager.clear_all()

        assert count == 2
        assert len(manager.list_checkpoints()) == 0

    def test_checkpoint_file_path(self, manager: CheckpointManager) -> None:
        """Test checkpoint file naming convention."""
        path = manager._get_checkpoint_path("BTCUSDT", "1h")
        assert path.name == "BTCUSDT_1h.json"

    def test_update_checkpoint(
        self, manager: CheckpointManager, sample_checkpoint: DownloadCheckpoint
    ) -> None:
        """Test updating an existing checkpoint."""
        manager.save(sample_checkpoint)

        # Update with new values
        updated_checkpoint = DownloadCheckpoint(
            symbol="BTCUSDT",
            timeframe="1h",
            exchange="binance",
            market_type="futures",
            last_timestamp=1704153600000,  # Updated timestamp
            last_updated=datetime(2024, 1, 2, 12, 0, 0),
            total_rows=2000,  # Updated row count
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        manager.save(updated_checkpoint)

        loaded = manager.load("BTCUSDT", "1h")

        assert loaded is not None
        assert loaded.last_timestamp == 1704153600000
        assert loaded.total_rows == 2000

    def test_corrupted_checkpoint_file(
        self, manager: CheckpointManager, checkpoint_dir: Path
    ) -> None:
        """Test handling of corrupted checkpoint file."""
        # Create a corrupted checkpoint file
        corrupted_path = checkpoint_dir / "BTCUSDT_1h.json"
        with open(corrupted_path, "w") as f:
            f.write("{ invalid json }")

        loaded = manager.load("BTCUSDT", "1h")
        assert loaded is None

    def test_checkpoint_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Test that checkpoint directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoints"
        assert not new_dir.exists()

        manager = CheckpointManager(new_dir)

        assert new_dir.exists()

    def test_multiple_timeframes_same_symbol(self, manager: CheckpointManager) -> None:
        """Test checkpoints for multiple timeframes of the same symbol."""
        for timeframe in ["1h", "4h", "1d"]:
            checkpoint = DownloadCheckpoint(
                symbol="BTCUSDT",
                timeframe=timeframe,
                exchange="binance",
                market_type="futures",
                last_timestamp=1704067200000,
                last_updated=datetime.now(),
                total_rows=1000,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            manager.save(checkpoint)

        # Load each timeframe
        for timeframe in ["1h", "4h", "1d"]:
            loaded = manager.load("BTCUSDT", timeframe)
            assert loaded is not None
            assert loaded.timeframe == timeframe

        # Delete one, others should remain
        manager.delete("BTCUSDT", "4h")
        assert manager.load("BTCUSDT", "1h") is not None
        assert manager.load("BTCUSDT", "4h") is None
        assert manager.load("BTCUSDT", "1d") is not None
