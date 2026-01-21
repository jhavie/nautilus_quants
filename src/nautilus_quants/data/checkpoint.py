"""
Checkpoint manager for resumable downloads.

Provides atomic save/load operations for download checkpoints.
"""

import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from nautilus_quants.data.types import DownloadCheckpoint


class CheckpointManager:
    """Manages download checkpoints for resume capability.

    Checkpoints are stored as JSON files with atomic writes to prevent
    corruption on crash or interrupt.
    """

    def __init__(self, checkpoint_dir: Path | str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, symbol: str, timeframe: str) -> Path:
        """Get the checkpoint file path for a symbol/timeframe pair."""
        return self.checkpoint_dir / f"{symbol}_{timeframe}.json"

    def load(self, symbol: str, timeframe: str) -> Optional[DownloadCheckpoint]:
        """Load checkpoint if it exists.

        Args:
            symbol: Trading pair symbol
            timeframe: K-line interval

        Returns:
            DownloadCheckpoint if exists, None otherwise
        """
        path = self._get_checkpoint_path(symbol, timeframe)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            # Parse datetime string back to datetime object
            last_updated = data.get("last_updated")
            if isinstance(last_updated, str):
                last_updated = datetime.fromisoformat(last_updated)

            return DownloadCheckpoint(
                symbol=data["symbol"],
                timeframe=data["timeframe"],
                exchange=data["exchange"],
                market_type=data["market_type"],
                last_timestamp=data["last_timestamp"],
                last_updated=last_updated,
                total_rows=data["total_rows"],
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Checkpoint file is corrupted, return None
            print(f"Warning: Corrupted checkpoint file {path}: {e}")
            return None

    def save(self, checkpoint: DownloadCheckpoint) -> None:
        """Atomically save checkpoint.

        Uses write-to-temp-then-rename pattern for crash safety.

        Args:
            checkpoint: Checkpoint to save
        """
        path = self._get_checkpoint_path(checkpoint.symbol, checkpoint.timeframe)

        # Convert to dict for JSON serialization
        data = asdict(checkpoint)
        # Convert datetime to ISO string
        if isinstance(data["last_updated"], datetime):
            data["last_updated"] = data["last_updated"].isoformat()

        # Atomic write: write to temp file first, then rename
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.checkpoint_dir,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = Path(tmp.name)

        # Atomic rename (POSIX guarantees atomicity)
        tmp_path.rename(path)

    def delete(self, symbol: str, timeframe: str) -> bool:
        """Delete checkpoint after successful completion.

        Args:
            symbol: Trading pair symbol
            timeframe: K-line interval

        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        path = self._get_checkpoint_path(symbol, timeframe)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_checkpoints(self) -> list[DownloadCheckpoint]:
        """List all existing checkpoints.

        Returns:
            List of DownloadCheckpoint objects
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.json"):
            # Parse symbol and timeframe from filename
            parts = path.stem.rsplit("_", 1)
            if len(parts) == 2:
                symbol, timeframe = parts
                checkpoint = self.load(symbol, timeframe)
                if checkpoint:
                    checkpoints.append(checkpoint)
        return checkpoints

    def clear_all(self) -> int:
        """Clear all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        count = 0
        for path in self.checkpoint_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
