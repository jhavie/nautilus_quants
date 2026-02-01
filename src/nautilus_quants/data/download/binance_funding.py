# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Binance funding rate downloader using official API.

Downloads historical funding rate data from Binance Futures public API.
No API key required.
"""

import csv
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from binance import AsyncClient

from nautilus_quants.data.types import FundingCheckpoint


@dataclass
class BinanceFundingResult:
    """Result of Binance funding rate download operation."""

    success: bool
    symbol: str
    file_path: Path
    rows_downloaded: int
    resumed_from_checkpoint: bool = False
    error: str | None = None


class FundingCheckpointManager:
    """Manages funding rate download checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: Path | str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, symbol: str) -> Path:
        """Get the checkpoint file path for a symbol."""
        return self.checkpoint_dir / f"{symbol}_funding.json"

    def load(self, symbol: str) -> Optional[FundingCheckpoint]:
        """Load checkpoint if it exists."""
        path = self._get_checkpoint_path(symbol)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            last_updated = data.get("last_updated")
            if isinstance(last_updated, str):
                last_updated = datetime.fromisoformat(last_updated)

            return FundingCheckpoint(
                symbol=data["symbol"],
                exchange=data["exchange"],
                last_date=data["last_date"],
                last_updated=last_updated,
                total_rows=data["total_rows"],
                start_date=data["start_date"],
                end_date=data["end_date"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Corrupted checkpoint file {path}: {e}")
            return None

    def save(self, checkpoint: FundingCheckpoint) -> None:
        """Atomically save checkpoint."""
        path = self._get_checkpoint_path(checkpoint.symbol)

        data = asdict(checkpoint)
        if isinstance(data["last_updated"], datetime):
            data["last_updated"] = data["last_updated"].isoformat()

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.checkpoint_dir,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = Path(tmp.name)

        tmp_path.rename(path)

    def delete(self, symbol: str) -> bool:
        """Delete checkpoint after successful completion."""
        path = self._get_checkpoint_path(symbol)
        if path.exists():
            path.unlink()
            return True
        return False


def _date_to_ms(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD) to Unix milliseconds (UTC)."""
    from datetime import timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    """Convert Unix milliseconds to date string (YYYY-MM-DD)."""
    from datetime import timezone

    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


class BinanceFundingDownloader:
    """Binance official API funding rate downloader.

    Features:
    - No API key required (public endpoint)
    - 8-hour interval data (settlement times)
    - Automatic pagination for large date ranges
    - Checkpoint-based resume capability
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Path | str | None = None,
    ):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded CSV files
            checkpoint_dir: Directory for checkpoint files (default: output_dir/.checkpoints)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_manager = FundingCheckpointManager(checkpoint_dir)

    async def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        resume: bool = True,
    ) -> BinanceFundingResult:
        """Download funding rate history from Binance.

        Uses GET /fapi/v1/fundingRate endpoint.
        - No API key required
        - Max 1000 records per request
        - 8-hour settlement interval (~3 records per day)

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resume: Resume from checkpoint if available

        Returns:
            BinanceFundingResult with file path and row counts
        """
        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        actual_start_date = start_date

        if resume:
            checkpoint = self.checkpoint_manager.load(symbol)
            if checkpoint:
                if (
                    checkpoint.start_date == start_date
                    and checkpoint.end_date == end_date
                    and checkpoint.exchange == "binance"
                ):
                    # Resume from day after last_date
                    last_dt = datetime.strptime(checkpoint.last_date, "%Y-%m-%d")
                    next_dt = last_dt + timedelta(days=1)
                    actual_start_date = next_dt.strftime("%Y-%m-%d")
                    resumed_from_checkpoint = True
                    print(
                        f"Resuming {symbol} funding from {actual_start_date} "
                        f"({checkpoint.total_rows} rows already)"
                    )
                else:
                    checkpoint = None

        # Check if already complete
        if actual_start_date > end_date:
            print(f"{symbol} funding already complete")
            output_file = self._get_output_path(symbol, start_date, end_date)
            return BinanceFundingResult(
                success=True,
                symbol=symbol,
                file_path=output_file,
                rows_downloaded=checkpoint.total_rows if checkpoint else 0,
                resumed_from_checkpoint=True,
            )

        output_file = self._get_output_path(symbol, start_date, end_date)

        try:
            rows_downloaded = await self._fetch_funding_rates(
                symbol=symbol,
                start_date=actual_start_date,
                end_date=end_date,
                output_file=output_file,
                append=resumed_from_checkpoint,
            )

            # Calculate total rows
            total_rows = rows_downloaded
            if checkpoint and resumed_from_checkpoint:
                total_rows = checkpoint.total_rows + rows_downloaded

            # Save checkpoint
            new_checkpoint = FundingCheckpoint(
                symbol=symbol,
                exchange="binance",
                last_date=end_date,
                last_updated=datetime.now(),
                total_rows=total_rows,
                start_date=start_date,
                end_date=end_date,
            )
            self.checkpoint_manager.save(new_checkpoint)

            return BinanceFundingResult(
                success=True,
                symbol=symbol,
                file_path=output_file,
                rows_downloaded=total_rows,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )

        except Exception as e:
            return BinanceFundingResult(
                success=False,
                symbol=symbol,
                file_path=output_file if "output_file" in dir() else Path(),
                rows_downloaded=0,
                resumed_from_checkpoint=resumed_from_checkpoint,
                error=str(e),
            )

    async def _fetch_funding_rates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        output_file: Path,
        append: bool = False,
    ) -> int:
        """Fetch funding rates using python-binance AsyncClient.

        Args:
            symbol: Trading pair symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_file: Path to save CSV
            append: Append to existing file

        Returns:
            Number of rows downloaded
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        start_ms = _date_to_ms(start_date)
        end_ms = _date_to_ms(end_date) + 86400000 - 1  # End of day

        all_records = []
        current_start = start_ms

        client = await AsyncClient.create()
        try:
            while current_start < end_ms:
                data = await client.futures_funding_rate(
                    symbol=symbol,
                    startTime=current_start,
                    endTime=end_ms,
                    limit=1000,
                )

                if not data:
                    break

                all_records.extend(data)
                current_start = data[-1]["fundingTime"] + 1

                if len(data) < 1000:
                    break
        finally:
            await client.close_connection()

        # Write to CSV
        write_mode = "a" if append else "w"
        write_header = not append or not output_file.exists()

        with open(output_file, write_mode, newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["symbol", "funding_time", "funding_rate", "mark_price"])

            for record in all_records:
                writer.writerow([
                    record["symbol"],
                    record["fundingTime"],
                    record["fundingRate"],
                    record.get("markPrice", ""),
                ])

        return len(all_records)

    def _get_output_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate output file path for funding rate data."""
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")

        dir_path = self.output_dir / symbol / "funding"
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_funding_{start_fmt}_{end_fmt}.csv"
        return dir_path / filename
