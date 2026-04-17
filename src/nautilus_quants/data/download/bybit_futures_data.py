# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Bybit Futures data downloaders for funding rates and open interest.

Uses the official pybit SDK (sync). Provides checkpoint-based resume,
following the same patterns as the Binance K-line downloader.
"""

import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from pybit.unified_trading import HTTP

from nautilus_quants.data.checkpoint import CheckpointManager
from nautilus_quants.data.download.binance import (
    DownloadResult,
    _date_to_ms,
    _validate_date_range,
    _validate_symbol,
)
from nautilus_quants.data.types import DownloadCheckpoint

# Valid periods for open interest endpoint
VALID_OI_PERIODS = {
    "5min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1d",
}


class BybitFundingRateDownloader:
    """Sync downloader for Bybit Futures funding rate data.

    Features:
    - Automatic pagination (timestamp-based, newest-first)
    - Checkpoint-based resume capability
    - Incremental CSV writes for memory efficiency
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        batch_size: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.batch_size = batch_size

    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> DownloadResult:
        """Download historical funding rate data from Bybit Futures.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DownloadResult with file path and row counts
        """
        timeframe_key = "funding_rate"
        checkpoint_key = f"{symbol}_fr"

        if not _validate_symbol(symbol):
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=Path(), rows_downloaded=0,
                start_timestamp=0, end_timestamp=0,
                errors=[f"Invalid symbol: {symbol}"],
            )

        if not _validate_date_range(start_date, end_date):
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=Path(), rows_downloaded=0,
                start_timestamp=0, end_timestamp=0,
                errors=[f"Invalid date range: {start_date} to {end_date}"],
            )

        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        start_ms = _date_to_ms(start_date)
        end_ms = _date_to_ms(end_date) + 86400000 - 1

        checkpoint = self.checkpoint_manager.load(checkpoint_key, timeframe_key)
        if checkpoint:
            if checkpoint.start_date == start_date and checkpoint.end_date == end_date:
                # Resume: only fetch data after the checkpoint
                start_ms = checkpoint.last_timestamp + 1
                resumed_from_checkpoint = True
                print(
                    f"Resuming {symbol} funding_rate from checkpoint "
                    f"({checkpoint.total_rows} rows already)"
                )
            else:
                checkpoint = None

        # Snapshot resume baseline before loop (checkpoint is reassigned on each
        # batch save below; re-reading its total_rows would double-count).
        initial_checkpoint_rows = checkpoint.total_rows if checkpoint else 0

        # Output file
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        dir_path = self.output_dir / symbol / "funding_rate"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{symbol}_fr_{start_fmt}_{end_fmt}.csv"

        write_mode = "a" if resumed_from_checkpoint else "w"
        write_header = not resumed_from_checkpoint

        session = HTTP()

        try:
            rows_downloaded = 0
            first_timestamp = 0
            last_timestamp = 0
            batch: list[dict] = []

            # Bybit FR API returns newest-first. We paginate by moving
            # endTime backwards. Collect all rows, then write sorted.
            all_rows: list[tuple[int, str]] = []
            current_end_ms = end_ms

            while current_end_ms >= start_ms:
                resp = session.get_funding_rate_history(
                    category="linear",
                    symbol=symbol,
                    startTime=start_ms,
                    endTime=current_end_ms,
                    limit=200,
                )

                records = resp.get("result", {}).get("list", [])
                if not records:
                    break

                for record in records:
                    ts = int(record["fundingRateTimestamp"])
                    fr = record["fundingRate"]
                    all_rows.append((ts, fr))

                # Move endTime before the oldest record in this page
                oldest_ts = int(records[-1]["fundingRateTimestamp"])
                current_end_ms = oldest_ts - 1

                if len(records) < 200:
                    break

            # Sort ascending by timestamp and deduplicate
            all_rows.sort(key=lambda r: r[0])
            seen: set[int] = set()
            unique_rows: list[tuple[int, str]] = []
            for row in all_rows:
                if row[0] not in seen:
                    seen.add(row[0])
                    unique_rows.append(row)

            # Write to CSV
            with open(file_path, write_mode, newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "funding_rate"])

                for ts, fr in unique_rows:
                    writer.writerow([ts, fr])
                    rows_downloaded += 1

                    if first_timestamp == 0:
                        first_timestamp = ts
                    last_timestamp = ts

                    batch.append({"ts": ts})
                    if len(batch) >= self.batch_size:
                        total_rows = initial_checkpoint_rows + rows_downloaded
                        new_cp = DownloadCheckpoint(
                            symbol=checkpoint_key, timeframe=timeframe_key,
                            exchange="bybit", market_type="futures",
                            last_timestamp=last_timestamp,
                            last_updated=datetime.now(),
                            total_rows=total_rows,
                            start_date=start_date, end_date=end_date,
                        )
                        self.checkpoint_manager.save(new_cp)
                        checkpoint = new_cp
                        batch = []

            # Final checkpoint
            if batch:
                total_rows = initial_checkpoint_rows + rows_downloaded
                self.checkpoint_manager.save(DownloadCheckpoint(
                    symbol=checkpoint_key, timeframe=timeframe_key,
                    exchange="bybit", market_type="futures",
                    last_timestamp=last_timestamp,
                    last_updated=datetime.now(),
                    total_rows=total_rows,
                    start_date=start_date, end_date=end_date,
                ))

            return DownloadResult(
                success=True, symbol=symbol, timeframe=timeframe_key,
                file_path=file_path, rows_downloaded=rows_downloaded,
                start_timestamp=first_timestamp, end_timestamp=last_timestamp,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )

        except Exception as e:
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=rows_downloaded if "rows_downloaded" in dir() else 0,
                start_timestamp=first_timestamp if "first_timestamp" in dir() else 0,
                end_timestamp=last_timestamp if "last_timestamp" in dir() else 0,
                resumed_from_checkpoint=resumed_from_checkpoint,
                errors=[str(e)],
            )


class BybitOpenInterestDownloader:
    """Sync downloader for Bybit Futures open interest data.

    Features:
    - Automatic cursor-based pagination
    - Checkpoint-based resume capability
    - Incremental CSV writes for memory efficiency
    - Configurable period (5min to 1d)
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        batch_size: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.batch_size = batch_size

    def download(
        self,
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
    ) -> DownloadResult:
        """Download historical open interest data from Bybit Futures.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            period: Data period ("5min", "15min", "30min", "1h",
                "2h", "4h", "6h", "12h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DownloadResult with file path and row counts
        """
        timeframe_key = f"oi_{period}"
        checkpoint_key = f"{symbol}_oi_{period}"

        if not _validate_symbol(symbol):
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=Path(), rows_downloaded=0,
                start_timestamp=0, end_timestamp=0,
                errors=[f"Invalid symbol: {symbol}"],
            )

        if period not in VALID_OI_PERIODS:
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=Path(), rows_downloaded=0,
                start_timestamp=0, end_timestamp=0,
                errors=[
                    f"Invalid period: {period}. "
                    f"Must be one of {sorted(VALID_OI_PERIODS)}"
                ],
            )

        if not _validate_date_range(start_date, end_date):
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=Path(), rows_downloaded=0,
                start_timestamp=0, end_timestamp=0,
                errors=[f"Invalid date range: {start_date} to {end_date}"],
            )

        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        start_ms = _date_to_ms(start_date)
        end_ms = _date_to_ms(end_date) + 86400000 - 1

        checkpoint = self.checkpoint_manager.load(checkpoint_key, timeframe_key)
        if checkpoint:
            if checkpoint.start_date == start_date and checkpoint.end_date == end_date:
                start_ms = checkpoint.last_timestamp + 1
                resumed_from_checkpoint = True
                print(
                    f"Resuming {symbol} oi_{period} from checkpoint "
                    f"({checkpoint.total_rows} rows already)"
                )
            else:
                checkpoint = None

        # Snapshot resume baseline before loop (checkpoint is reassigned on each
        # batch save below; re-reading its total_rows would double-count).
        initial_checkpoint_rows = checkpoint.total_rows if checkpoint else 0

        # Output file
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        dir_path = self.output_dir / symbol / "open_interest"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{symbol}_oi_{period}_{start_fmt}_{end_fmt}.csv"

        write_mode = "a" if resumed_from_checkpoint else "w"
        write_header = not resumed_from_checkpoint

        session = HTTP()

        try:
            rows_downloaded = 0
            first_timestamp = 0
            last_timestamp = 0
            batch: list[dict] = []

            # Bybit OI API returns newest-first with cursor pagination.
            # Collect all, sort ascending, then write.
            all_rows: list[tuple[int, str]] = []
            current_end_ms = end_ms
            cursor = ""

            while True:
                kwargs: dict = {
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": period,
                    "startTime": start_ms,
                    "endTime": current_end_ms,
                    "limit": 200,
                }
                if cursor:
                    kwargs["cursor"] = cursor

                resp = session.get_open_interest(**kwargs)

                records = resp.get("result", {}).get("list", [])
                if not records:
                    break

                for record in records:
                    ts = int(record["timestamp"])
                    oi = record["openInterest"]
                    all_rows.append((ts, oi))

                cursor = resp.get("result", {}).get("nextPageCursor", "")
                if not cursor or len(records) < 200:
                    break

            # Sort ascending and deduplicate
            all_rows.sort(key=lambda r: r[0])
            seen: set[int] = set()
            unique_rows: list[tuple[int, str]] = []
            for row in all_rows:
                if row[0] not in seen:
                    seen.add(row[0])
                    unique_rows.append(row)

            # Write to CSV
            with open(file_path, write_mode, newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "open_interest"])

                for ts, oi in unique_rows:
                    writer.writerow([ts, oi])
                    rows_downloaded += 1

                    if first_timestamp == 0:
                        first_timestamp = ts
                    last_timestamp = ts

                    batch.append({"ts": ts})
                    if len(batch) >= self.batch_size:
                        total_rows = initial_checkpoint_rows + rows_downloaded
                        new_cp = DownloadCheckpoint(
                            symbol=checkpoint_key, timeframe=timeframe_key,
                            exchange="bybit", market_type="futures",
                            last_timestamp=last_timestamp,
                            last_updated=datetime.now(),
                            total_rows=total_rows,
                            start_date=start_date, end_date=end_date,
                        )
                        self.checkpoint_manager.save(new_cp)
                        checkpoint = new_cp
                        batch = []

            # Final checkpoint
            if batch:
                total_rows = initial_checkpoint_rows + rows_downloaded
                self.checkpoint_manager.save(DownloadCheckpoint(
                    symbol=checkpoint_key, timeframe=timeframe_key,
                    exchange="bybit", market_type="futures",
                    last_timestamp=last_timestamp,
                    last_updated=datetime.now(),
                    total_rows=total_rows,
                    start_date=start_date, end_date=end_date,
                ))

            return DownloadResult(
                success=True, symbol=symbol, timeframe=timeframe_key,
                file_path=file_path, rows_downloaded=rows_downloaded,
                start_timestamp=first_timestamp, end_timestamp=last_timestamp,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )

        except Exception as e:
            return DownloadResult(
                success=False, symbol=symbol, timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=rows_downloaded if "rows_downloaded" in dir() else 0,
                start_timestamp=first_timestamp if "first_timestamp" in dir() else 0,
                end_timestamp=last_timestamp if "last_timestamp" in dir() else 0,
                resumed_from_checkpoint=resumed_from_checkpoint,
                errors=[str(e)],
            )


def download_funding_rates(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path | str = "data/raw/bybit",
) -> DownloadResult:
    """Download historical funding rate data from Bybit Futures."""
    downloader = BybitFundingRateDownloader(output_dir=output_dir)
    return downloader.download(symbol=symbol, start_date=start_date, end_date=end_date)


def download_open_interest(
    symbol: str,
    period: str,
    start_date: str,
    end_date: str,
    output_dir: Path | str = "data/raw/bybit",
) -> DownloadResult:
    """Download historical open interest data from Bybit Futures."""
    downloader = BybitOpenInterestDownloader(output_dir=output_dir)
    return downloader.download(
        symbol=symbol, period=period, start_date=start_date, end_date=end_date,
    )
