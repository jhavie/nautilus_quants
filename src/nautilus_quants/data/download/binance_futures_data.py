# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Binance Futures data downloaders for funding rates and open interest.

Provides async download with checkpoint-based resume capability,
following the same patterns as the K-line downloader.
"""

import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from binance import AsyncClient

from nautilus_quants.data.checkpoint import CheckpointManager
from nautilus_quants.data.download.binance import (
    DownloadResult,
    _date_to_ms,
    _validate_date_range,
    _validate_symbol,
)
from nautilus_quants.data.types import DownloadCheckpoint

# Valid periods for open interest history endpoint
VALID_OI_PERIODS = {
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d",
}


class FundingRateDownloader:
    """Async downloader for Binance Futures funding rate data.

    Features:
    - Async download with automatic pagination
    - Checkpoint-based resume capability
    - Incremental CSV writes for memory efficiency
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        batch_size: int = 500,
    ):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded CSV files
            checkpoint_dir: Directory for checkpoint files
                (default: output_dir/.checkpoints)
            batch_size: Number of rows to batch before saving checkpoint
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.batch_size = batch_size

    async def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> DownloadResult:
        """Download historical funding rate data from Binance Futures.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DownloadResult with file path and row counts
        """
        timeframe_key = "funding_rate"
        checkpoint_key = f"{symbol}_fr"

        # Validate inputs
        if not _validate_symbol(symbol):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[f"Invalid symbol: {symbol}"],
            )

        if not _validate_date_range(start_date, end_date):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[
                    f"Invalid date range: {start_date} to {end_date}"
                ],
            )

        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        actual_start_ms = _date_to_ms(start_date)

        checkpoint = self.checkpoint_manager.load(
            checkpoint_key, timeframe_key
        )
        if checkpoint:
            if (
                checkpoint.start_date == start_date
                and checkpoint.end_date == end_date
            ):
                actual_start_ms = checkpoint.last_timestamp + 1
                resumed_from_checkpoint = True
                print(
                    f"Resuming {symbol} funding_rate from checkpoint "
                    f"({checkpoint.total_rows} rows already)"
                )
            else:
                checkpoint = None

        # Create output file path
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        dir_path = self.output_dir / symbol / "funding_rate"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{symbol}_fr_{start_fmt}_{end_fmt}.csv"

        # Determine write mode based on resume
        write_mode = "a" if resumed_from_checkpoint else "w"
        write_header = not resumed_from_checkpoint

        end_ms = _date_to_ms(end_date) + 86400000 - 1  # End of day

        client = await AsyncClient.create()

        try:
            rows_downloaded = 0
            first_timestamp = 0
            last_timestamp = 0
            batch: list[dict] = []
            current_start_ms = actual_start_ms

            with open(file_path, write_mode, newline="") as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow([
                        "timestamp",
                        "funding_rate",
                        "mark_price",
                    ])

                while current_start_ms <= end_ms:
                    records = await client.futures_funding_rate(
                        symbol=symbol,
                        startTime=current_start_ms,
                        endTime=end_ms,
                        limit=1000,
                    )

                    if not records:
                        break

                    for record in records:
                        ts = int(record["fundingTime"])
                        fr = Decimal(str(record["fundingRate"]))
                        mp = Decimal(str(record["markPrice"]))

                        if first_timestamp == 0:
                            first_timestamp = ts
                        last_timestamp = ts

                        writer.writerow([ts, str(fr), str(mp)])
                        rows_downloaded += 1
                        batch.append(record)

                        if len(batch) >= self.batch_size:
                            total_rows = (
                                checkpoint.total_rows
                                if checkpoint
                                else 0
                            ) + rows_downloaded
                            new_checkpoint = DownloadCheckpoint(
                                symbol=checkpoint_key,
                                timeframe=timeframe_key,
                                exchange="binance",
                                market_type="futures",
                                last_timestamp=last_timestamp,
                                last_updated=datetime.now(),
                                total_rows=total_rows,
                                start_date=start_date,
                                end_date=end_date,
                            )
                            self.checkpoint_manager.save(
                                new_checkpoint
                            )
                            checkpoint = new_checkpoint
                            batch = []

                    # Paginate: next page starts after last record
                    last_record_ts = int(
                        records[-1]["fundingTime"]
                    )
                    current_start_ms = last_record_ts + 1

                    # If fewer records than limit, we've reached the end
                    if len(records) < 1000:
                        break

            # Save final checkpoint
            if batch:
                total_rows = (
                    checkpoint.total_rows if checkpoint else 0
                ) + rows_downloaded
                final_checkpoint = DownloadCheckpoint(
                    symbol=checkpoint_key,
                    timeframe=timeframe_key,
                    exchange="binance",
                    market_type="futures",
                    last_timestamp=last_timestamp,
                    last_updated=datetime.now(),
                    total_rows=total_rows,
                    start_date=start_date,
                    end_date=end_date,
                )
                self.checkpoint_manager.save(final_checkpoint)

            return DownloadResult(
                success=True,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=rows_downloaded,
                start_timestamp=first_timestamp,
                end_timestamp=last_timestamp,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )

        except Exception as e:
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=(
                    rows_downloaded
                    if "rows_downloaded" in dir()
                    else 0
                ),
                start_timestamp=(
                    first_timestamp
                    if "first_timestamp" in dir()
                    else 0
                ),
                end_timestamp=(
                    last_timestamp
                    if "last_timestamp" in dir()
                    else 0
                ),
                resumed_from_checkpoint=resumed_from_checkpoint,
                errors=[str(e)],
            )

        finally:
            await client.close_connection()


class OpenInterestDownloader:
    """Async downloader for Binance Futures open interest data.

    Features:
    - Async download with automatic pagination
    - Checkpoint-based resume capability
    - Incremental CSV writes for memory efficiency
    - Configurable period (5m to 1d)
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        batch_size: int = 500,
    ):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded CSV files
            checkpoint_dir: Directory for checkpoint files
                (default: output_dir/.checkpoints)
            batch_size: Number of rows to batch before saving checkpoint
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.batch_size = batch_size

    async def download(
        self,
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
    ) -> DownloadResult:
        """Download historical open interest data from Binance Futures.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            period: Data period ("5m", "15m", "30m", "1h", "2h",
                "4h", "6h", "12h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DownloadResult with file path and row counts
        """
        timeframe_key = f"oi_{period}"
        checkpoint_key = f"{symbol}_oi_{period}"

        # Validate inputs
        if not _validate_symbol(symbol):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[f"Invalid symbol: {symbol}"],
            )

        if period not in VALID_OI_PERIODS:
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[
                    f"Invalid period: {period}. "
                    f"Must be one of {sorted(VALID_OI_PERIODS)}"
                ],
            )

        if not _validate_date_range(start_date, end_date):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[
                    f"Invalid date range: {start_date} to {end_date}"
                ],
            )

        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        actual_start_ms = _date_to_ms(start_date)

        checkpoint = self.checkpoint_manager.load(
            checkpoint_key, timeframe_key
        )
        if checkpoint:
            if (
                checkpoint.start_date == start_date
                and checkpoint.end_date == end_date
            ):
                actual_start_ms = checkpoint.last_timestamp + 1
                resumed_from_checkpoint = True
                print(
                    f"Resuming {symbol} oi_{period} from checkpoint "
                    f"({checkpoint.total_rows} rows already)"
                )
            else:
                checkpoint = None

        # Create output file path
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        dir_path = self.output_dir / symbol / "open_interest"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = (
            dir_path
            / f"{symbol}_oi_{period}_{start_fmt}_{end_fmt}.csv"
        )

        # Determine write mode based on resume
        write_mode = "a" if resumed_from_checkpoint else "w"
        write_header = not resumed_from_checkpoint

        end_ms = _date_to_ms(end_date) + 86400000 - 1  # End of day

        client = await AsyncClient.create()

        try:
            rows_downloaded = 0
            first_timestamp = 0
            last_timestamp = 0
            batch: list[dict] = []
            current_start_ms = actual_start_ms

            with open(file_path, write_mode, newline="") as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow([
                        "timestamp",
                        "open_interest",
                        "open_interest_value",
                    ])

                while current_start_ms <= end_ms:
                    records = (
                        await client.futures_open_interest_hist(
                            symbol=symbol,
                            period=period,
                            startTime=current_start_ms,
                            endTime=end_ms,
                            limit=500,
                        )
                    )

                    if not records:
                        break

                    for record in records:
                        ts = int(record["timestamp"])
                        oi = Decimal(
                            str(record["sumOpenInterest"])
                        )
                        oi_val = Decimal(
                            str(record["sumOpenInterestValue"])
                        )

                        if first_timestamp == 0:
                            first_timestamp = ts
                        last_timestamp = ts

                        writer.writerow([
                            ts, str(oi), str(oi_val)
                        ])
                        rows_downloaded += 1
                        batch.append(record)

                        if len(batch) >= self.batch_size:
                            total_rows = (
                                checkpoint.total_rows
                                if checkpoint
                                else 0
                            ) + rows_downloaded
                            new_checkpoint = DownloadCheckpoint(
                                symbol=checkpoint_key,
                                timeframe=timeframe_key,
                                exchange="binance",
                                market_type="futures",
                                last_timestamp=last_timestamp,
                                last_updated=datetime.now(),
                                total_rows=total_rows,
                                start_date=start_date,
                                end_date=end_date,
                            )
                            self.checkpoint_manager.save(
                                new_checkpoint
                            )
                            checkpoint = new_checkpoint
                            batch = []

                    # Paginate: next page starts after last record
                    last_record_ts = int(
                        records[-1]["timestamp"]
                    )
                    current_start_ms = last_record_ts + 1

                    # If fewer records than limit, we've reached end
                    if len(records) < 500:
                        break

            # Save final checkpoint
            if batch:
                total_rows = (
                    checkpoint.total_rows if checkpoint else 0
                ) + rows_downloaded
                final_checkpoint = DownloadCheckpoint(
                    symbol=checkpoint_key,
                    timeframe=timeframe_key,
                    exchange="binance",
                    market_type="futures",
                    last_timestamp=last_timestamp,
                    last_updated=datetime.now(),
                    total_rows=total_rows,
                    start_date=start_date,
                    end_date=end_date,
                )
                self.checkpoint_manager.save(final_checkpoint)

            return DownloadResult(
                success=True,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=rows_downloaded,
                start_timestamp=first_timestamp,
                end_timestamp=last_timestamp,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )

        except Exception as e:
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe_key,
                file_path=file_path,
                rows_downloaded=(
                    rows_downloaded
                    if "rows_downloaded" in dir()
                    else 0
                ),
                start_timestamp=(
                    first_timestamp
                    if "first_timestamp" in dir()
                    else 0
                ),
                end_timestamp=(
                    last_timestamp
                    if "last_timestamp" in dir()
                    else 0
                ),
                resumed_from_checkpoint=resumed_from_checkpoint,
                errors=[str(e)],
            )

        finally:
            await client.close_connection()


async def download_funding_rates(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path | str = "data/raw/binance",
) -> DownloadResult:
    """Download historical funding rate data from Binance Futures.

    Convenience function that creates a FundingRateDownloader
    and downloads data.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save CSV files

    Returns:
        DownloadResult with file path and row counts
    """
    downloader = FundingRateDownloader(output_dir=output_dir)
    return await downloader.download(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


async def download_open_interest(
    symbol: str,
    period: str,
    start_date: str,
    end_date: str,
    output_dir: Path | str = "data/raw/binance",
) -> DownloadResult:
    """Download historical open interest data from Binance Futures.

    Convenience function that creates an OpenInterestDownloader
    and downloads data.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        period: Data period ("5m", "15m", "30m", "1h", "2h",
            "4h", "6h", "12h", "1d")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save CSV files

    Returns:
        DownloadResult with file path and row counts
    """
    downloader = OpenInterestDownloader(output_dir=output_dir)
    return await downloader.download(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
    )
