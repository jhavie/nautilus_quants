"""
Binance downloader for historical K-line data.

Provides async download with checkpoint-based resume capability.
"""

import csv
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal, Optional

import json

from binance import AsyncClient

from nautilus_quants.data.checkpoint import CheckpointManager
from nautilus_quants.data.types import DownloadCheckpoint, RawKline


# Valid timeframes supported by Binance
VALID_TIMEFRAMES = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}

# Estimated bytes per kline row in CSV (conservative estimate)
BYTES_PER_KLINE_ROW = 120

# Minimum free disk space required (100 MB)
MIN_FREE_DISK_SPACE_MB = 100


@dataclass
class DownloadResult:
    """Result of a download operation."""

    success: bool
    symbol: str
    timeframe: str
    file_path: Path
    rows_downloaded: int
    start_timestamp: int
    end_timestamp: int
    resumed_from_checkpoint: bool = False
    errors: list[str] = field(default_factory=list)


def _parse_kline(kline: list, symbol: str, timeframe: str) -> RawKline:
    """Parse Binance API kline format to RawKline.

    Binance kline format:
    [0] Open time (ms)
    [1] Open price
    [2] High price
    [3] Low price
    [4] Close price
    [5] Volume
    [6] Close time (ms)
    [7] Quote asset volume
    [8] Number of trades
    [9] Taker buy base asset volume
    [10] Taker buy quote asset volume
    [11] Ignore
    """
    return RawKline(
        timestamp=int(kline[0]),
        open=Decimal(str(kline[1])),
        high=Decimal(str(kline[2])),
        low=Decimal(str(kline[3])),
        close=Decimal(str(kline[4])),
        volume=Decimal(str(kline[5])),
        quote_volume=Decimal(str(kline[7])),
        trades_count=int(kline[8]),
        taker_buy_base_volume=Decimal(str(kline[9])),
        taker_buy_quote_volume=Decimal(str(kline[10])),
        symbol=symbol,
        timeframe=timeframe,
        exchange="binance",
    )


def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])

    multipliers = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
        "M": 30 * 24 * 60 * 60 * 1000,  # Approximate
    }

    return value * multipliers.get(unit, 60 * 1000)


def _date_to_ms(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD) to Unix milliseconds (UTC)."""
    from datetime import timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _get_output_path(
    output_dir: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> Path:
    """Generate output file path for downloaded data."""
    # Format dates without hyphens
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    # Create directory structure
    dir_path = output_dir / symbol / timeframe
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{timeframe}_{start_fmt}_{end_fmt}.csv"
    return dir_path / filename


def _validate_symbol(symbol: str) -> bool:
    """Validate symbol format."""
    if not symbol:
        return False
    # Basic validation: alphanumeric, uppercase
    return symbol.isalnum() and symbol.isupper()


def _validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe is supported."""
    return timeframe in VALID_TIMEFRAMES


def _validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return end >= start
    except ValueError:
        return False


def estimate_download_size(
    start_date: str,
    end_date: str,
    timeframe: str,
    num_symbols: int = 1,
) -> int:
    """Estimate download size in bytes.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: K-line interval (e.g., '1h', '4h')
        num_symbols: Number of symbols to download

    Returns:
        Estimated size in bytes
    """
    interval_ms = _interval_to_ms(timeframe)
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date) + 86400000  # End of day

    # Calculate expected number of klines
    duration_ms = end_ms - start_ms
    expected_klines = duration_ms // interval_ms

    # Estimate total size
    return expected_klines * num_symbols * BYTES_PER_KLINE_ROW


def check_disk_space(
    output_dir: Path | str,
    required_bytes: int,
    min_free_mb: int = MIN_FREE_DISK_SPACE_MB,
) -> tuple[bool, str]:
    """Check if sufficient disk space is available.

    Args:
        output_dir: Directory where files will be written
        required_bytes: Estimated bytes needed for download
        min_free_mb: Minimum free space to maintain (MB)

    Returns:
        Tuple of (has_sufficient_space, message)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        disk_usage = shutil.disk_usage(output_dir)
        free_bytes = disk_usage.free
        free_mb = free_bytes / (1024 * 1024)
        required_mb = required_bytes / (1024 * 1024)
        min_required_bytes = required_bytes + (min_free_mb * 1024 * 1024)

        if free_bytes < min_required_bytes:
            return (
                False,
                f"Insufficient disk space. Need ~{required_mb:.1f} MB + {min_free_mb} MB buffer, "
                f"but only {free_mb:.1f} MB available.",
            )

        return (
            True,
            f"Disk space OK: {free_mb:.1f} MB available, ~{required_mb:.1f} MB needed",
        )

    except OSError as e:
        return (False, f"Failed to check disk space: {e}")


class BinanceDownloader:
    """Async downloader for Binance historical K-line data.

    Features:
    - Async download with automatic rate limit handling
    - Checkpoint-based resume capability
    - Incremental CSV writes for memory efficiency
    - Support for both spot and futures markets
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        batch_size: int = 1000,
    ):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded CSV files
            checkpoint_dir: Directory for checkpoint files (default: output_dir/.checkpoints)
            batch_size: Number of klines to batch before saving checkpoint
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
        timeframe: str,
        start_date: str,
        end_date: str,
        market_type: Literal["spot", "futures"] = "futures",
        resume: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> DownloadResult:
        """Download historical K-line data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            timeframe: K-line interval (e.g., "1h", "4h")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market_type: "spot" or "futures"
            resume: Resume from checkpoint if available
            on_progress: Optional callback for progress updates (current, total)

        Returns:
            DownloadResult with file path and row counts

        Raises:
            ValueError: Invalid parameters
            BinanceAPIError: API request failed after retries
        """
        # Validate inputs
        if not _validate_symbol(symbol):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[f"Invalid symbol: {symbol}"],
            )

        if not _validate_timeframe(timeframe):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[f"Invalid timeframe: {timeframe}"],
            )

        if not _validate_date_range(start_date, end_date):
            return DownloadResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                file_path=Path(),
                rows_downloaded=0,
                start_timestamp=0,
                end_timestamp=0,
                errors=[f"Invalid date range: {start_date} to {end_date}"],
            )

        # Check for existing checkpoint
        checkpoint = None
        resumed_from_checkpoint = False
        actual_start_ms = _date_to_ms(start_date)

        if resume:
            checkpoint = self.checkpoint_manager.load(symbol, timeframe)
            if checkpoint:
                # Validate checkpoint matches our request
                if (
                    checkpoint.start_date == start_date
                    and checkpoint.end_date == end_date
                ):
                    # Resume from last timestamp + 1 interval
                    actual_start_ms = checkpoint.last_timestamp + _interval_to_ms(
                        timeframe
                    )
                    resumed_from_checkpoint = True
                    print(
                        f"Resuming {symbol} {timeframe} from checkpoint ({checkpoint.total_rows} rows already)"
                    )
                else:
                    # Request params changed, start fresh
                    checkpoint = None

        # Snapshot resume baseline before loop (checkpoint is reassigned on each save
        # below; using checkpoint.total_rows there would double-count rows_downloaded).
        initial_checkpoint_rows = checkpoint.total_rows if checkpoint else 0

        # Create output file path
        file_path = _get_output_path(
            self.output_dir, symbol, timeframe, start_date, end_date
        )

        # Determine write mode based on resume
        write_mode = "a" if resumed_from_checkpoint else "w"
        write_header = not resumed_from_checkpoint

        # Create async client
        client = await AsyncClient.create()

        try:
            # Fetch and save exchange info for precision lookup
            await _fetch_and_save_exchange_info(
                client, symbol, self.output_dir, market_type
            )

            # Get klines generator based on market type
            end_ms = _date_to_ms(end_date) + 86400000 - 1  # End of day

            if market_type == "futures":
                generator = await client.futures_historical_klines_generator(
                    symbol=symbol,
                    interval=timeframe,
                    start_str=actual_start_ms,
                    end_str=end_ms,
                )
            else:
                generator = await client.get_historical_klines_generator(
                    symbol=symbol,
                    interval=timeframe,
                    start_str=actual_start_ms,
                    end_str=end_ms,
                )

            # Download and write to CSV
            rows_downloaded = 0
            first_timestamp = 0
            last_timestamp = 0
            batch = []

            with open(file_path, write_mode, newline="") as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "quote_volume",
                            "trades_count",
                            "taker_buy_base_volume",
                            "taker_buy_quote_volume",
                        ]
                    )

                async for kline in generator:
                    raw_kline = _parse_kline(kline, symbol, timeframe)

                    if first_timestamp == 0:
                        first_timestamp = raw_kline.timestamp
                    last_timestamp = raw_kline.timestamp

                    writer.writerow(
                        [
                            raw_kline.timestamp,
                            str(raw_kline.open),
                            str(raw_kline.high),
                            str(raw_kline.low),
                            str(raw_kline.close),
                            str(raw_kline.volume),
                            str(raw_kline.quote_volume),
                            raw_kline.trades_count,
                            str(raw_kline.taker_buy_base_volume),
                            str(raw_kline.taker_buy_quote_volume),
                        ]
                    )

                    rows_downloaded += 1
                    batch.append(kline)

                    # Save checkpoint every batch_size klines
                    if len(batch) >= self.batch_size:
                        total_rows = initial_checkpoint_rows + rows_downloaded
                        new_checkpoint = DownloadCheckpoint(
                            symbol=symbol,
                            timeframe=timeframe,
                            exchange="binance",
                            market_type=market_type,
                            last_timestamp=last_timestamp,
                            last_updated=datetime.now(),
                            total_rows=total_rows,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        self.checkpoint_manager.save(new_checkpoint)
                        checkpoint = new_checkpoint
                        batch = []

                        if on_progress:
                            on_progress(total_rows, 0)  # Total unknown

            # Save final checkpoint
            if batch:
                total_rows = initial_checkpoint_rows + rows_downloaded
                final_checkpoint = DownloadCheckpoint(
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange="binance",
                    market_type=market_type,
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
                timeframe=timeframe,
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
                timeframe=timeframe,
                file_path=file_path,
                rows_downloaded=rows_downloaded if "rows_downloaded" in dir() else 0,
                start_timestamp=first_timestamp if "first_timestamp" in dir() else 0,
                end_timestamp=last_timestamp if "last_timestamp" in dir() else 0,
                resumed_from_checkpoint=resumed_from_checkpoint,
                errors=[str(e)],
            )

        finally:
            await client.close_connection()


async def download_klines(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    market_type: Literal["spot", "futures"] = "futures",
    output_dir: Path | str = "data/raw/binance",
    resume: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> DownloadResult:
    """Download historical K-line data from Binance.

    Convenience function that creates a BinanceDownloader and downloads data.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: K-line interval (e.g., "1h", "4h")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        market_type: "spot" or "futures"
        output_dir: Directory to save CSV files
        resume: Resume from checkpoint if available
        on_progress: Optional callback for progress updates

    Returns:
        DownloadResult with file path and row counts
    """
    downloader = BinanceDownloader(output_dir=output_dir)
    return await downloader.download(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
        resume=resume,
        on_progress=on_progress,
    )


async def _fetch_and_save_exchange_info(
    client: AsyncClient,
    symbol: str,
    output_dir: Path,
    market_type: Literal["spot", "futures"],
) -> None:
    """Fetch exchange info and save symbol precision to JSON.

    Args:
        client: Binance async client
        symbol: Trading pair symbol
        output_dir: Base output directory
        market_type: "spot" or "futures"
    """
    exchange_info_dir = output_dir / ".exchange_info"
    exchange_info_dir.mkdir(parents=True, exist_ok=True)
    precision_file = exchange_info_dir / f"{symbol}_precision.json"

    # Skip if already exists
    if precision_file.exists():
        return

    try:
        if market_type == "futures":
            info = await client.futures_exchange_info()
        else:
            info = await client.get_exchange_info()

        # Find symbol info
        for sym_info in info.get("symbols", []):
            if sym_info.get("symbol") == symbol:
                precision_data = {
                    "symbol": symbol,
                    "pricePrecision": sym_info.get("pricePrecision", 8),
                    "quantityPrecision": sym_info.get("quantityPrecision", 8),
                }
                with open(precision_file, "w") as f:
                    json.dump(precision_data, f, indent=2)
                break
    except Exception:
        # Silently ignore errors - will use default precision
        pass
