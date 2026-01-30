# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tardis derivative_ticker (funding rate) downloader.

Downloads funding rate data from Tardis datasets API.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from tardis_dev import datasets


@dataclass
class TardisFundingConfig:
    """Configuration for Tardis funding rate download.

    Parameters
    ----------
    exchange : str
        Tardis exchange identifier (e.g., "binance-futures").
    symbols : list[str]
        List of symbols to download.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    api_key : str
        Tardis API key.
    """

    exchange: str = "binance-futures"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    start_date: str = "2025-01-01"
    end_date: str = "2025-01-31"
    api_key: str = ""


@dataclass
class TardisDownloadResult:
    """Result of Tardis download operation.

    Parameters
    ----------
    success : bool
        Whether download succeeded.
    exchange : str
        Exchange identifier.
    symbols : list[str]
        Downloaded symbols.
    output_dir : Path
        Directory containing downloaded files.
    file_count : int
        Number of files downloaded.
    """

    success: bool
    exchange: str
    symbols: list[str]
    output_dir: Path
    file_count: int = 0
    error: str | None = None


def download_derivative_ticker(
    config: TardisFundingConfig,
    output_dir: Path | str,
) -> TardisDownloadResult:
    """Download derivative_ticker (funding rate) data from Tardis.

    Uses the tardis-dev Python client to download data.

    Args:
        config: Download configuration.
        output_dir: Directory to save downloaded files.

    Returns:
        TardisDownloadResult with download status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        datasets.download(
            exchange=config.exchange,
            data_types=["derivative_ticker"],
            from_date=config.start_date,
            to_date=config.end_date,
            symbols=config.symbols,
            api_key=config.api_key,
            download_dir=str(output_dir),
        )

        # Count downloaded files
        gz_files = list(output_dir.glob("*.csv.gz"))

        return TardisDownloadResult(
            success=True,
            exchange=config.exchange,
            symbols=config.symbols,
            output_dir=output_dir,
            file_count=len(gz_files),
        )

    except Exception as e:
        return TardisDownloadResult(
            success=False,
            exchange=config.exchange,
            symbols=config.symbols,
            output_dir=output_dir,
            error=str(e),
        )


def merge_funding_csvs(
    input_dir: Path | str,
    output_file: Path | str,
    symbol: str | None = None,
) -> int:
    """Merge multiple funding rate CSV files into single file.

    Args:
        input_dir: Directory containing .csv.gz files.
        output_file: Path for merged output CSV.
        symbol: Optional symbol filter (e.g., "BTCUSDT").

    Returns:
        Number of rows in merged file.
    """
    import gzip

    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Find files
    pattern = f"*{symbol}*.csv.gz" if symbol else "*.csv.gz"
    gz_files = sorted(input_dir.glob(pattern))

    if not gz_files:
        raise FileNotFoundError(f"No .csv.gz files found in {input_dir}")

    total_rows = 0

    with open(output_file, "w") as outf:
        for i, gz_file in enumerate(gz_files):
            with gzip.open(gz_file, "rt") as inf:
                content = inf.read()

                if i == 0:
                    # Include header from first file
                    outf.write(content)
                    total_rows += content.count("\n")
                else:
                    # Skip header for subsequent files
                    lines = content.split("\n", 1)
                    if len(lines) > 1:
                        outf.write(lines[1])
                        total_rows += lines[1].count("\n")

    return total_rows
