"""
Download module for fetching historical data.

Provides:
- BinanceDownloader: Async downloader for K-line data with checkpoint-based resume
- download_klines: Main entry point for downloading historical K-line data
- TardisFundingConfig: Configuration for Tardis funding rate download
- download_derivative_ticker: Download funding rate data from Tardis
- merge_funding_csvs: Merge multiple funding rate CSV files
"""

from nautilus_quants.data.download.binance import BinanceDownloader, download_klines
from nautilus_quants.data.download.tardis import (
    TardisDownloadResult,
    TardisFundingConfig,
    download_derivative_ticker,
    merge_funding_csvs,
)

__all__ = [
    "BinanceDownloader",
    "download_klines",
    "TardisFundingConfig",
    "TardisDownloadResult",
    "download_derivative_ticker",
    "merge_funding_csvs",
]
