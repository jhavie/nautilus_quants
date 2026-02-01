"""
Download module for fetching historical data.

Provides:
- BinanceDownloader: Async downloader for K-line data with checkpoint-based resume
- download_klines: Main entry point for downloading historical K-line data
- BinanceFundingDownloader: Downloader for funding rate data from Binance API
- BinanceFundingResult: Result of BinanceFundingDownloader download operation
- FundingCheckpointManager: Checkpoint manager for funding rate downloads
"""

from nautilus_quants.data.download.binance import BinanceDownloader, download_klines
from nautilus_quants.data.download.binance_funding import (
    BinanceFundingDownloader,
    BinanceFundingResult,
    FundingCheckpointManager,
)

__all__ = [
    "BinanceDownloader",
    "download_klines",
    "BinanceFundingDownloader",
    "BinanceFundingResult",
    "FundingCheckpointManager",
]
