"""
Download module for fetching historical data.

Provides:
- BinanceDownloader: Async downloader for K-line data with checkpoint-based resume
- download_klines: Main entry point for downloading historical K-line data
- BybitFundingRateDownloader: Sync downloader for funding rate data (Bybit)
- BybitOpenInterestDownloader: Sync downloader for open interest data (Bybit)
- download_funding_rates: Convenience function for funding rate downloads
- download_open_interest: Convenience function for open interest downloads
"""

from nautilus_quants.data.download.binance import (
    BinanceDownloader,
    download_klines,
)
from nautilus_quants.data.download.bybit_futures_data import (
    BybitFundingRateDownloader,
    BybitOpenInterestDownloader,
    download_funding_rates,
    download_open_interest,
)

__all__ = [
    "BinanceDownloader",
    "download_klines",
    "BybitFundingRateDownloader",
    "BybitOpenInterestDownloader",
    "download_funding_rates",
    "download_open_interest",
]
