"""
Download module for fetching historical data from Binance.

Provides:
- BinanceDownloader: Async downloader with checkpoint-based resume capability
- download_klines: Main entry point for downloading historical K-line data
- FundingRateDownloader: Async downloader for funding rate data
- OpenInterestDownloader: Async downloader for open interest data
- download_funding_rates: Convenience function for funding rate downloads
- download_open_interest: Convenience function for open interest downloads
"""

from nautilus_quants.data.download.binance import (
    BinanceDownloader,
    download_klines,
)
from nautilus_quants.data.download.binance_futures_data import (
    FundingRateDownloader,
    OpenInterestDownloader,
    download_funding_rates,
    download_open_interest,
)

__all__ = [
    "BinanceDownloader",
    "download_klines",
    "FundingRateDownloader",
    "OpenInterestDownloader",
    "download_funding_rates",
    "download_open_interest",
]
