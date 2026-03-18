"""
Download module for fetching historical K-line data from Binance.

Provides:
- BinanceDownloader: Async downloader with checkpoint-based resume capability
- download_klines: Main entry point for downloading historical data
"""

from nautilus_quants.data.download.binance import BinanceDownloader, download_klines

__all__ = ["BinanceDownloader", "download_klines"]
