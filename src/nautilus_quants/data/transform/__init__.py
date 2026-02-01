"""
Transform module for converting processed data to Nautilus formats.

Provides:
- transform_to_parquet: Main entry point for Parquet conversion
- csv_to_bars: Convert CSV rows to Nautilus Bar objects
- load_funding_rates: Load Binance funding rate CSV to FundingRateUpdate
"""

from nautilus_quants.data.transform.funding import load_funding_rates
from nautilus_quants.data.transform.parquet import csv_to_bars, transform_to_parquet

__all__ = [
    "transform_to_parquet",
    "csv_to_bars",
    "load_funding_rates",
]
