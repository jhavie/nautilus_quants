"""
Transform module for converting processed data to Nautilus Parquet format.

Provides:
- transform_to_parquet: Main entry point for Parquet conversion
- csv_to_bars: Convert CSV rows to Nautilus Bar objects
- csv_to_funding_rate_updates: Convert funding rate CSV to FundingRateUpdate objects
- transform_funding_rates: Batch transform funding rate CSVs to catalog
- write_oi_parquet: Convert OI CSV to standalone Parquet
- transform_open_interest: Batch transform OI CSVs to Parquet
- load_oi_lookup: Load OI Parquet files as nested lookup dict
"""

from nautilus_quants.data.transform.funding_rate import (
    csv_to_funding_rate_updates,
    transform_funding_rates,
)
from nautilus_quants.data.transform.open_interest import (
    load_oi_lookup,
    transform_open_interest,
    write_oi_parquet,
)
from nautilus_quants.data.transform.parquet import csv_to_bars, transform_to_parquet

__all__ = [
    "csv_to_bars",
    "csv_to_funding_rate_updates",
    "load_oi_lookup",
    "transform_funding_rates",
    "transform_open_interest",
    "transform_to_parquet",
    "write_oi_parquet",
]
