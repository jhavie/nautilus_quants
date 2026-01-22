"""Pytest configuration and shared fixtures."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

from nautilus_quants.backtest.engine import create_crypto_perpetual


@pytest.fixture(scope="session")
def backtest_catalog_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a ParquetDataCatalog from fixture parquet files.
    
    This fixture converts the raw parquet files in tests/fixtures/backtest/
    into proper Nautilus ParquetDataCatalog format.
    
    Returns:
        Path to the catalog directory
    """
    # Create catalog in session-scoped temp directory
    catalog_path = tmp_path_factory.mktemp("catalog")
    
    # Source fixture path
    fixture_path = Path(__file__).parent / "fixtures" / "backtest"
    
    # Load fixture data
    df = pd.read_parquet(fixture_path / "BTCUSDT_1h.parquet")
    
    # Prepare DataFrame for BarDataWrangler (expects timestamp index)
    df_prepared = df.copy()
    df_prepared = df_prepared.set_index("datetime")
    
    # Use the same instrument that BacktestRunner uses (size_precision=3)
    # This ensures consistency between catalog data and runtime instrument
    instrument = create_crypto_perpetual(
        instrument_id="BTCUSDT",
        venue="BINANCE",
        maker_fee=0.0002,
        taker_fee=0.0004,
    )
    
    # Round volume to match instrument.size_precision (3 decimal places)
    # This fixes: "invalid bar.volume.precision=6 did not match instrument.size_precision=3"
    df_prepared["volume"] = df_prepared["volume"].round(instrument.size_precision)
    
    # Create bar type
    bar_type = BarType.from_str("BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL")
    
    # Create wrangler and process
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(df_prepared)
    
    # Create catalog
    catalog = ParquetDataCatalog(catalog_path)
    
    # Write instrument and bars
    catalog.write_data([instrument])
    catalog.write_data(bars)
    
    return catalog_path
