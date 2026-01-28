#!/usr/bin/env python3
"""
FMZ Factor Verification Script

Compare FMZ pandas-based factor calculations with our factor engine.
This validates that our factor expressions produce identical results.

FMZ Original Formulas (from https://www.fmz.com/digest-topic/9647):
  - volume: df_volume
  - momentum: (df_close - df_close.shift(3)) / df_close.shift(3)
  - volatility: (df_close / df_open).rolling(24).std()
  - corr: df_close.rolling(96).corr(df_volume)

Our Expressions (from cross_sectional_factors.yaml):
  - volume: volume
  - momentum: (close - delay(close, 3)) / delay(close, 3)
  - volatility: ts_std(close / open, 24)
  - corr: correlation(close, volume, 96)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import Bar

from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.config import load_factor_config


def load_bar_data(catalog_path: str, instrument_id: str, start: str, end: str) -> pd.DataFrame:
    """Load bar data from Nautilus catalog into pandas DataFrame."""
    catalog = ParquetDataCatalog(catalog_path)
    
    # Query bars
    bars = catalog.bars(
        instrument_ids=[instrument_id],
        start=start,
        end=end,
    )
    
    if len(bars) == 0:
        raise ValueError(f"No bars found for {instrument_id}")
    
    # Convert to DataFrame
    data = []
    for bar in bars:
        data.append({
            "timestamp": pd.Timestamp(bar.ts_event, unit="ns"),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        })
    
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    return df


def calculate_fmz_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate factors using FMZ original pandas formulas.
    
    These are the EXACT formulas from the FMZ article.
    """
    result = pd.DataFrame(index=df.index)
    
    # volume: df_volume (raw volume)
    result["fmz_volume"] = df["volume"]
    
    # momentum: (df_close - df_close.shift(3)) / df_close.shift(3)
    # 3-hour return (mean reversion signal)
    result["fmz_momentum"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3)
    
    # volatility: (df_close / df_open).rolling(24).std()
    # 24-hour volatility of close/open ratio
    result["fmz_volatility"] = (df["close"] / df["open"]).rolling(24).std()
    
    # corr: df_close.rolling(96).corr(df_volume)
    # 96-hour price-volume correlation
    result["fmz_corr"] = df["close"].rolling(96).corr(df["volume"])
    
    return result


def calculate_engine_factors(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """
    Calculate factors using our FactorEngine with expressions.
    
    Expressions from cross_sectional_factors.yaml:
      - volume: volume
      - momentum: (close - delay(close, 3)) / delay(close, 3)
      - volatility: ts_std(close / open, 24)
      - corr: correlation(close, volume, 96)
    """
    # Load factor config
    config = load_factor_config(config_path)
    
    # Initialize engine
    engine = FactorEngine(config=config, max_history=500)
    
    results = []
    
    # Process each bar
    for idx, row in df.iterrows():
        # Create a mock bar for the engine
        # We need to simulate what Nautilus does
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.model.data import BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        
        bar_type = BarType(
            instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
            bar_spec=BarSpecification(
                step=1,
                aggregation=BarAggregation.HOUR,
                price_type=PriceType.LAST,
            ),
            aggregation_source=2,  # EXTERNAL
        )
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(row["open"])),
            high=Price.from_str(str(row["high"])),
            low=Price.from_str(str(row["low"])),
            close=Price.from_str(str(row["close"])),
            volume=Quantity.from_str(str(row["volume"])),
            ts_event=int(idx.value),
            ts_init=int(idx.value),
        )
        
        # Compute factors
        factor_values = engine.on_bar(bar)
        
        if factor_values is not None:
            factor_dict = {"timestamp": idx}
            for factor_name, inst_values in factor_values.factors.items():
                for inst_id, value in inst_values.items():
                    factor_dict[f"engine_{factor_name}"] = value
            results.append(factor_dict)
        else:
            results.append({
                "timestamp": idx,
                "engine_volume": np.nan,
                "engine_momentum": np.nan,
                "engine_volatility": np.nan,
                "engine_corr": np.nan,
            })
    
    result_df = pd.DataFrame(results)
    result_df.set_index("timestamp", inplace=True)
    
    return result_df


def compare_factors(fmz_df: pd.DataFrame, engine_df: pd.DataFrame) -> pd.DataFrame:
    """Compare FMZ and engine factor calculations."""
    # Merge on index
    combined = fmz_df.join(engine_df, how="inner")
    
    # Calculate differences
    factors = ["volume", "momentum", "volatility", "corr"]
    
    for factor in factors:
        fmz_col = f"fmz_{factor}"
        engine_col = f"engine_{factor}"
        diff_col = f"diff_{factor}"
        pct_diff_col = f"pct_diff_{factor}"
        
        if fmz_col in combined.columns and engine_col in combined.columns:
            combined[diff_col] = combined[engine_col] - combined[fmz_col]
            # Percentage difference (avoid division by zero)
            combined[pct_diff_col] = np.where(
                combined[fmz_col] != 0,
                (combined[engine_col] - combined[fmz_col]) / np.abs(combined[fmz_col]) * 100,
                np.nan
            )
    
    return combined


def print_comparison_summary(comparison_df: pd.DataFrame) -> None:
    """Print summary of factor comparison."""
    factors = ["volume", "momentum", "volatility", "corr"]
    
    print("\n" + "=" * 80)
    print("FACTOR COMPARISON SUMMARY")
    print("=" * 80)
    
    all_match = True
    
    for factor in factors:
        fmz_col = f"fmz_{factor}"
        engine_col = f"engine_{factor}"
        diff_col = f"diff_{factor}"
        
        if diff_col not in comparison_df.columns:
            print(f"\n{factor.upper()}: Missing data")
            continue
        
        # Get non-NaN differences
        valid_diff = comparison_df[diff_col].dropna()
        
        if len(valid_diff) == 0:
            print(f"\n{factor.upper()}: No valid comparisons")
            continue
        
        max_abs_diff = valid_diff.abs().max()
        mean_abs_diff = valid_diff.abs().mean()
        
        # Consider match if max difference < 1e-10
        is_match = max_abs_diff < 1e-10
        
        status = "MATCH" if is_match else "MISMATCH"
        if not is_match:
            all_match = False
        
        print(f"\n{factor.upper()}: {status}")
        print(f"  Valid comparisons: {len(valid_diff)}")
        print(f"  Max absolute diff: {max_abs_diff:.2e}")
        print(f"  Mean absolute diff: {mean_abs_diff:.2e}")
        
        if not is_match:
            # Show first few mismatches
            print(f"  First 5 mismatches:")
            mismatches = comparison_df[comparison_df[diff_col].abs() > 1e-10][[fmz_col, engine_col, diff_col]].head()
            print(mismatches.to_string(index=True))
    
    print("\n" + "=" * 80)
    if all_match:
        print("ALL FACTORS MATCH!")
    else:
        print("SOME FACTORS DO NOT MATCH - INVESTIGATION NEEDED")
    print("=" * 80)


def main():
    """Main verification routine."""
    # Configuration
    catalog_path = "/Users/joe/Sync/nautilus_quants2/data/marketcap/catalog/1h"
    instrument_id = "BTCUSDT.BINANCE"
    start_date = "2022-01-01"
    end_date = "2022-01-16"  # 15 days for quick test
    config_path = str(project_root / "config" / "cross_sectional_factors.yaml")
    output_dir = Path(__file__).parent
    
    print("=" * 80)
    print("FMZ Factor Verification")
    print("=" * 80)
    print(f"Catalog: {catalog_path}")
    print(f"Instrument: {instrument_id}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Config: {config_path}")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1] Loading bar data from catalog...")
    df = load_bar_data(catalog_path, instrument_id, start_date, end_date)
    print(f"    Loaded {len(df)} bars")
    print(f"    Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Calculate FMZ factors
    print("\n[2] Calculating factors using FMZ pandas formulas...")
    fmz_factors = calculate_fmz_factors(df)
    print(f"    Calculated {len(fmz_factors.columns)} factors")
    
    # Step 3: Calculate engine factors
    print("\n[3] Calculating factors using our FactorEngine...")
    engine_factors = calculate_engine_factors(df, config_path)
    print(f"    Calculated {len(engine_factors.columns)} factors")
    
    # Step 4: Compare
    print("\n[4] Comparing results...")
    comparison = compare_factors(fmz_factors, engine_factors)
    
    # Step 5: Save to CSV
    output_file = output_dir / "factor_comparison.csv"
    comparison.to_csv(output_file)
    print(f"\n[5] Saved comparison to: {output_file}")
    
    # Also save individual results
    fmz_file = output_dir / "fmz_factors.csv"
    fmz_factors.to_csv(fmz_file)
    print(f"    FMZ factors: {fmz_file}")
    
    engine_file = output_dir / "engine_factors.csv"
    engine_factors.to_csv(engine_file)
    print(f"    Engine factors: {engine_file}")
    
    # Step 6: Print summary
    print_comparison_summary(comparison)
    
    # Return comparison for further analysis
    return comparison


if __name__ == "__main__":
    comparison = main()
