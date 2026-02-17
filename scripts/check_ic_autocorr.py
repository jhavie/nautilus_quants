"""Check IC autocorrelation structure."""
import pandas as pd
import numpy as np
from pathlib import Path
from nautilus_quants.alpha.analysis.config import load_analysis_config
from nautilus_quants.alpha.analysis.evaluator import FactorEvaluator
from nautilus_quants.alpha.data_loader import CatalogDataLoader
from nautilus_quants.factors.config import load_factor_config
from statsmodels.tsa.stattools import acf
import alphalens as al

# Load config
config = load_analysis_config('config/alpha_fmz_4factor.yaml')

# Load data
loader = CatalogDataLoader(config.catalog_path, config.bar_spec)
bars_by_instrument = loader.load_bars(config.instrument_ids)

# Compute factors
factor_config = load_factor_config(config.factor_config_path)
evaluator = FactorEvaluator(factor_config)
factor_data, pricing = evaluator.evaluate(bars_by_instrument)

# Compute IC for each factor
print("Computing IC for each factor...")
for factor_name, factor_series in factor_data.items():
    print(f"\n{'='*60}")
    print(f"Factor: {factor_name}")
    print(f"{'='*60}")

    # Get forward returns
    factor_values = factor_series.copy()
    prices = pricing.copy()

    factor_data_df = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_values,
        prices=prices,
        periods=(1,),
        quantiles=5,
    )

    # Compute IC by date
    ic_by_date = al.performance.factor_information_coefficient(factor_data_df).iloc[:, 0]

    print(f"IC: n={len(ic_by_date)}, mean={ic_by_date.mean():.4f}, std={ic_by_date.std():.4f}")

    # Check autocorrelation
    ic_clean = ic_by_date.dropna()
    if len(ic_clean) > 100:
        acf_vals = acf(ic_clean, nlags=min(500, len(ic_clean)//2), fft=True)
        print("\nAutocorrelation:")
        for lag in [1, 5, 10, 24, 48, 100, 240]:
            if lag < len(acf_vals):
                print(f"  lag {lag:3d}: {acf_vals[lag]:.4f}")
