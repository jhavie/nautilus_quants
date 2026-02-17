"""Plot IC autocorrelation for all factors."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
print("Loading data...")
loader = CatalogDataLoader(config.catalog_path, config.bar_spec)
bars_by_instrument = loader.load_bars(config.instrument_ids)

# Compute factors
print("Computing factors...")
factor_config = load_factor_config(config.factor_config_path)
evaluator = FactorEvaluator(factor_config)
factor_data, pricing = evaluator.evaluate(bars_by_instrument)

# Compute autocorrelation for each factor
acf_results = {}
max_lag = 240

for factor_name, factor_series in factor_data.items():
    print(f"Processing {factor_name}...")

    # Get forward returns
    factor_data_df = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=pricing,
        periods=(1,),
        quantiles=5,
    )

    # Compute IC by date
    ic_by_date = al.performance.factor_information_coefficient(factor_data_df).iloc[:, 0]

    # Compute autocorrelation
    ic_clean = ic_by_date.dropna()
    if len(ic_clean) > max_lag:
        acf_vals = acf(ic_clean, nlags=max_lag, fft=True)
        acf_results[factor_name] = acf_vals

# Plot
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, (factor_name, acf_vals) in enumerate(acf_results.items()):
    ax = axes[idx]
    lags = np.arange(len(acf_vals))

    # Plot ACF
    ax.stem(lags, acf_vals, linefmt='C0-', markerfmt='C0o', basefmt='k-', use_line_collection=True)

    # Add confidence interval (95%)
    conf_int = 1.96 / np.sqrt(len(ic_clean))
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    ax.set_title(factor_name)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, max_lag + 5)

# Remove empty subplots
for idx in range(len(acf_results), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
output_path = Path('output/ic_autocorrelation.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {output_path}")
plt.close()

print("\nDone!")
