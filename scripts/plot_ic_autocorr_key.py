"""Plot IC autocorrelation for key factors only."""
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

# Only plot key factors
key_factors = ['volume', 'momentum_3h', 'volatility', 'corr', 'composite']
acf_results = {}
max_lag = 100

for factor_name in key_factors:
    if factor_name not in factor_data:
        continue

    print(f"Processing {factor_name}...")
    factor_series = factor_data[factor_name]

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
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (factor_name, acf_vals) in enumerate(acf_results.items()):
    ax = axes[idx]
    lags = np.arange(len(acf_vals))

    # Plot ACF
    ax.stem(lags, acf_vals, linefmt='C0-', markerfmt='C0o', basefmt='k-')

    # Add confidence interval (95%)
    n_obs = 35000  # approximate
    conf_int = 1.96 / np.sqrt(n_obs)
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    ax.set_title(factor_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, max_lag + 2)
    if idx == 0:
        ax.legend()

# Remove empty subplot
if len(acf_results) < len(axes):
    fig.delaxes(axes[-1])

plt.suptitle('IC Autocorrelation Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('output/ic_autocorrelation_key.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {output_path}")
plt.close()

print("\nDone!")
