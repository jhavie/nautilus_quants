# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Integration tests for Alpha2 8-factor composite — individual factor verification.

Loads real 12-coin BinanceBar 4h data from the catalog parquet files and feeds
them into PanelFactorEngine directly (bypassing the Nautilus backtest framework)
to verify that each factor in factors_alpha2.yaml produces non-empty results.

Data source: /Users/joe/Sync/nautilus_quants2/data/12coin_catalog
  - 12 coins × 4h bars, 2025-01-01 ~ 2025-03-31 (~540 bars/coin, ~90 days)
  - Includes quote_volume, count extra fields

Test sequence:
  1. Each raw factor independently (alpha044 through alpha003)
  2. Each normalized factor (alpha044_norm through alpha003_norm)
  3. Complete composite factor
  4. FactorValues publish content verification
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_DIR = Path("/Users/joe/Sync/nautilus_quants2/data/12coin_catalog/data/binance_bar")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "fmz" / "factors_alpha2.yaml"

# Minimum fraction of instruments that must have non-NaN values for a factor
# to be considered "working". 50% means at least 6 out of 12 instruments.
MIN_VALID_FRACTION = 0.5

# All 8 raw factors from factors_alpha2.yaml
RAW_FACTORS = {
    "alpha044": "correlation(high, rank(volume), 5)",
    "alpha033": "rank(1 - open / close)",
    "alpha017": "rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5))",
    "alpha026": "ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "alpha055": "correlation(rank((close - ts_min(low, 12)) / replace_zero(ts_max(high, 12) - ts_min(low, 12), 0.0001)), rank(volume), 6)",
    "alpha002": "correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)",
    "alpha039": "rank(delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9)))) * (1 + rank(ts_mean(returns, 250)))",
    "alpha003": "correlation(rank(open), rank(volume), 10)",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_catalog_data() -> dict[str, pd.DataFrame]:
    """Load all 12-coin parquet files and build OHLCV panel DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of field name → DataFrame[T x N] where T=timestamps, N=instruments.
    """
    if not CATALOG_DIR.exists():
        pytest.skip(f"Catalog directory not found: {CATALOG_DIR}")

    instrument_dirs = sorted(CATALOG_DIR.iterdir())
    if not instrument_dirs:
        pytest.skip("No instrument directories in catalog")

    # Load each instrument's parquet
    all_data: dict[str, pd.DataFrame] = {}
    for inst_dir in instrument_dirs:
        parquet_files = list(inst_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        df = pd.read_parquet(parquet_files[0])
        # Extract instrument name from directory (e.g., "BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL" → "BTCUSDT.BINANCE")
        inst_name = inst_dir.name.split("-4-HOUR")[0]
        all_data[inst_name] = df

    if not all_data:
        pytest.skip("No parquet data loaded from catalog")

    # Build panel DataFrames (T x N)
    # Standard OHLCV fields
    fields = ["open", "high", "low", "close", "volume"]
    # Check for extra fields in first instrument
    first_df = next(iter(all_data.values()))
    extra_fields = []
    for col in ["quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        if col in first_df.columns:
            extra_fields.append(col)

    all_fields = fields + extra_fields
    panel: dict[str, pd.DataFrame] = {}

    for field_name in all_fields:
        field_data = {}
        for inst_name, df in all_data.items():
            if field_name in df.columns:
                series = df[field_name].astype(float)
                series.index = range(len(series))  # Normalize index
                field_data[inst_name] = series
        if field_data:
            panel[field_name] = pd.DataFrame(field_data)

    return panel


@pytest.fixture(scope="module")
def catalog_panel() -> dict[str, pd.DataFrame]:
    """Module-scoped fixture for catalog panel data."""
    return _load_catalog_data()


@pytest.fixture(scope="module")
def catalog_panel_with_returns(catalog_panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Panel data with pre-computed 'returns' variable."""
    panel = dict(catalog_panel)
    close = panel["close"]
    panel["returns"] = close / close.shift(1) - 1
    return panel


def _create_engine_with_single_factor(
    name: str,
    expression: str,
    panel: dict[str, pd.DataFrame],
    with_returns: bool = False,
) -> dict[str, dict[str, float]]:
    """Create a PanelFactorEngine, feed panel data, and compute a single factor.

    Returns the last compute results: {factor_name: {instrument: value}}.
    """
    n_timestamps = len(next(iter(panel.values())))
    instruments = list(next(iter(panel.values())).columns)

    engine = PanelFactorEngine(max_history=600)

    # Register returns variable if needed
    if with_returns:
        engine.register_variable("returns", "delta(close, 1) / delay(close, 1)")

    engine.register_expression_factor(name, expression)

    # Feed data bar-by-bar
    results = {}
    for ts_idx in range(n_timestamps):
        for inst in instruments:
            bar_data = {}
            for field_name, df in panel.items():
                if field_name == "returns":
                    continue  # Variable, not bar data
                if inst in df.columns:
                    val = df[inst].iloc[ts_idx]
                    if not np.isnan(val):
                        bar_data[field_name] = float(val)
            if bar_data:
                engine.on_bar(inst, bar_data, ts_idx)

        results = engine.flush_and_compute(ts_idx)

    return results


def _create_full_engine(
    panel: dict[str, pd.DataFrame],
) -> tuple[PanelFactorEngine, dict[str, dict[str, float]]]:
    """Create a PanelFactorEngine loaded from config, feed all data.

    Returns (engine, last_results).
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config not found: {CONFIG_PATH}")

    config = load_factor_config(str(CONFIG_PATH))
    n_timestamps = len(next(iter(panel.values())))
    instruments = list(next(iter(panel.values())).columns)

    engine = PanelFactorEngine(config=config, max_history=600)

    results = {}
    for ts_idx in range(n_timestamps):
        for inst in instruments:
            bar_data = {}
            for field_name, df in panel.items():
                if field_name == "returns":
                    continue
                if inst in df.columns:
                    val = df[inst].iloc[ts_idx]
                    if not np.isnan(val):
                        bar_data[field_name] = float(val)
            if bar_data:
                engine.on_bar(inst, bar_data, ts_idx)

        results = engine.flush_and_compute(ts_idx)

    return engine, results


# ---------------------------------------------------------------------------
# Test 1: Individual raw factors
# ---------------------------------------------------------------------------


class TestIndividualRawFactors:
    """Each of the 8 raw factors should produce non-NaN values for >50% of instruments."""

    @pytest.mark.parametrize(
        "factor_name,expression",
        list(RAW_FACTORS.items()),
        ids=list(RAW_FACTORS.keys()),
    )
    def test_individual_raw_factor(
        self,
        factor_name: str,
        expression: str,
        catalog_panel: dict[str, pd.DataFrame],
    ) -> None:
        """Each factor independently produces sufficient non-NaN values."""
        # alpha039 uses 'returns' variable
        needs_returns = "returns" in expression
        results = _create_engine_with_single_factor(
            name=factor_name,
            expression=expression,
            panel=catalog_panel,
            with_returns=needs_returns,
        )

        n_instruments = len(next(iter(catalog_panel.values())).columns)

        assert factor_name in results, f"{factor_name} not in results"
        factor_result = results[factor_name]
        n_valid = len(factor_result)
        valid_fraction = n_valid / n_instruments

        print(
            f"\n  {factor_name}: {n_valid}/{n_instruments} instruments "
            f"({valid_fraction:.0%}) have valid values"
        )
        if factor_result:
            values = list(factor_result.values())
            print(
                f"    range: [{min(values):.6f}, {max(values):.6f}], "
                f"mean={np.mean(values):.6f}, std={np.std(values):.6f}"
            )

        assert valid_fraction >= MIN_VALID_FRACTION, (
            f"{factor_name} has only {n_valid}/{n_instruments} "
            f"({valid_fraction:.0%}) valid instruments, expected >= {MIN_VALID_FRACTION:.0%}"
        )


# ---------------------------------------------------------------------------
# Test 2: Normalized factors
# ---------------------------------------------------------------------------


NORM_FACTORS = [
    "alpha044_norm",
    "alpha033_norm",
    "alpha017_norm",
    "alpha026_norm",
    "alpha055_norm",
    "alpha002_norm",
    "alpha039_norm",
    "alpha003_norm",
]


class TestNormalizedFactors:
    """Each normalized factor (z-score) should produce valid values."""

    @pytest.mark.parametrize("factor_name", NORM_FACTORS, ids=NORM_FACTORS)
    def test_normalized_factor(
        self,
        factor_name: str,
        catalog_panel: dict[str, pd.DataFrame],
    ) -> None:
        """Verify normalize(alpha_xxx, true, 0) z-score works end-to-end."""
        # We need the full config to get the dependency chain:
        # raw factor → normalize(raw, true, 0)
        _, results = _create_full_engine(catalog_panel)

        n_instruments = len(next(iter(catalog_panel.values())).columns)

        assert factor_name in results, (
            f"{factor_name} not in results. "
            f"Available factors: {sorted(results.keys())}"
        )
        factor_result = results[factor_name]
        n_valid = len(factor_result)
        valid_fraction = n_valid / n_instruments

        print(
            f"\n  {factor_name}: {n_valid}/{n_instruments} instruments "
            f"({valid_fraction:.0%}) have valid values"
        )
        if factor_result:
            values = list(factor_result.values())
            print(
                f"    range: [{min(values):.6f}, {max(values):.6f}], "
                f"mean={np.mean(values):.6f}, std={np.std(values):.6f}"
            )

        assert valid_fraction >= MIN_VALID_FRACTION, (
            f"{factor_name} has only {n_valid}/{n_instruments} "
            f"({valid_fraction:.0%}) valid instruments"
        )


# ---------------------------------------------------------------------------
# Test 3: Complete composite factor
# ---------------------------------------------------------------------------


class TestCompositeFactor:
    """The final composite factor must produce valid values for all instruments."""

    def test_composite_non_empty(
        self,
        catalog_panel: dict[str, pd.DataFrame],
    ) -> None:
        """Composite factor should have values for all (or most) instruments."""
        _, results = _create_full_engine(catalog_panel)

        n_instruments = len(next(iter(catalog_panel.values())).columns)

        assert "composite" in results, (
            f"'composite' not in results. "
            f"Available: {sorted(results.keys())}"
        )
        composite = results["composite"]
        n_valid = len(composite)
        valid_fraction = n_valid / n_instruments

        print(f"\n  composite: {n_valid}/{n_instruments} instruments ({valid_fraction:.0%})")
        if composite:
            values = list(composite.values())
            print(
                f"    range: [{min(values):.6f}, {max(values):.6f}], "
                f"mean={np.mean(values):.6f}, std={np.std(values):.6f}"
            )

        # Composite must have ALL instruments valid (NaN propagation check)
        assert valid_fraction >= 0.9, (
            f"composite has only {n_valid}/{n_instruments} ({valid_fraction:.0%}) "
            f"valid instruments — likely NaN propagation from a sub-factor"
        )

    def test_composite_factor_values_detail(
        self,
        catalog_panel: dict[str, pd.DataFrame],
    ) -> None:
        """Print detailed per-factor diagnostics to identify NaN propagation."""
        _, results = _create_full_engine(catalog_panel)
        n_instruments = len(next(iter(catalog_panel.values())).columns)

        print("\n=== Per-factor diagnostic summary ===")
        for name in list(RAW_FACTORS.keys()) + NORM_FACTORS + ["composite"]:
            if name in results:
                n = len(results[name])
                status = "OK" if n >= n_instruments * MIN_VALID_FRACTION else "FAIL"
                print(f"  [{status}] {name}: {n}/{n_instruments} valid")
            else:
                print(f"  [MISSING] {name}: not in results")


# ---------------------------------------------------------------------------
# Test 4: Factor values publish content
# ---------------------------------------------------------------------------


class TestFactorValuesPublish:
    """Verify FactorValues.create() produces composite with values."""

    def test_factor_values_create_with_composite(
        self,
        catalog_panel: dict[str, pd.DataFrame],
    ) -> None:
        """FactorValues created from engine results should contain non-empty composite."""
        from nautilus_quants.factors.types import FactorValues

        _, results = _create_full_engine(catalog_panel)

        # Create FactorValues like the actor does
        factor_values = FactorValues.create(
            ts_event=12345678,
            factors=results,
        )

        assert factor_values.factors is not None
        assert "composite" in factor_values.factors, (
            f"'composite' missing from FactorValues.factors. "
            f"Keys: {sorted(factor_values.factors.keys())}"
        )
        composite = factor_values.factors["composite"]
        assert len(composite) > 0, (
            "FactorValues.factors['composite'] is empty — "
            "strategy on_data() will never accumulate values"
        )
        print(f"\n  FactorValues composite: {len(composite)} instruments with values")
