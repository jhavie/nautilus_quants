# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Real-data engine consistency test.

Loads BinanceBar parquet data from data_4years_1h catalog,
feeds IDENTICAL bars to both FactorEngine+CsFactorEngine and PanelFactorEngine,
and compares composite factor values at every timestamp.

Period: 2022-01-01 ~ 2022-03-01 (2 months)
Data source: /Users/joe/Sync/nautilus_quants2/data/data_4years_1h/catalog
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.cs_factor_engine import CsFactorEngine
from nautilus_quants.factors.engine.factor_engine import FactorEngine
from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine

# ---------------------------------------------------------------------------
CATALOG_PATH = "/Users/joe/Sync/nautilus_quants2/data/data_4years_1h/catalog"
FACTOR_CONFIG_PATH = "config/fmz/factors.yaml"
START_NS = 1640995200_000_000_000  # 2022-01-01 00:00 UTC
END_NS = 1646092800_000_000_000    # 2022-03-01 00:00 UTC
EXTRA_FIELDS = ["quote_volume"]
# ---------------------------------------------------------------------------


class MockBar:
    """Minimal mock bar for FactorEngine.on_bar()."""

    def __init__(self, instrument_id: str, open_: float, high: float,
                 low: float, close: float, volume: float,
                 quote_volume: float, ts_event: int):
        self.bar_type = type("BarType", (), {"instrument_id": instrument_id})()
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.quote_volume = quote_volume
        self.ts_event = ts_event


def _load_bars() -> pd.DataFrame:
    """Load all BinanceBar data for 2022-01 ~ 2022-03 from parquet catalog."""
    binance_dir = Path(CATALOG_PATH) / "data" / "binance_bar"
    if not binance_dir.exists():
        pytest.skip(f"Catalog not found: {binance_dir}")

    frames = []
    for inst_dir in sorted(binance_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        instrument_id = inst_dir.name.replace("-1-HOUR-LAST-EXTERNAL", "")
        for pq_file in inst_dir.glob("*.parquet"):
            df = pq.read_table(pq_file).to_pandas()
            # Filter time range
            df = df[(df["ts_event"] >= START_NS) & (df["ts_event"] < END_NS)]
            if df.empty:
                continue
            df["instrument_id_clean"] = instrument_id
            frames.append(df[["instrument_id_clean", "ts_event",
                               "open", "high", "low", "close",
                               "volume", "quote_volume"]])

    if not frames:
        pytest.skip("No data found in catalog")

    all_bars = pd.concat(frames, ignore_index=True)
    # Sort by ts_event, then instrument (Nautilus replay order)
    all_bars = all_bars.sort_values(["ts_event", "instrument_id_clean"]).reset_index(drop=True)
    # Convert numeric columns
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        all_bars[col] = all_bars[col].astype(float)
    all_bars["ts_event"] = all_bars["ts_event"].astype(int)

    return all_bars


class TestEngineRealData:
    """Compare engines using real BinanceBar data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = load_factor_config(FACTOR_CONFIG_PATH)
        self.all_bars = _load_bars()
        n_instruments = self.all_bars["instrument_id_clean"].nunique()
        n_bars = len(self.all_bars)
        print(f"\nLoaded {n_bars} bars for {n_instruments} instruments "
              f"({self.all_bars['ts_event'].min()} ~ {self.all_bars['ts_event'].max()})")

    def test_composite_consistency_real_data(self):
        """Compare composite factor at every timestamp after warmup."""
        config = self.config

        # --- Set up engines ---
        inc_engine = FactorEngine(config=config, max_history=500)
        inc_engine.set_extra_fields(EXTRA_FIELDS)
        cs_engine = CsFactorEngine(config=config)

        panel_engine = PanelFactorEngine(config=config, max_history=500)
        panel_engine.set_extra_fields(EXTRA_FIELDS)

        # --- Feed data ---
        all_bars = self.all_bars
        timestamps = sorted(all_bars["ts_event"].unique())
        warmup_ts = 100  # skip first 100 timestamps

        mismatches: list[str] = []
        value_diffs: list[float] = []
        total_compared = 0
        ts_compared = 0

        for t_idx, ts in enumerate(timestamps):
            ts_bars = all_bars[all_bars["ts_event"] == ts]

            # --- Incremental engine ---
            inc_ts_values: dict[str, dict[str, float]] = {}
            for _, row in ts_bars.iterrows():
                mock_bar = MockBar(
                    instrument_id=row["instrument_id_clean"],
                    open_=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    quote_volume=row["quote_volume"],
                    ts_event=int(ts),
                )
                ts_result = inc_engine.on_bar(mock_bar)
                for fname, inst_vals in ts_result.items():
                    if fname not in inc_ts_values:
                        inc_ts_values[fname] = {}
                    inc_ts_values[fname].update(inst_vals)

            inc_cs_values = cs_engine.compute(inc_ts_values)
            inc_full = dict(inc_ts_values)
            inc_full.update(inc_cs_values)

            # --- Panel engine ---
            for _, row in ts_bars.iterrows():
                bar_dict = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "quote_volume": row["quote_volume"],
                }
                panel_engine.on_bar(row["instrument_id_clean"], bar_dict, int(ts))

            panel_results = panel_engine.flush_and_compute(int(ts))

            # Skip warmup
            if t_idx < warmup_ts:
                continue

            ts_compared += 1

            # Compare composite and composite_neg
            for fname in ("composite", "composite_neg"):
                inc_vals = inc_full.get(fname, {})
                pan_vals = panel_results.get(fname, {})

                common = sorted(set(inc_vals.keys()) & set(pan_vals.keys()))
                only_inc = set(inc_vals.keys()) - set(pan_vals.keys())
                only_pan = set(pan_vals.keys()) - set(inc_vals.keys())

                for inst in common:
                    iv, pv = inc_vals[inst], pan_vals[inst]
                    if math.isnan(iv) and math.isnan(pv):
                        continue
                    total_compared += 1
                    diff = abs(iv - pv)
                    value_diffs.append(diff)

                    if not np.isclose(iv, pv, rtol=1e-4, atol=1e-6):
                        if len(mismatches) < 50:
                            mismatches.append(
                                f"ts_idx={t_idx}, {fname}, {inst}: "
                                f"inc={iv:.8f}, panel={pv:.8f}, diff={diff:.2e}"
                            )

                if only_inc or only_pan:
                    if len(mismatches) < 50:
                        mismatches.append(
                            f"ts_idx={t_idx}, {fname}: inst_mismatch — "
                            f"inc_only={len(only_inc)}, pan_only={len(only_pan)}"
                        )

        # --- Report ---
        print(f"\nTimestamps compared: {ts_compared}")
        print(f"Total value comparisons: {total_compared}")

        if value_diffs:
            diffs_arr = np.array(value_diffs)
            print(f"Max diff:  {diffs_arr.max():.2e}")
            print(f"Mean diff: {diffs_arr.mean():.2e}")
            print(f"P99 diff:  {np.percentile(diffs_arr, 99):.2e}")
            print(f"P50 diff:  {np.percentile(diffs_arr, 50):.2e}")

        if mismatches:
            detail = "\n".join(mismatches[:50])
            n = len(mismatches)
            pytest.fail(
                f"Engine consistency failures ({n} issues, "
                f"showing first {min(n, 50)}):\n{detail}"
            )
        else:
            print("\nAll values match between engines!")

    def test_ts_factors_real_data(self):
        """Compare individual TS factors (momentum_3h, volatility, corr)."""
        config = self.config

        inc_engine = FactorEngine(config=config, max_history=500)
        inc_engine.set_extra_fields(EXTRA_FIELDS)
        panel_engine = PanelFactorEngine(config=config, max_history=500)
        panel_engine.set_extra_fields(EXTRA_FIELDS)

        all_bars = self.all_bars
        timestamps = sorted(all_bars["ts_event"].unique())
        warmup_ts = 100

        ts_factors_to_check = ["momentum_3h", "volatility", "corr", "quote_volume"]
        factor_diffs: dict[str, list[float]] = {f: [] for f in ts_factors_to_check}
        factor_mismatches: dict[str, int] = {f: 0 for f in ts_factors_to_check}

        for t_idx, ts in enumerate(timestamps):
            ts_bars = all_bars[all_bars["ts_event"] == ts]

            # Incremental
            inc_ts_values: dict[str, dict[str, float]] = {}
            for _, row in ts_bars.iterrows():
                mock_bar = MockBar(
                    instrument_id=row["instrument_id_clean"],
                    open_=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    quote_volume=row["quote_volume"],
                    ts_event=int(ts),
                )
                ts_result = inc_engine.on_bar(mock_bar)
                for fname, inst_vals in ts_result.items():
                    if fname not in inc_ts_values:
                        inc_ts_values[fname] = {}
                    inc_ts_values[fname].update(inst_vals)

            # Panel
            for _, row in ts_bars.iterrows():
                bar_dict = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "quote_volume": row["quote_volume"],
                }
                panel_engine.on_bar(row["instrument_id_clean"], bar_dict, int(ts))

            panel_results = panel_engine.flush_and_compute(int(ts))

            if t_idx < warmup_ts:
                continue

            for fname in ts_factors_to_check:
                inc_vals = inc_ts_values.get(fname, {})
                pan_vals = panel_results.get(fname, {})
                common = set(inc_vals.keys()) & set(pan_vals.keys())

                for inst in common:
                    iv, pv = inc_vals[inst], pan_vals[inst]
                    if math.isnan(iv) and math.isnan(pv):
                        continue
                    if math.isnan(iv) or math.isnan(pv):
                        factor_mismatches[fname] += 1
                        continue
                    diff = abs(iv - pv)
                    factor_diffs[fname].append(diff)
                    if not np.isclose(iv, pv, rtol=1e-4, atol=1e-6):
                        factor_mismatches[fname] += 1

        # Report per-factor statistics
        print("\n=== TS Factor Consistency Report (real data) ===")
        all_ok = True
        for fname in ts_factors_to_check:
            diffs = factor_diffs[fname]
            n_mismatch = factor_mismatches[fname]
            if diffs:
                arr = np.array(diffs)
                print(f"{fname:20s}: n={len(diffs):6d}, max_diff={arr.max():.2e}, "
                      f"mean_diff={arr.mean():.2e}, mismatches={n_mismatch}")
            else:
                print(f"{fname:20s}: no comparisons")
            if n_mismatch > 0:
                all_ok = False

        if not all_ok:
            pytest.fail(f"TS factor mismatches detected: {factor_mismatches}")
