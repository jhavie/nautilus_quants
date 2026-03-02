#!/usr/bin/env python3
# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Profiling script for PanelFactorEngine performance.

Runs a full e2e data feed through the engine with profiling enabled,
generating per-phase timing breakdowns and cProfile reports.

Usage:
    python -m tests.profiling.profile_panel_engine
    python -m tests.profiling.profile_panel_engine --cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nautilus_quants.factors.config import load_factor_config
from nautilus_quants.factors.engine.panel_factor_engine import PanelFactorEngine

CATALOG_DIR = Path("/Users/joe/Sync/nautilus_quants2/data/12coin_catalog/data/binance_bar")
CONFIG_PATH = PROJECT_ROOT / "config" / "fmz" / "factors_alpha2.yaml"
LOG_DIR = PROJECT_ROOT / "logs" / "performance"


def load_catalog_panel() -> dict[str, pd.DataFrame]:
    """Load 12-coin catalog data into panel DataFrames."""
    if not CATALOG_DIR.exists():
        raise FileNotFoundError(f"Catalog directory not found: {CATALOG_DIR}")

    all_data: dict[str, pd.DataFrame] = {}
    for inst_dir in sorted(CATALOG_DIR.iterdir()):
        parquet_files = list(inst_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        df = pd.read_parquet(parquet_files[0])
        inst_name = inst_dir.name.split("-4-HOUR")[0]
        all_data[inst_name] = df

    fields = ["open", "high", "low", "close", "volume"]
    first_df = next(iter(all_data.values()))
    for col in ["quote_volume", "count"]:
        if col in first_df.columns:
            fields.append(col)

    panel: dict[str, pd.DataFrame] = {}
    for field_name in fields:
        field_data = {}
        for inst_name, df in all_data.items():
            if field_name in df.columns:
                series = df[field_name].astype(float)
                series.index = range(len(series))
                field_data[inst_name] = series
        if field_data:
            panel[field_name] = pd.DataFrame(field_data)

    return panel


def run_engine(panel: dict[str, pd.DataFrame], enable_profiling: bool = True) -> PanelFactorEngine:
    """Feed all panel data through the engine."""
    config = load_factor_config(str(CONFIG_PATH))
    n_ts = len(next(iter(panel.values())))
    instruments = list(next(iter(panel.values())).columns)

    engine = PanelFactorEngine(config=config, max_history=200)
    engine.enable_profiling = enable_profiling

    for ts_idx in range(n_ts):
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
        engine.flush_and_compute(ts_idx)

    return engine


def generate_report(engine: PanelFactorEngine, wall_time: float, cprofile_output: str | None = None) -> str:
    """Generate markdown performance report."""
    perf = engine.get_performance_stats()
    prof = engine.get_profiling_stats()

    lines = [
        "# Panel Factor Engine — Performance Baseline",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Config**: factors_alpha2.yaml (17 factors, 12 instruments)",
        f"**max_history**: 200",
        "",
        "## Wall Clock",
        "",
        f"- **Total wall time**: {wall_time:.2f}s",
        f"- **Total flush_and_compute calls**: {perf['total_computes']}",
        f"- **Mean per call**: {perf['mean_ms']:.2f}ms",
        f"- **Max per call**: {perf['max_ms']:.2f}ms",
        f"- **P95 per call**: {perf.get('p95_ms', 0):.2f}ms",
        "",
        "## Per-Phase Breakdown",
        "",
        "| Phase | Mean (ms) | Max (ms) | P95 (ms) | % of Total |",
        "|-------|-----------|----------|----------|------------|",
    ]

    for phase in ["flush", "to_panel", "variables", "factors", "total"]:
        if phase in prof:
            s = prof[phase]
            lines.append(
                f"| {phase} | {s['mean_ms']:.3f} | {s['max_ms']:.3f} | "
                f"{s['p95_ms']:.3f} | {s['pct']:.1f}% |"
            )

    if cprofile_output:
        lines.extend([
            "",
            "## cProfile Top-50",
            "",
            "```",
            cprofile_output,
            "```",
        ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PanelFactorEngine")
    parser.add_argument("--cprofile", action="store_true", help="Run with cProfile")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    print("Loading 12-coin catalog data...")
    panel = load_catalog_panel()
    n_ts = len(next(iter(panel.values())))
    n_inst = len(next(iter(panel.values())).columns)
    print(f"  Loaded: {n_inst} instruments × {n_ts} timestamps")

    cprofile_output = None

    if args.cprofile:
        print("Running with cProfile...")
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        engine = run_engine(panel, enable_profiling=True)
        profiler.disable()
        wall_time = time.perf_counter() - t0

        stream = StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(50)
        cprofile_output = stream.getvalue()
    else:
        print("Running with segment timing...")
        t0 = time.perf_counter()
        engine = run_engine(panel, enable_profiling=True)
        wall_time = time.perf_counter() - t0

    print(f"\nCompleted in {wall_time:.2f}s")

    # Print summary
    perf = engine.get_performance_stats()
    prof = engine.get_profiling_stats()
    print(f"  Mean per flush_and_compute: {perf['mean_ms']:.2f}ms")
    print(f"  Max per flush_and_compute: {perf['max_ms']:.2f}ms")
    print(f"  P95 per flush_and_compute: {perf.get('p95_ms', 0):.2f}ms")
    print("\n  Phase breakdown:")
    for phase in ["flush", "to_panel", "variables", "factors"]:
        if phase in prof:
            s = prof[phase]
            print(f"    {phase:12s}: mean={s['mean_ms']:.3f}ms  ({s['pct']:.1f}%)")

    # Save report
    report = generate_report(engine, wall_time, cprofile_output)
    output_path = args.output or str(LOG_DIR / "panel-opt-0-baseline.md")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
