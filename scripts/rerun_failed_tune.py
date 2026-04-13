# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Re-tune the SanAPI factors that produced 0 completed trials in a prior run.

A previous tune session bug skipped derived-variable injection, causing every
trial of factors that referenced ``returns`` / ``vwap`` / ``btc_beta`` / ...
to be pruned. This script reads a list of failed ``factor_id``s, loads the
shared panel data ONCE, and re-runs the tune pipeline (with the now-fixed
variable injection) on each factor — saving the ~30s startup cost that
``alpha tune --factor-id ...`` would pay per invocation.

Usage:
    python scripts/rerun_failed_tune.py /tmp/failed_factors.txt

The factor IDs file is one ``factor_id`` per line. Output mirrors what
``alpha tune`` writes — same registry tables, same per-factor artefact dir
under ``logs/alpha_tune_sanapi/<run_id>/``.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nautilus_quants.alpha.analysis.config import load_analysis_config
from nautilus_quants.alpha.data_loader import CatalogDataLoader
from nautilus_quants.alpha.formatters import (
    print_operator_comparison,
    print_tune_result,
    print_variable_comparison,
)
from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning.config_loader import load_tune_config
from nautilus_quants.alpha.tuning.objective import (
    build_cv_folds,
    compute_forward_returns_panel,
    evaluate_expression_panel,
)
from nautilus_quants.alpha.tuning.optimizer import (
    OptimizeInputs,
    optimize_factor,
)
from nautilus_quants.alpha.tuning.report import (
    build_factor_dir,
    build_run_dir,
    write_factor_artefacts,
    write_run_summary,
)
from nautilus_quants.alpha.tuning.variant_registration import (
    register_tuned_variants,
)
from nautilus_quants.backtest.utils.reporting import generate_run_id
from nautilus_quants.factors.config import FactorConfig
from nautilus_quants.factors.engine.extra_data import ExtraDataManager

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


CONFIG_PATH = "config/cs/alpha_mining_sanapi.yaml"


def _enrich_panel(base_panel, variables, parameters):
    """Mirror cli._enrich_panel_with_factor_variables — keep this in sync."""
    if not variables:
        return base_panel
    enriched = dict(base_panel)
    for name, value in (parameters or {}).items():
        if name not in enriched:
            enriched[name] = value
    for var_name, var_expr in variables.items():
        try:
            value = evaluate_expression_panel(var_expr, enriched, parameters)
            enriched[var_name] = value
        except Exception as exc:
            print(f"  ⚠️  Variable '{var_name}' failed: {exc}")
    return enriched


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <factor_ids_file>")
        sys.exit(1)
    factor_ids_file = Path(sys.argv[1])
    if not factor_ids_file.exists():
        print(f"File not found: {factor_ids_file}")
        sys.exit(1)
    factor_ids = [
        line.strip() for line in factor_ids_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Loaded {len(factor_ids)} factor IDs to rerun.")

    # ── Load configs + data once ──
    print("\nLoading configs and panel data...")
    t0 = time.time()
    tune_cfg = load_tune_config(CONFIG_PATH)
    analysis_cfg = load_analysis_config(CONFIG_PATH)

    loader = CatalogDataLoader(analysis_cfg.catalog_path, analysis_cfg.bar_spec)
    bars = loader.load_bars(
        analysis_cfg.instrument_ids,
        start=analysis_cfg.start_date,
        end=analysis_cfg.end_date,
    )
    ohlcv = {i: CatalogDataLoader.bars_to_dataframe(b) for i, b in bars.items() if b}
    pricing = pd.DataFrame({i: df["close"] for i, df in ohlcv.items()})
    base_panel = {
        f: pd.concat({i: df[f] for i, df in ohlcv.items()}, axis=1)
        for f in ("open", "high", "low", "close", "volume")
    }
    if analysis_cfg.extra_data:
        ExtraDataManager(analysis_cfg.extra_data).inject_panels(
            base_panel, list(ohlcv.keys()),
            bars_by_instrument=bars, catalog_path=analysis_cfg.catalog_path,
        )

    fwd_returns = compute_forward_returns_panel(pricing, period_bars=1)
    cv_schedule = build_cv_folds(len(pricing.index), tune_cfg.cv)
    base_inputs = OptimizeInputs(
        panel_fields=base_panel,
        pricing=pricing,
        fwd_returns=fwd_returns,
        cv_schedule=cv_schedule,
    )
    factor_config = FactorConfig()  # mining yaml has no factor_config_path
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Panel: {pricing.shape}  ({len(base_panel)} fields)")
    print(f"Periods: {analysis_cfg.periods}  CV folds: {len(cv_schedule.folds)}")

    # ── Setup output dir + registry ──
    run_id = generate_run_id()
    run_dir = build_run_dir(tune_cfg.output_dir, run_id)
    print(f"\nRerun run_id: {run_id}\nOutput: {run_dir}\n")

    db = RegistryDatabase.for_environment(
        tune_cfg.candidates.env, "/Users/joe/Sync/nautilus_quants/logs/registry"
    )
    repo = FactorRepository(db)

    # ── Loop over failed factors ──
    all_results = []
    n_success = n_failed = n_total_variants = 0
    loop_start = time.time()

    for idx, factor_id in enumerate(factor_ids, start=1):
        factor_t0 = time.time()
        source = repo.get_factor(factor_id)
        if source is None:
            print(f"[{idx}/{len(factor_ids)}] ⊘ {factor_id}: not found in registry")
            n_failed += 1
            continue

        print(
            f"[{idx}/{len(factor_ids)}] tuning {factor_id} "
            f"(vars: {list(source.variables.keys())[:3]}{'...' if len(source.variables) > 3 else ''})",
            flush=True,
        )

        try:
            f_panel = _enrich_panel(base_panel, source.variables, source.parameters)
            f_inputs = replace(base_inputs, panel_fields=f_panel)
            result = optimize_factor(
                expression=source.expression,
                inputs=f_inputs,
                tune_config=tune_cfg,
                parameters=dict(source.parameters),
                available_vars=set(f_panel.keys()),
                derived_vars=set(factor_config.variables.keys()),
            )

            registration = register_tuned_variants(
                tune_result=result,
                source_factor=source,
                repo=repo,
                panel_fields=f_panel,
                pricing=pricing,
                periods=tuple(analysis_cfg.periods),
                tune_config=tune_cfg,
                run_id=run_id,
                timeframe=analysis_cfg.bar_spec,
                quantiles=analysis_cfg.quantiles,
                filter_zscore=analysis_cfg.filter_zscore,
                max_loss=analysis_cfg.max_loss,
                output_dir=str(run_dir),
            )

            factor_dir = build_factor_dir(run_dir, factor_id)
            write_factor_artefacts(factor_dir, result, registration)

            elapsed = time.time() - factor_t0
            n_variants = registration.n_registered if registration else 0
            n_total_variants += n_variants
            n_success += 1
            print(
                f"    ✓ {elapsed:.1f}s  "
                f"best ICIR={result.best_icir_cv:.4f}  "
                f"trials={result.n_trials}  "
                f"variants={n_variants}"
            )
            all_results.append((factor_id, result, registration))
        except Exception as exc:
            print(f"    ✗ FAILED: {exc!r}")
            n_failed += 1

    write_run_summary(run_dir, tune_config=tune_cfg, results=all_results)

    total_elapsed = time.time() - loop_start
    print(
        f"\n=== Done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) ===\n"
        f"Success:        {n_success} / {len(factor_ids)}\n"
        f"Failed:         {n_failed}\n"
        f"New variants:   {n_total_variants}\n"
        f"Artefacts:      {run_dir}"
    )

    db.close()


if __name__ == "__main__":
    main()
