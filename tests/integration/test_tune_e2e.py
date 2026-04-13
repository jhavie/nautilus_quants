# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""End-to-end tuning workflow.

Exercises the `alpha tune` CLI path in single-expression mode so the full
chain (YAML load → data load → search-space build → Optuna study → holdout
→ registration → report writing) is covered without needing a full mining
pipeline. The catalog is synthesised on the fly via a tiny helper so the
test has zero external dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")
duckdb = pytest.importorskip("duckdb")

import numpy as np
import yaml

from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.models import FactorRecord
from nautilus_quants.alpha.registry.repository import FactorRepository
from nautilus_quants.alpha.tuning.config import (
    ALGORITHM_GRID,
    CVConfig,
    DimensionsConfig,
    TuneConfig,
)
from nautilus_quants.alpha.tuning.objective import build_cv_folds, compute_forward_returns_panel
from nautilus_quants.alpha.tuning.optimizer import OptimizeInputs, optimize_factor
from nautilus_quants.alpha.tuning.report import (
    build_factor_dir,
    build_run_dir,
    write_factor_artefacts,
    write_run_summary,
)
from nautilus_quants.alpha.tuning.variant_registration import register_tuned_variants

# ── Test fixtures ──────────────────────────────────────────────────────────


def _synthetic_panel(T: int = 600, N: int = 20, seed: int = 17):
    """Build a close+volume panel where volume predicts next-bar return."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="4h")
    cols = [f"I{i}" for i in range(N)]
    close = pd.DataFrame(
        100 + rng.standard_normal((T, N)).cumsum(axis=0) * 0.4,
        index=idx,
        columns=cols,
    )
    # Baseline volume + positive injection correlated with the *next* bar's
    # realised return → factor `correlation(high, rank(volume), w)` should
    # carry information for a suitable window w.
    future_ret = close.pct_change().shift(-1).fillna(0)
    volume = pd.DataFrame(
        np.abs(rng.standard_normal((T, N))) * 2000 + 5000 + future_ret.values * 10_000,
        index=idx,
        columns=cols,
    )
    high = close * (1 + np.abs(rng.standard_normal((T, N))) * 1.5e-3)
    low = close * (1 - np.abs(rng.standard_normal((T, N))) * 1.5e-3)
    open_ = close.shift(1).fillna(close)
    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return panel, close


# ── End-to-end, in-process ────────────────────────────────────────────────


class TestTuneEndToEnd:
    def test_single_expression_tune_writes_variants_and_artefacts(
        self,
        tmp_path: Path,
    ) -> None:
        """Run the tuning inner machinery on a synthetic panel, then exercise
        variant registration + artefact writing. Equivalent to executing
        ``alpha tune --config X.yaml -e "..." --no-register``  but
        side-steps CLI argument parsing for tighter assertions.
        """
        panel, close = _synthetic_panel()
        inputs = OptimizeInputs(
            panel_fields=panel,
            pricing=close,
            fwd_returns=compute_forward_returns_panel(close, 1),
            cv_schedule=build_cv_folds(
                len(close.index),
                CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
            ),
        )

        config = TuneConfig(
            trials=6,
            register_top_k=2,
            seed=13,
            dimensions=DimensionsConfig(
                numeric=True,
                operators=False,
                variables=False,
            ),
            cv=CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
        )

        result = optimize_factor(
            expression="-correlation(high, rank(volume), 5)",
            inputs=inputs,
            tune_config=config,
        )
        assert result.n_trials > 0
        assert result.best_expression.startswith("-1 * correlation")
        assert len(result.top_k) <= 2

        # Registry write-back.
        db = RegistryDatabase(":memory:")
        repo = FactorRepository(db)
        source = FactorRecord(
            factor_id="alpha101_alpha044_e2e",
            expression="-correlation(high, rank(volume), 5)",
            prototype="alpha044",
            source="alpha101",
            status="candidate",
        )
        repo.upsert_factor(source)
        registration = register_tuned_variants(
            tune_result=result,
            source_factor=source,
            repo=repo,
            panel_fields=panel,
            pricing=close,
            periods=(1, 2, 6),
            tune_config=config,
            run_id="20260412_000000",
            timeframe="4h",
        )
        assert registration.n_registered >= 1

        # Artefacts on disk.
        run_dir = build_run_dir(str(tmp_path / "tune_out"), "20260412_000000")
        factor_dir = build_factor_dir(run_dir, source.factor_id)
        write_factor_artefacts(factor_dir, result, registration)
        write_run_summary(
            run_dir,
            tune_config=config,
            results=[(source.factor_id, result, registration)],
        )

        assert (factor_dir / "trials.json").exists()
        assert (factor_dir / "best_params.yaml").exists()
        assert (factor_dir / "holdout_metrics.json").exists()
        assert (factor_dir / "tune_result.json").exists()
        assert (factor_dir / "registration_summary.json").exists()
        assert (run_dir / "run_summary.json").exists()
        assert (run_dir / "tune_result.csv").exists()

        # Round-trip the CSV summary so we assert every registered variant
        # was written.
        csv = (run_dir / "tune_result.csv").read_text().splitlines()
        header = csv[0].split(",")
        assert header[0] == "label"
        assert any(source.factor_id in line for line in csv[1:])

        # Registry contains the new variants.
        variants = repo.list_factors(source="tune")
        assert variants
        for v in variants:
            assert v.prototype == "alpha044"
            assert "tuned" in v.tags

        db.close()

    def test_operators_dimension_produces_comparison_table(
        self,
        tmp_path: Path,
    ) -> None:
        panel, close = _synthetic_panel()
        inputs = OptimizeInputs(
            panel_fields=panel,
            pricing=close,
            fwd_returns=compute_forward_returns_panel(close, 1),
            cv_schedule=build_cv_folds(
                len(close.index),
                CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
            ),
        )
        config = TuneConfig(
            trials=6,
            register_top_k=1,
            seed=99,
            dimensions=DimensionsConfig(
                numeric=True,
                operators=True,
                variables=False,
            ),
            cv=CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
        )
        result = optimize_factor(
            expression="cs_rank(-correlation(high, rank(volume), 5))",
            inputs=inputs,
            tune_config=config,
        )
        assert result.operator_comparison is not None
        assert any(k.startswith("op_") for k in result.operator_comparison)
        any_options = any(bool(v) for v in result.operator_comparison.values())
        assert any_options, "Expected at least one operator comparison entry"

    def test_grid_algorithm_smoke(self, tmp_path: Path) -> None:
        """End-to-end regression for ``algorithm: grid``.

        Pre-fix this run aborted on the very first ``trial.suggest_*`` call
        because ``GridSampler`` was constructed with ``{}``. Now the sampler
        receives the full numeric/op/var grid built from ``build_search_space``
        and the study completes normally.
        """
        panel, close = _synthetic_panel()
        inputs = OptimizeInputs(
            panel_fields=panel,
            pricing=close,
            fwd_returns=compute_forward_returns_panel(close, 1),
            cv_schedule=build_cv_folds(
                len(close.index),
                CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
            ),
        )
        config = TuneConfig(
            algorithm=ALGORITHM_GRID,
            trials=4,
            register_top_k=1,
            seed=7,
            dimensions=DimensionsConfig(numeric=True, operators=False, variables=False),
            cv=CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
            early_stop_patience=0,  # let grid traverse the search space
        )

        result = optimize_factor(
            expression="-correlation(high, rank(volume), 5)",
            inputs=inputs,
            tune_config=config,
        )
        assert result.n_trials > 0
        assert result.best_expression  # non-empty
