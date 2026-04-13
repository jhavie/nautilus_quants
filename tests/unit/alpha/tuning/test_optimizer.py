# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Tests for optimizer.py — small synthetic optimizations + helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from nautilus_quants.alpha.tuning.config import CVConfig, DimensionsConfig, ParamSpec, TuneConfig
from nautilus_quants.alpha.tuning.optimizer import (
    _apply_correction,
    _newey_west_t_pvalue,
    _unique_top_k,
    optimize_factor,
    prepare_inputs,
)


def _make_panel(
    seed: int = 42,
    T: int = 400,
    N: int = 15,
    signal_strength: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="4h")
    cols = [f"I{i}" for i in range(N)]
    close = pd.DataFrame(
        100 + rng.standard_normal((T, N)).cumsum(axis=0) * 0.3,
        index=idx,
        columns=cols,
    )
    # Inject positive correlation: volume proxies future returns
    future_ret = close.pct_change().shift(-1).fillna(0)
    volume = pd.DataFrame(
        np.abs(rng.standard_normal((T, N))) * 1000 + 5000 + future_ret.values * signal_strength * 5000,
        index=idx,
        columns=cols,
    )
    high = close * (1 + np.abs(rng.standard_normal((T, N))) * 1e-3)
    low = close * (1 - np.abs(rng.standard_normal((T, N))) * 1e-3)
    open_ = close.shift(1).fillna(close)
    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return panel, close


# ── Correction helpers ─────────────────────────────────────────────────────


class TestApplyCorrection:
    def test_bonferroni_scales_by_n(self) -> None:
        assert _apply_correction(0.01, 10, "bonferroni") == pytest.approx(0.1)

    def test_bonferroni_caps_at_one(self) -> None:
        assert _apply_correction(0.2, 100, "bonferroni") == 1.0

    def test_none_pass_through(self) -> None:
        assert _apply_correction(0.01, 10, "none") == 0.01

    def test_single_test_no_change(self) -> None:
        assert _apply_correction(0.05, 1, "bonferroni") == 0.05

    def test_nan_preserved(self) -> None:
        assert np.isnan(_apply_correction(float("nan"), 5, "bonferroni"))


class TestNeweyWestTPvalue:
    def test_zero_std_returns_zero_t(self) -> None:
        s = pd.Series([0.1] * 20)
        t, p, n_eff = _newey_west_t_pvalue(s)
        assert t == 0.0
        assert p == 1.0

    def test_large_positive_ic_gives_significant_t(self) -> None:
        s = pd.Series([0.5, 0.4, 0.6, 0.45, 0.55] * 10)
        t, p, n_eff = _newey_west_t_pvalue(s)
        assert t > 5.0
        assert p < 0.01

    def test_short_series_returns_nan(self) -> None:
        s = pd.Series([0.1, 0.2])
        t, p, n_eff = _newey_west_t_pvalue(s)
        assert np.isnan(t) and np.isnan(p)


class _FakeStudy:
    """Minimal stand-in for ``optuna.Study`` used by early-stop tests."""

    study_name = "test"

    def __init__(self) -> None:
        self.trials: list = []
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _make_trial(number: int, params: dict, state=None):
    state = state or optuna.trial.TrialState.COMPLETE
    return optuna.trial.FrozenTrial(
        number=number,
        state=state,
        value=0.1,
        datetime_start=None,
        datetime_complete=None,
        params=params,
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


class TestConvergenceEarlyStop:
    def test_does_not_stop_before_min_trials(self) -> None:
        from nautilus_quants.alpha.tuning.optimizer import _ConvergenceEarlyStop

        stopper = _ConvergenceEarlyStop(patience=3, min_trials=100)
        study = _FakeStudy()
        for i in range(5):
            t = _make_trial(i, {"p": 5.0})
            study.trials.append(t)
            stopper(study, t)
        assert not study.stopped  # min_trials gate blocks triggering

    def test_triggers_after_patience_identical_trials(self) -> None:
        from nautilus_quants.alpha.tuning.optimizer import _ConvergenceEarlyStop

        stopper = _ConvergenceEarlyStop(patience=3, min_trials=1)
        study = _FakeStudy()
        for i in range(4):
            t = _make_trial(i, {"p": 5.0, "op": "cs_rank"})
            study.trials.append(t)
            stopper(study, t)
        assert study.stopped
        assert stopper.triggered

    def test_resets_streak_when_params_change(self) -> None:
        from nautilus_quants.alpha.tuning.optimizer import _ConvergenceEarlyStop

        stopper = _ConvergenceEarlyStop(patience=3, min_trials=1)
        study = _FakeStudy()
        # Two identical, then different, then two identical — streak never
        # reaches patience=3.
        for i, p in enumerate([{"p": 5}, {"p": 5}, {"p": 10}, {"p": 5}, {"p": 5}]):
            t = _make_trial(i, p)
            study.trials.append(t)
            stopper(study, t)
        assert not study.stopped

    def test_ignores_pruned_trials(self) -> None:
        from nautilus_quants.alpha.tuning.optimizer import _ConvergenceEarlyStop

        stopper = _ConvergenceEarlyStop(patience=3, min_trials=1)
        study = _FakeStudy()
        for i in range(4):
            t = _make_trial(i, {"p": 5.0}, state=optuna.trial.TrialState.PRUNED)
            study.trials.append(t)
            stopper(study, t)
        assert not study.stopped


class TestUniqueTopK:
    def test_dedups_by_expression(self) -> None:
        from nautilus_quants.alpha.tuning.config import TrialResult

        trials = [
            TrialResult(
                trial_number=i,
                params={},
                expression=f"expr_{i % 3}",
                cv_icir=(),
                mean_icir=1.0 - i * 0.1,
                objective_value=1.0 - i * 0.1,
            )
            for i in range(6)
        ]
        kept = _unique_top_k(trials, k=4)
        assert [t.expression for t in kept] == ["expr_0", "expr_1", "expr_2"]


# ── End-to-end optimise ────────────────────────────────────────────────────


class TestOptimizeFactor:
    def test_numeric_only_finds_improvement(self) -> None:
        panel, close = _make_panel(signal_strength=1.0)
        inputs = prepare_inputs(
            panel,
            close,
            CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
            forward_period_bars=1,
        )
        config = TuneConfig(
            trials=8,
            dimensions=DimensionsConfig(numeric=True, operators=False, variables=False),
            register_top_k=2,
            seed=123,
        )
        result = optimize_factor(
            expression="-correlation(high, rank(volume), 5)",
            inputs=inputs,
            tune_config=config,
        )
        assert result.best_expression.startswith("-1 * correlation")
        assert result.n_trials > 0
        assert isinstance(result.best_icir_cv, float)
        assert len(result.top_k) >= 1

    def test_returns_empty_result_when_nothing_tunable(self) -> None:
        panel, close = _make_panel()
        inputs = prepare_inputs(
            panel,
            close,
            CVConfig(n_folds=2, test_ratio=0.2, holdout_ratio=0.2),
        )
        config = TuneConfig(
            trials=3,
            dimensions=DimensionsConfig(numeric=True, operators=False, variables=False),
            register_top_k=1,
            seed=123,
        )
        # No tunable parameters in this expression.
        result = optimize_factor(
            expression="close",
            inputs=inputs,
            tune_config=config,
        )
        assert result.best_expression == "close"
        assert result.n_trials == 0

    def test_operators_dim_expands_search(self) -> None:
        panel, close = _make_panel(signal_strength=1.0)
        inputs = prepare_inputs(
            panel,
            close,
            CVConfig(n_folds=2, test_ratio=0.15, holdout_ratio=0.15),
        )
        config = TuneConfig(
            trials=8,
            dimensions=DimensionsConfig(numeric=True, operators=True, variables=False),
            register_top_k=2,
            seed=123,
        )
        result = optimize_factor(
            expression="cs_rank(-correlation(high, rank(volume), 5))",
            inputs=inputs,
            tune_config=config,
        )
        # Operator comparison should report at least one op_* slot summary.
        assert result.operator_comparison is not None
        assert any(k.startswith("op_") for k in result.operator_comparison)
