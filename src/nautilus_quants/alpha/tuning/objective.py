# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Optuna objective and cross-validation helpers for the tuning engine.

The heavy inner loop runs tens of thousands of factor evaluations per tuning
session, so it avoids the alphalens-reloaded pipeline used by ``alpha
analyze``. Instead:

* ``build_cv_folds`` carves out walk-forward splits on the bar grid once.
* ``_spearman_ic_vectorized`` computes rank-IC via ``pandas.DataFrame.corrwith``
  — dense linear algebra, no per-row Python loop.
* ``create_objective`` snapshots the panel + pricing, parses the template
  once, and returns an Optuna-compatible callable that reuses everything.

All three sit below the public ``optimizer.TuneOptimizer`` / ``optimize_*``
entry points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from nautilus_quants.alpha.tuning.config import (
    CV_METHOD_EXPANDING,
    CV_METHOD_ROLLING,
    PARAM_TYPE_COEFFICIENT,
    PARAM_TYPE_FIXED,
    PARAM_TYPE_SIGN,
    PARAM_TYPE_THRESHOLD,
    PARAM_TYPE_WINDOW,
    CVConfig,
    OperatorSlot,
    ParamSpec,
    VariableSlot,
)
from nautilus_quants.alpha.tuning.search_space import reconstruct_expression
from nautilus_quants.factors.engine.evaluator import Evaluator
from nautilus_quants.factors.expression.parser import parse_expression
from nautilus_quants.factors.operators.cross_sectional import CS_OPERATOR_INSTANCES
from nautilus_quants.factors.operators.math import MATH_OPERATORS
from nautilus_quants.factors.operators.time_series import TS_OPERATOR_INSTANCES

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# ── CV folds ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CVFold:
    """A single walk-forward cross-validation split.

    Indices are row indices into the common timestamp axis shared by the
    factor panel and the forward-returns frame.
    """

    fold_idx: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


@dataclass(frozen=True)
class CVSchedule:
    """Collection of fold splits + dedicated holdout window."""

    folds: tuple[CVFold, ...]
    holdout_start: int
    holdout_end: int
    total_timestamps: int

    def is_usable(self) -> bool:
        return len(self.folds) > 0 and self.holdout_end > self.holdout_start


def build_cv_folds(n_timestamps: int, config: CVConfig) -> CVSchedule:
    """Build an expanding / rolling window schedule.

    Layout (expanding, defaults):

    ::

        ├────────── train ──────────┼── test ──┼── holdout ──┤
        0                   train_end  test_end  N

    ``test_ratio * n_folds`` stacked test windows are placed immediately
    before a ``holdout_ratio`` buffer. Each fold's ``train_end`` is pulled
    back by ``gap_bars`` to give forward-return overlap room.
    """
    if n_timestamps <= 0:
        return CVSchedule((), 0, 0, n_timestamps)

    holdout_size = int(n_timestamps * config.holdout_ratio)
    usable_end = n_timestamps - holdout_size
    test_size = max(1, int(n_timestamps * config.test_ratio))
    total_test = test_size * config.n_folds

    if usable_end - total_test <= 0:
        logger.warning(
            "build_cv_folds: not enough data for %d folds of size %d " "(usable_end=%d, total_test=%d)",
            config.n_folds,
            test_size,
            usable_end,
            total_test,
        )
        return CVSchedule((), usable_end, n_timestamps, n_timestamps)

    folds: list[CVFold] = []
    for i in range(config.n_folds):
        test_end = usable_end - (config.n_folds - 1 - i) * test_size
        test_start = test_end - test_size
        if config.method == CV_METHOD_EXPANDING:
            train_start = 0
        elif config.method == CV_METHOD_ROLLING:
            train_start = max(0, test_start - test_size * 2)
        else:  # pragma: no cover — validated in config
            raise ValueError(f"Unknown cv method: {config.method}")
        train_end = max(train_start + 1, test_start - config.gap_bars)
        if train_end <= train_start or test_start >= test_end:
            continue
        folds.append(
            CVFold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return CVSchedule(
        folds=tuple(folds),
        holdout_start=usable_end,
        holdout_end=n_timestamps,
        total_timestamps=n_timestamps,
    )


# ── Fast IC computation ────────────────────────────────────────────────────


def _spearman_ic_vectorized(
    factor_panel: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    min_assets: int = 5,
) -> pd.Series:
    """Row-wise Spearman rank correlation between factor and forward returns.

    Both inputs are panels ``index=timestamp, columns=instrument``. The
    correlation is computed via pandas vectorised rank + ``corrwith``. Rows
    with fewer than ``min_assets`` non-NaN pairs produce ``NaN`` IC, which
    is then dropped.
    """
    if factor_panel.empty or fwd_returns.empty:
        return pd.Series(dtype=float)

    common_idx = factor_panel.index.intersection(fwd_returns.index)
    common_cols = factor_panel.columns.intersection(fwd_returns.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return pd.Series(dtype=float)

    f = factor_panel.loc[common_idx, common_cols]
    r = fwd_returns.loc[common_idx, common_cols]

    valid_counts = (~f.isna() & ~r.isna()).sum(axis=1)
    keep_mask = valid_counts >= min_assets
    if not keep_mask.any():
        return pd.Series(dtype=float)

    f = f.loc[keep_mask]
    r = r.loc[keep_mask]

    f_ranks = f.rank(axis=1, method="average")
    r_ranks = r.rank(axis=1, method="average")

    ic = f_ranks.corrwith(r_ranks, axis=1)
    ic = ic.dropna()
    return ic


def _fold_icir(ic_series: pd.Series, min_observations: int = 10) -> float:
    """ICIR for a single fold. Returns NaN on insufficient data."""
    if ic_series is None or len(ic_series) < min_observations:
        return float("nan")
    mean = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return mean / std


# ── Forward-returns helper ─────────────────────────────────────────────────


def compute_forward_returns_panel(
    pricing: pd.DataFrame,
    period_bars: int = 1,
) -> pd.DataFrame:
    """Simple forward-return panel at ``period_bars`` horizon.

    ``pricing`` is ``index=timestamp, columns=instrument`` closes. The output
    is aligned to the same axes — a row at timestamp ``t`` holds the return
    realised over ``(t, t + period_bars]``.
    """
    if pricing.empty:
        return pricing
    return pricing.pct_change(periods=period_bars, fill_method=None).shift(-period_bars)


# ── Panel factor evaluation ────────────────────────────────────────────────


def evaluate_expression_panel(
    expression: str,
    panel_fields: dict[str, pd.DataFrame | float],
    parameters: dict[str, float] | None = None,
) -> pd.DataFrame | float:
    """Parse + evaluate ``expression`` on the shared panel, returning the
    full factor DataFrame.

    Shared across ``create_objective`` (inner loop) and holdout validation.
    """
    ast = parse_expression(expression)
    evaluator = Evaluator(
        panel_fields=panel_fields,
        ts_ops=TS_OPERATOR_INSTANCES,
        cs_ops=CS_OPERATOR_INSTANCES,
        math_ops=MATH_OPERATORS,
        parameters=parameters or {},
    )
    return evaluator.evaluate(ast)


# ── Sampling from Optuna ───────────────────────────────────────────────────


def _sample_numeric(
    trial: "optuna.Trial",
    specs: tuple[ParamSpec, ...],
) -> dict[str, float]:
    """Ask Optuna for values for every tunable numeric parameter."""
    values: dict[str, float] = {}
    for spec in specs:
        if spec.param_type in (PARAM_TYPE_SIGN, PARAM_TYPE_FIXED):
            values[spec.name] = float(spec.original_value)
            continue
        if spec.values is not None:
            # ``suggest_categorical`` infers type from the tuple.
            chosen = trial.suggest_categorical(spec.name, list(spec.values))
            values[spec.name] = float(chosen)
            continue
        if spec.low is None or spec.high is None:
            # Defensive — should have been caught by ParamSpec validation.
            values[spec.name] = float(spec.original_value)
            continue
        if spec.log_scale:
            values[spec.name] = float(trial.suggest_float(spec.name, spec.low, spec.high, log=True))
        elif spec.step is not None:
            values[spec.name] = float(trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step))
        else:
            values[spec.name] = float(trial.suggest_float(spec.name, spec.low, spec.high))
    return values


def _sample_operators(
    trial: "optuna.Trial",
    slots: tuple[OperatorSlot, ...],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Ask Optuna for a chosen alternative (plus any extras) per slot."""
    choices: dict[str, str] = {}
    extras: dict[str, dict[str, float]] = {}
    for slot in slots:
        names = [alt.name for alt in slot.alternatives]
        chosen = trial.suggest_categorical(slot.slot_id, names)
        choices[slot.slot_id] = chosen
        alt = slot.alt_by_name(chosen)
        if alt is None or not alt.extra_params:
            continue
        extra_values: dict[str, float] = {}
        for extra in alt.extra_params:
            key = f"{slot.slot_id}__{extra.name}"
            if extra.values is not None:
                extra_values[extra.name] = float(trial.suggest_categorical(key, list(extra.values)))
            elif extra.low is not None and extra.high is not None:
                if extra.log_scale:
                    extra_values[extra.name] = float(trial.suggest_float(key, extra.low, extra.high, log=True))
                else:
                    extra_values[extra.name] = float(trial.suggest_float(key, extra.low, extra.high))
            else:
                extra_values[extra.name] = float(extra.original_value)
        extras[slot.slot_id] = extra_values
    return choices, extras


def _sample_variables(
    trial: "optuna.Trial",
    slots: tuple[VariableSlot, ...],
) -> dict[str, str]:
    """Ask Optuna which variable to use at each slot."""
    choices: dict[str, str] = {}
    for slot in slots:
        chosen = trial.suggest_categorical(slot.slot_id, list(slot.alternatives))
        choices[slot.slot_id] = chosen
    return choices


# ── Objective ──────────────────────────────────────────────────────────────


@dataclass
class EvaluationContext:
    """Frozen context passed into the Optuna objective closure."""

    template: str
    numeric_params: tuple[ParamSpec, ...]
    operator_slots: tuple[OperatorSlot, ...]
    variable_slots: tuple[VariableSlot, ...]
    panel_fields: dict[str, pd.DataFrame | float]
    pricing: pd.DataFrame
    fwd_returns: pd.DataFrame
    cv_schedule: CVSchedule


def create_objective(
    ctx: EvaluationContext,
    *,
    monotonicity_weight: float = 0.0,
    enable_pruning: bool = True,
) -> Callable[["optuna.Trial"], float]:
    """Build an Optuna-compatible objective callable for the given context.

    The returned function:

    1. Samples numeric / operator / variable choices from ``trial``.
    2. Reconstructs the expression deterministically.
    3. Evaluates the factor across the shared panel.
    4. Computes CV-fold ICIR using ``_spearman_ic_vectorized`` and averages
       them into a single scalar — higher is better in magnitude.
    5. Reports fold-level progress for Optuna's pruner.
    """

    import optuna

    def objective(trial: "optuna.Trial") -> float:
        numeric_values = _sample_numeric(trial, ctx.numeric_params)
        op_choices, op_extras = _sample_operators(trial, ctx.operator_slots)
        var_choices = _sample_variables(trial, ctx.variable_slots)

        try:
            expression = reconstruct_expression(
                ctx.template,
                numeric_values,
                op_choices=op_choices,
                operator_slots=ctx.operator_slots,
                var_choices=var_choices,
                variable_slots=ctx.variable_slots,
                op_extra_params=op_extras,
            )
            factor_panel = evaluate_expression_panel(expression, ctx.panel_fields, numeric_values)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trial %d failed to evaluate: %s", trial.number, exc)
            raise optuna.TrialPruned()

        if not isinstance(factor_panel, pd.DataFrame) or factor_panel.empty:
            raise optuna.TrialPruned()

        trial.set_user_attr("expression", expression)
        trial.set_user_attr("numeric_values", dict(numeric_values))
        trial.set_user_attr("op_choices", dict(op_choices))
        trial.set_user_attr("var_choices", dict(var_choices))
        trial.set_user_attr("op_extras", {k: dict(v) for k, v in op_extras.items()})

        fold_icirs: list[float] = []
        for fold in ctx.cv_schedule.folds:
            test_factor = factor_panel.iloc[fold.test_start : fold.test_end]
            test_returns = ctx.fwd_returns.iloc[fold.test_start : fold.test_end]
            ic_series = _spearman_ic_vectorized(test_factor, test_returns)
            icir = _fold_icir(ic_series)
            fold_icirs.append(icir)
            if enable_pruning:
                report_value = 0.0 if not np.isfinite(icir) else abs(icir)
                trial.report(report_value, fold.fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        trial.set_user_attr("fold_icirs", list(fold_icirs))
        finite = [v for v in fold_icirs if np.isfinite(v)]
        if not finite:
            raise optuna.TrialPruned()

        mean_icir = float(np.mean(finite))
        trial.set_user_attr("mean_icir", mean_icir)

        # The optimiser maximises; Optuna's convention is fine with signed
        # ICIR because Spearman is symmetric — strong negative ICIR factors
        # should be surfaced too, so we optimise ``|mean_icir|``.
        score = abs(mean_icir)
        if monotonicity_weight > 0.0:
            # Proxy for monotonicity: |IC_mean| magnitude agreement across
            # folds (tight cluster → high monotonicity). Cheap enough to
            # compute inline.
            spread = float(np.std(finite))
            score += monotonicity_weight * max(0.0, abs(mean_icir) - spread)
        return score

    return objective
