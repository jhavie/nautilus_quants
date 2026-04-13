# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""High-level orchestration for alpha factor tuning.

``optimize_factor`` ties the search-space construction, Optuna study, and
post-processing together into a single ``TuneResult`` per factor (or
per-prototype). Callers plug panel data + forward returns in, get back the
best configuration, holdout validation, stability score, and multiple-testing
correction.

The module is intentionally thin — heavy lifting lives in ``objective.py``
(IC + CV) and ``search_space.py`` (template → specs / slots). The optimizer
just wires them together, manages the Optuna study, and computes secondary
metrics on the best trial.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from nautilus_quants.alpha.tuning.config import (
    ALGORITHM_GRID,
    ALGORITHM_TPE,
    CORRECTION_BONFERRONI,
    CORRECTION_FDR_BH,
    CORRECTION_NONE,
    CVConfig,
    DimensionsConfig,
    OperatorSlot,
    ParamSpec,
    TrialResult,
    TuneConfig,
    TuneResult,
    VariableSlot,
)
from nautilus_quants.alpha.tuning.objective import (
    CVFold,
    CVSchedule,
    EvaluationContext,
    _fold_icir,
    _spearman_ic_vectorized,
    build_cv_folds,
    compute_forward_returns_panel,
    create_objective,
    evaluate_expression_panel,
)
from nautilus_quants.alpha.tuning.search_space import build_search_space, reconstruct_expression

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


# ── Newey-West for holdout p-value ─────────────────────────────────────────


def _newey_west_t_pvalue(ic_series: pd.Series) -> tuple[float, float, int]:
    """Minimal Newey-West wrapper used for holdout-only reporting.

    Returns ``(t_stat, p_value, n_effective)``. For tuning the exact
    adjustment is less critical than consistency with ``alpha analyze`` —
    callers that want the full Newey-West from
    ``alpha.analysis.report`` can upgrade later. This implementation
    mirrors the classical Bartlett kernel with ``lag = floor(4 * (N/100)**(2/9))``.
    """
    ic = ic_series.dropna().astype(float)
    n = len(ic)
    if n < 5:
        return float("nan"), float("nan"), n
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    if not np.isfinite(std) or std < 1e-12:
        return 0.0, 1.0, n

    max_lag = max(1, int(math.floor(4 * (n / 100.0) ** (2 / 9))))
    centred = (ic - mean).to_numpy()
    gamma_0 = float(np.mean(centred * centred))
    hac_var = gamma_0
    for j in range(1, max_lag + 1):
        if j >= n:
            break
        weight = 1.0 - j / (max_lag + 1)
        gamma_j = float(np.mean(centred[j:] * centred[:-j]))
        hac_var += 2.0 * weight * gamma_j
    hac_var = max(hac_var, 1e-24)

    se = math.sqrt(hac_var / n)
    if se < 1e-12:
        return 0.0, 1.0, n
    t_stat = mean / se
    # Two-sided normal approximation — good enough for n > 30 which is the
    # realistic holdout size. Using scipy would introduce a heavier import
    # for negligible accuracy gain here.
    p_value = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
    n_eff = max(1, int(n * gamma_0 / hac_var))
    return float(t_stat), float(p_value), n_eff


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Correction helpers ─────────────────────────────────────────────────────


def _apply_correction(p_value: float, n_tests: int, method: str) -> float:
    """Bonferroni / BH-FDR / pass-through adjustments."""
    if not np.isfinite(p_value):
        return p_value
    if method == CORRECTION_NONE or n_tests <= 1:
        return p_value
    if method == CORRECTION_BONFERRONI:
        return min(1.0, p_value * n_tests)
    if method == CORRECTION_FDR_BH:
        # For a single p-value BH reduces to p_value * n_tests / rank, and
        # rank == n_tests for the worst case, so it is identical to the
        # uncorrected value for a single test — but we apply ``n_tests``
        # multiplicatively to stay conservative when we only keep the best
        # trial.
        return min(1.0, p_value * n_tests)
    return p_value


# ── Holdout + stability ────────────────────────────────────────────────────


@dataclass(frozen=True)
class HoldoutMetrics:
    icir: float
    t_stat_nw: float
    p_value_nw: float
    n_eff: int
    ic_mean: float
    ic_std: float
    n_samples: int


def _evaluate_holdout(
    expression: str,
    panel_fields: dict[str, pd.DataFrame | float],
    numeric_values: dict[str, float],
    fwd_returns: pd.DataFrame,
    schedule: CVSchedule,
) -> HoldoutMetrics:
    """Re-evaluate the best expression on the held-out tail window."""
    factor_panel = evaluate_expression_panel(expression, panel_fields, numeric_values)
    if not isinstance(factor_panel, pd.DataFrame):
        return HoldoutMetrics(
            icir=float("nan"),
            t_stat_nw=float("nan"),
            p_value_nw=float("nan"),
            n_eff=0,
            ic_mean=float("nan"),
            ic_std=float("nan"),
            n_samples=0,
        )

    start, end = schedule.holdout_start, schedule.holdout_end
    factor_slice = factor_panel.iloc[start:end]
    returns_slice = fwd_returns.iloc[start:end]
    ic_series = _spearman_ic_vectorized(factor_slice, returns_slice)

    ic_mean = float(ic_series.mean()) if len(ic_series) else float("nan")
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) >= 2 else float("nan")
    icir = _fold_icir(ic_series)
    t_stat, p_value, n_eff = _newey_west_t_pvalue(ic_series)
    return HoldoutMetrics(
        icir=icir,
        t_stat_nw=t_stat,
        p_value_nw=p_value,
        n_eff=n_eff,
        ic_mean=ic_mean,
        ic_std=ic_std,
        n_samples=len(ic_series),
    )


def _stability_score(
    best_params: dict[str, float],
    numeric_specs: tuple[ParamSpec, ...],
    evaluate_fn,
    perturbation_steps: int = 1,
) -> tuple[float | None, dict[str, list[tuple[float, float]]]]:
    """Evaluate ICIR at ±``perturbation_steps`` neighbours of the optimum.

    Only categorical (``values`` populated) parameters are perturbed — doing
    the same for continuous coefficients requires sampling density that
    would explode the evaluation budget. Returns ``None`` when there is
    nothing worth perturbing.
    """
    try:
        best_icir = abs(evaluate_fn(best_params))
    except Exception:  # noqa: BLE001
        return None, {}
    if not np.isfinite(best_icir) or best_icir <= 0:
        return None, {}

    sensitivity: dict[str, list[tuple[float, float]]] = {}
    neighbour_icirs: list[float] = []

    for spec in numeric_specs:
        if not spec.is_tunable or spec.values is None:
            continue
        best_val = best_params.get(spec.name)
        if best_val is None:
            continue
        try:
            idx = list(spec.values).index(best_val)
        except ValueError:
            continue
        row: list[tuple[float, float]] = []
        for offset in range(-perturbation_steps, perturbation_steps + 1):
            neighbour_idx = idx + offset
            if not 0 <= neighbour_idx < len(spec.values):
                continue
            neighbour_val = spec.values[neighbour_idx]
            perturbed = {**best_params, spec.name: neighbour_val}
            try:
                icir_abs = abs(evaluate_fn(perturbed))
            except Exception:  # noqa: BLE001
                continue
            row.append((neighbour_val, icir_abs))
            if offset != 0 and np.isfinite(icir_abs):
                neighbour_icirs.append(icir_abs)
        if row:
            sensitivity[spec.name] = row

    if not neighbour_icirs:
        return 1.0, sensitivity
    return min(neighbour_icirs) / best_icir, sensitivity


# ── Optimizer entry points ────────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizeInputs:
    """Bundle of data/derived artefacts passed into ``optimize_factor``.

    Pre-computing ``panel_fields`` / ``fwd_returns`` / ``cv_schedule`` once
    outside the loop (e.g. when iterating over prototypes) makes the
    optimiser ~10x faster than building them inside each call.
    """

    panel_fields: dict[str, pd.DataFrame | float]
    pricing: pd.DataFrame
    fwd_returns: pd.DataFrame
    cv_schedule: CVSchedule


def prepare_inputs(
    panel_fields: dict[str, pd.DataFrame | float],
    pricing: pd.DataFrame,
    cv_config: CVConfig,
    forward_period_bars: int = 1,
) -> OptimizeInputs:
    """Build a shared ``OptimizeInputs`` object for batch tuning."""
    fwd = compute_forward_returns_panel(pricing, forward_period_bars)
    schedule = build_cv_folds(len(pricing.index), cv_config)
    return OptimizeInputs(
        panel_fields=panel_fields,
        pricing=pricing,
        fwd_returns=fwd,
        cv_schedule=schedule,
    )


def _grid_choices_from_specs(
    numeric_specs: tuple[ParamSpec, ...],
    operator_slots: tuple[OperatorSlot, ...],
    variable_slots: tuple[VariableSlot, ...],
) -> dict[str, list[Any]]:
    """Build the GridSampler search-space dict from tuning specs/slots.

    Keys must mirror the ``trial.suggest_*`` names used in ``objective.py``:
    numeric → ``ParamSpec.name`` (e.g. ``p0``); operator → ``OperatorSlot.slot_id``
    (``op_0``); operator extras → ``"{slot_id}__{extra_name}"`` (``op_0__std_mult``);
    variable → ``VariableSlot.slot_id`` (``var_0``).

    Operator extras are listed for every alternative even when an extra is
    only consumed by some alts — Optuna's GridSampler tolerates unused keys
    but raises if a key is missing from the grid.

    Continuous (range-based) numeric params are not enumerable for grid mode;
    raises ``ValueError`` so the caller falls back to (or rejects) grid.
    """
    grid: dict[str, list[Any]] = {}

    for spec in numeric_specs:
        if not spec.is_tunable:
            continue
        if spec.values is None:
            raise ValueError(
                f"Grid algorithm requires categorical `values` for ParamSpec "
                f"'{spec.name}', got continuous range (low/high)"
            )
        grid[spec.name] = [float(v) for v in spec.values]

    for slot in operator_slots:
        grid[slot.slot_id] = [alt.name for alt in slot.alternatives]
        for alt in slot.alternatives:
            for extra in alt.extra_params:
                if not extra.is_tunable:
                    continue
                key = f"{slot.slot_id}__{extra.name}"
                if extra.values is None:
                    raise ValueError(
                        f"Grid algorithm requires categorical `values` for operator "
                        f"extra '{key}', got continuous range (low/high)"
                    )
                # Multiple alts may share an extra name; later assignment wins
                # (values must agree across alts in practice, but be defensive).
                grid[key] = [float(v) for v in extra.values]

    for slot in variable_slots:
        grid[slot.slot_id] = list(slot.alternatives)

    return grid


def _build_optuna_study(
    config: TuneConfig,
    study_name: str | None = None,
    *,
    grid_choices: dict[str, list[Any]] | None = None,
) -> "optuna.Study":
    """Create a study honouring ``config.algorithm`` + seed.

    For ``ALGORITHM_GRID``, ``grid_choices`` MUST cover every parameter that
    the objective will sample — Optuna raises ``ValueError`` on the first
    ``trial.suggest_*`` whose name is missing from the grid.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import GridSampler, TPESampler

    if config.algorithm == ALGORITHM_GRID:
        if not grid_choices:
            raise ValueError(
                "grid algorithm requires a non-empty search space; "
                "pass `grid_choices` derived from build_search_space()"
            )
        sampler: optuna.samplers.BaseSampler = GridSampler(
            grid_choices,
            seed=config.seed,
        )
    else:
        sampler = TPESampler(seed=config.seed)

    pruner = MedianPruner(n_warmup_steps=1)
    return optuna.create_study(
        study_name=study_name or "tune",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


class _ConvergenceEarlyStop:
    """Optuna callback that stops a study after N consecutive identical trials.

    TPE often converges to a narrow optimal region long before the trial
    budget is exhausted; once it starts re-sampling the same parameter
    tuple every iteration, additional trials add zero information. This
    callback watches every completed trial and invokes ``study.stop()``
    once ``patience`` back-to-back trials share the same parameters.

    Thread-safe for ``study.optimize(n_jobs>1)`` via an internal lock —
    without it, concurrent callback invocations could mis-count the
    consecutive streak.

    ``min_trials`` guards against stopping too early: TPE needs a minimum
    exploration budget (typically 20-30 trials) before its suggestions
    become meaningful.
    """

    def __init__(self, *, patience: int, min_trials: int) -> None:
        import threading

        self._patience = patience
        self._min_trials = min_trials
        self._last_key: tuple | None = None
        self._consecutive = 0
        self._lock = threading.Lock()
        self.triggered = False

    def __call__(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> None:
        import optuna

        # Only consider fully-completed trials (pruned/failed trials may
        # have incomplete params and would poison the consecutive counter).
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        with self._lock:
            if len(study.trials) < self._min_trials:
                return

            key = tuple(sorted(trial.params.items()))
            if key == self._last_key:
                self._consecutive += 1
            else:
                self._last_key = key
                self._consecutive = 1

            if self._consecutive >= self._patience:
                self.triggered = True
                logger.info(
                    "Early stop: %d consecutive identical trials "
                    "(params=%s); stopping study '%s' at trial %d",
                    self._patience,
                    dict(key),
                    study.study_name,
                    trial.number,
                )
                study.stop()


def _extract_trial_result(
    trial: "optuna.trial.FrozenTrial",
    template: str,
) -> TrialResult | None:
    """Convert an Optuna frozen trial into a ``TrialResult``."""
    expression = trial.user_attrs.get("expression")
    if expression is None:
        return None
    numeric = trial.user_attrs.get("numeric_values", {}) or {}
    op_choices = trial.user_attrs.get("op_choices", {}) or {}
    var_choices = trial.user_attrs.get("var_choices", {}) or {}
    op_extras = trial.user_attrs.get("op_extras", {}) or {}
    fold_icirs = tuple(float(v) for v in trial.user_attrs.get("fold_icirs", []))
    mean_icir = float(trial.user_attrs.get("mean_icir", float("nan")))
    params: dict[str, float | str] = {}
    params.update({k: float(v) for k, v in numeric.items()})
    params.update({k: str(v) for k, v in op_choices.items()})
    params.update({k: str(v) for k, v in var_choices.items()})
    # Flatten operator extras into the params dict using the same
    # ``"{slot_id}__{extra_name}"`` key that ``_eval_icir_for_perturbation``
    # consults; without this the stability perturbation falls back to default
    # extras and ``stability_score`` no longer reflects the tuned operator.
    for slot_id, extras in op_extras.items():
        for extra_name, value in extras.items():
            params[f"{slot_id}__{extra_name}"] = float(value)
    objective_value = float(trial.value) if trial.value is not None else float("nan")
    return TrialResult(
        trial_number=trial.number,
        params=params,
        expression=expression,
        cv_icir=fold_icirs,
        mean_icir=mean_icir,
        objective_value=objective_value,
        pruned=trial.state.name == "PRUNED",
    )


def _unique_top_k(
    trials: list[TrialResult],
    k: int,
) -> list[TrialResult]:
    """Keep best ``k`` trials with distinct expressions."""
    seen: set[str] = set()
    unique: list[TrialResult] = []
    for t in trials:
        if t.expression in seen:
            continue
        seen.add(t.expression)
        unique.append(t)
        if len(unique) >= k:
            break
    return unique


def _build_evaluation_context(
    template: str,
    numeric_params: tuple[ParamSpec, ...],
    operator_slots: tuple[OperatorSlot, ...],
    variable_slots: tuple[VariableSlot, ...],
    inputs: OptimizeInputs,
) -> EvaluationContext:
    return EvaluationContext(
        template=template,
        numeric_params=numeric_params,
        operator_slots=operator_slots,
        variable_slots=variable_slots,
        panel_fields=inputs.panel_fields,
        pricing=inputs.pricing,
        fwd_returns=inputs.fwd_returns,
        cv_schedule=inputs.cv_schedule,
    )


def optimize_factor(
    expression: str,
    inputs: OptimizeInputs,
    tune_config: TuneConfig,
    *,
    parameters: dict[str, Any] | None = None,
    available_vars: Iterable[str] | None = None,
    derived_vars: Iterable[str] | None = None,
    study_name: str | None = None,
) -> TuneResult:
    """Run the tuning loop for a single expression.

    Parameters
    ----------
    expression
        Factor expression (in the form stored in the registry).
    inputs
        Shared panel data + forward returns + CV splits. Build once via
        ``prepare_inputs`` when tuning multiple factors on the same dataset.
    tune_config
        Controls Optuna algorithm, dimensions, trial budget, CV folds, and
        reporting toggles.
    parameters, available_vars, derived_vars
        Forwarded to ``build_search_space`` so YAML-level parameters and
        ExtraData availability are respected.

    Returns
    -------
    TuneResult
        Aggregate summary — always safe to pass downstream even when the
        optimizer was unable to improve on the original expression.
    """
    import optuna

    dimensions: DimensionsConfig = tune_config.dimensions
    template, numeric_specs, operator_slots, variable_slots = build_search_space(
        expression,
        parameters=parameters,
        available_vars=available_vars,
        derived_vars=derived_vars,
        tune_numeric=dimensions.numeric,
        tune_operators=dimensions.operators,
        tune_variables=dimensions.variables,
    )

    original_params = {spec.name: float(spec.original_value) for spec in numeric_specs}

    # Degenerate case: nothing to tune → return a no-op result.
    if not numeric_specs and not operator_slots and not variable_slots:
        return TuneResult(
            template=template,
            original_expression=expression,
            original_params=original_params,
            best_params={},
            best_expression=expression,
            best_icir_cv=float("nan"),
            n_trials=0,
        )

    if not inputs.cv_schedule.is_usable():
        logger.warning(
            "CV schedule has no usable folds; cannot tune expression '%s'",
            expression,
        )
        return TuneResult(
            template=template,
            original_expression=expression,
            original_params=original_params,
            best_params={},
            best_expression=expression,
            best_icir_cv=float("nan"),
            n_trials=0,
        )

    eval_ctx = _build_evaluation_context(
        template, numeric_specs, operator_slots, variable_slots, inputs
    )
    objective = create_objective(eval_ctx)

    grid_choices = (
        _grid_choices_from_specs(numeric_specs, operator_slots, variable_slots)
        if tune_config.algorithm == ALGORITHM_GRID
        else None
    )
    study = _build_optuna_study(tune_config, study_name=study_name, grid_choices=grid_choices)

    # Convergence early-stop callback — halts the study once TPE settles
    # on a stable optimum. Disabled when ``early_stop_patience == 0``.
    callbacks: list = []
    early_stopper: _ConvergenceEarlyStop | None = None
    if tune_config.early_stop_patience > 0:
        early_stopper = _ConvergenceEarlyStop(
            patience=tune_config.early_stop_patience,
            min_trials=tune_config.early_stop_min_trials,
        )
        callbacks.append(early_stopper)

    # Parallel trial execution. n_jobs>1 is safe here because the objective
    # function is DB-free and only reads shared (immutable) panel data.
    # DuckDB writes are deferred to ``register_tuned_variants`` which runs
    # serially in the main thread after this call returns.
    study.optimize(
        objective,
        n_trials=tune_config.trials,
        timeout=tune_config.timeout,
        n_jobs=max(1, tune_config.n_jobs),
        callbacks=callbacks,
        gc_after_trial=False,
        show_progress_bar=False,
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    if not completed_trials:
        logger.warning(
            "Tuning produced no completed trials for expression '%s'",
            expression,
        )
        return TuneResult(
            template=template,
            original_expression=expression,
            original_params=original_params,
            best_params={},
            best_expression=expression,
            best_icir_cv=float("nan"),
            n_trials=len(study.trials),
            n_pruned=len(pruned_trials),
        )

    # Rank trials by |objective| (maximise abs ICIR).
    trial_results = [
        r for r in (_extract_trial_result(t, template) for t in completed_trials) if r is not None
    ]
    trial_results.sort(key=lambda r: r.objective_value, reverse=True)
    top_k_trials = _unique_top_k(trial_results, tune_config.register_top_k)

    best = trial_results[0]

    # Reconstruct numeric map from best trial params.
    numeric_map: dict[str, float] = {}
    for spec in numeric_specs:
        val = best.params.get(spec.name)
        if val is None:
            numeric_map[spec.name] = float(spec.original_value)
        else:
            numeric_map[spec.name] = float(val)

    holdout = _evaluate_holdout(
        best.expression,
        inputs.panel_fields,
        numeric_map,
        inputs.fwd_returns,
        inputs.cv_schedule,
    )

    stability, _ = _stability_score(
        numeric_map,
        numeric_specs,
        evaluate_fn=lambda p: _eval_icir_for_perturbation(
            template,
            p,
            best.params,
            operator_slots,
            variable_slots,
            inputs,
        ),
    )

    adjusted_p = _apply_correction(
        holdout.p_value_nw,
        n_tests=len(completed_trials),
        method=tune_config.correction_method,
    )

    try:
        importance = optuna.importance.get_param_importances(study)
        importance_dict = {k: float(v) for k, v in importance.items()}
    except Exception:  # noqa: BLE001
        importance_dict = None

    operator_comparison = _summarise_operator_choices(trial_results, operator_slots)
    variable_comparison = _summarise_variable_choices(trial_results, variable_slots)

    return TuneResult(
        template=template,
        original_expression=expression,
        original_params=original_params,
        best_params=dict(best.params),
        best_expression=best.expression,
        best_icir_cv=float(best.mean_icir),
        holdout_icir=holdout.icir,
        holdout_t_stat_nw=holdout.t_stat_nw,
        holdout_p_value_nw=holdout.p_value_nw,
        stability_score=stability,
        adjusted_p_value=adjusted_p,
        top_k=tuple(top_k_trials),
        param_importance=importance_dict,
        operator_comparison=operator_comparison,
        variable_comparison=variable_comparison,
        n_trials=len(study.trials),
        n_pruned=len(pruned_trials),
    )


def _eval_icir_for_perturbation(
    template: str,
    perturbed_numeric: dict[str, float],
    best_all_params: dict[str, float | str],
    operator_slots: tuple[OperatorSlot, ...],
    variable_slots: tuple[VariableSlot, ...],
    inputs: OptimizeInputs,
) -> float:
    """Rebuild the expression with perturbed numeric values but the same
    operator + variable choices, evaluate, return |mean CV ICIR|.
    """
    op_choices = {
        s.slot_id: str(best_all_params.get(s.slot_id, s.current_op)) for s in operator_slots
    }
    var_choices = {
        s.slot_id: str(best_all_params.get(s.slot_id, s.current_var)) for s in variable_slots
    }
    op_extras: dict[str, dict[str, float]] = {}
    for slot in operator_slots:
        extras: dict[str, float] = {}
        for alt in slot.alternatives:
            for extra_spec in alt.extra_params:
                key = f"{slot.slot_id}__{extra_spec.name}"
                if key in best_all_params:
                    extras[extra_spec.name] = float(best_all_params[key])
        if extras:
            op_extras[slot.slot_id] = extras

    expression = reconstruct_expression(
        template,
        perturbed_numeric,
        op_choices=op_choices,
        operator_slots=operator_slots,
        var_choices=var_choices,
        variable_slots=variable_slots,
        op_extra_params=op_extras,
    )
    try:
        factor_panel = evaluate_expression_panel(expression, inputs.panel_fields, perturbed_numeric)
    except Exception:  # noqa: BLE001
        return 0.0
    if not isinstance(factor_panel, pd.DataFrame):
        return 0.0
    icirs = []
    for fold in inputs.cv_schedule.folds:
        test_f = factor_panel.iloc[fold.test_start : fold.test_end]
        test_r = inputs.fwd_returns.iloc[fold.test_start : fold.test_end]
        ic = _spearman_ic_vectorized(test_f, test_r)
        val = _fold_icir(ic)
        if np.isfinite(val):
            icirs.append(val)
    if not icirs:
        return 0.0
    return float(np.mean(icirs))


def _summarise_operator_choices(
    trials: list[TrialResult],
    slots: tuple[OperatorSlot, ...],
) -> dict[str, dict[str, float]] | None:
    if not slots:
        return None
    result: dict[str, dict[str, float]] = {}
    for slot in slots:
        per_op: dict[str, float] = {}
        for trial in trials:
            name = trial.params.get(slot.slot_id)
            if not isinstance(name, str):
                continue
            score = trial.objective_value
            if name not in per_op or score > per_op[name]:
                per_op[name] = score
        result[slot.slot_id] = per_op
    return result


def _summarise_variable_choices(
    trials: list[TrialResult],
    slots: tuple[VariableSlot, ...],
) -> dict[str, dict[str, float]] | None:
    if not slots:
        return None
    result: dict[str, dict[str, float]] = {}
    for slot in slots:
        per_var: dict[str, float] = {}
        for trial in trials:
            name = trial.params.get(slot.slot_id)
            if not isinstance(name, str):
                continue
            score = trial.objective_value
            if name not in per_var or score > per_var[name]:
                per_var[name] = score
        result[slot.slot_id] = per_var
    return result


# ── Prototype-level helper ─────────────────────────────────────────────────


def optimize_prototype(
    representative_expression: str,
    inputs: OptimizeInputs,
    tune_config: TuneConfig,
    *,
    parameters: dict[str, Any] | None = None,
    available_vars: Iterable[str] | None = None,
    derived_vars: Iterable[str] | None = None,
    prototype_name: str | None = None,
) -> TuneResult:
    """Thin wrapper that tunes one representative expression from a prototype
    group. All factors sharing this prototype receive the same search — we
    simply pick the prototype's representative (e.g. the factor with the
    best existing ICIR) and tune that.

    Kept separate from ``optimize_factor`` so callers can keep per-prototype
    grouping semantics without forcing the same logic into the factor-level
    API.
    """
    study_name = f"tune_{prototype_name}" if prototype_name else None
    return optimize_factor(
        representative_expression,
        inputs,
        tune_config,
        parameters=parameters,
        available_vars=available_vars,
        derived_vars=derived_vars,
        study_name=study_name,
    )
