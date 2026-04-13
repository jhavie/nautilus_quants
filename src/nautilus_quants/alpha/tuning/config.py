# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Data classes for alpha factor tuning.

Defines the configuration surface (``TuneConfig`` and children) and the
runtime state objects (``ParamSpec``, ``OperatorSlot``, ``VariableSlot``,
``TrialResult``, ``TuneResult``) used by the tuning engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Parameter / operator / variable specs ───────────────────────────────────

# Parameter type tags used during search-space classification.
PARAM_TYPE_WINDOW = "window"
PARAM_TYPE_THRESHOLD = "threshold"
PARAM_TYPE_COEFFICIENT = "coefficient"
PARAM_TYPE_SIGN = "sign"
PARAM_TYPE_FIXED = "fixed"
VALID_PARAM_TYPES = frozenset(
    {
        PARAM_TYPE_WINDOW,
        PARAM_TYPE_THRESHOLD,
        PARAM_TYPE_COEFFICIENT,
        PARAM_TYPE_SIGN,
        PARAM_TYPE_FIXED,
    }
)

# Variable scope tags — controls which substitutions are legal.
VAR_SCOPE_PER_INSTRUMENT = "per_instrument"
VAR_SCOPE_BROADCAST = "broadcast"
VALID_VAR_SCOPES = frozenset({VAR_SCOPE_PER_INSTRUMENT, VAR_SCOPE_BROADCAST})


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single tunable numeric parameter.

    ``values`` (categorical) and ``low``/``high`` (continuous) are mutually
    exclusive. ``sign`` and ``fixed`` types are not tuned — their
    ``original_value`` is reused unchanged during reconstruction.
    """

    name: str
    param_type: str
    original_value: float
    values: tuple[float, ...] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.param_type not in VALID_PARAM_TYPES:
            raise ValueError(f"Invalid param_type '{self.param_type}', expected one of " f"{sorted(VALID_PARAM_TYPES)}")
        if self.param_type in (PARAM_TYPE_SIGN, PARAM_TYPE_FIXED):
            return
        has_values = self.values is not None
        has_range = self.low is not None and self.high is not None
        if not has_values and not has_range:
            raise ValueError(
                f"ParamSpec '{self.name}' must provide either `values` " "(categorical) or `low`+`high` (continuous range)"
            )
        if has_values and has_range:
            raise ValueError(f"ParamSpec '{self.name}' cannot set both `values` and `low`/`high`")

    @property
    def is_tunable(self) -> bool:
        return self.param_type not in (PARAM_TYPE_SIGN, PARAM_TYPE_FIXED)


@dataclass(frozen=True)
class OperatorAlternative:
    """One alternative operator inside an ``OperatorSlot``.

    ``args_template`` is a positional format string whose placeholders are
    filled during expression reconstruction. For instance, ``cs_rank`` uses
    ``"({inner})"`` while ``winsorize`` uses ``"({inner}, {std_mult})"`` —
    placeholders must match fields in ``extra_params``.
    """

    name: str
    args_template: str = "({inner})"
    extra_params: tuple[ParamSpec, ...] = ()


@dataclass(frozen=True)
class OperatorSlot:
    """An operator position in an expression that admits substitution.

    ``slot_id`` is a stable string (e.g. ``"op_0"``) used as the Optuna
    categorical parameter name. ``inner_expr`` is the serialized inner
    expression that must be preserved when the outer operator changes.
    """

    slot_id: str
    current_op: str
    group: str
    alternatives: tuple[OperatorAlternative, ...]
    inner_expr: str
    position: str = "outer"

    def alt_by_name(self, name: str) -> OperatorAlternative | None:
        for alt in self.alternatives:
            if alt.name == name:
                return alt
        return None


@dataclass(frozen=True)
class VariableGroup:
    """A semantically-equivalent set of variables that may substitute.

    The ``scope`` guards against mixing per-instrument and broadcast data
    (e.g. ``btc_close`` cannot replace ``close``).
    """

    members: tuple[str, ...]
    scope: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.scope not in VALID_VAR_SCOPES:
            raise ValueError(f"Invalid scope '{self.scope}', expected one of " f"{sorted(VALID_VAR_SCOPES)}")


@dataclass(frozen=True)
class VariableSlot:
    """A variable reference in the expression that admits substitution.

    ``alternatives`` has already been filtered through (a) availability in the
    current ``ExtraDataManager`` universe and (b) scope compatibility.
    ``positions`` records every AST location referencing this variable — when
    a substitution happens, all occurrences flip together to keep the
    expression consistent.
    """

    slot_id: str
    current_var: str
    group_name: str
    alternatives: tuple[str, ...]
    positions: tuple[tuple[int, ...], ...] = ()
    scope: str = VAR_SCOPE_PER_INSTRUMENT


# ── Cross-validation ────────────────────────────────────────────────────────

CV_METHOD_EXPANDING = "expanding"
CV_METHOD_ROLLING = "rolling"
VALID_CV_METHODS = frozenset({CV_METHOD_EXPANDING, CV_METHOD_ROLLING})


@dataclass(frozen=True)
class CVConfig:
    """Time-series cross-validation configuration.

    ``test_ratio`` and ``holdout_ratio`` are fractions of the full timeline;
    ``n_folds`` test windows are placed immediately before the holdout.
    ``gap_bars`` separates train from test to prevent target leakage when
    forward returns straddle the boundary.
    """

    method: str = CV_METHOD_EXPANDING
    n_folds: int = 3
    test_ratio: float = 0.167
    holdout_ratio: float = 0.167
    gap_bars: int = 0

    def __post_init__(self) -> None:
        if self.method not in VALID_CV_METHODS:
            raise ValueError(f"Invalid cv.method '{self.method}', expected one of " f"{sorted(VALID_CV_METHODS)}")
        if self.n_folds < 1:
            raise ValueError(f"cv.n_folds must be >= 1, got {self.n_folds}")
        if not 0.0 < self.test_ratio < 1.0:
            raise ValueError(f"cv.test_ratio must be in (0, 1), got {self.test_ratio}")
        if not 0.0 <= self.holdout_ratio < 1.0:
            raise ValueError(f"cv.holdout_ratio must be in [0, 1), got {self.holdout_ratio}")
        if self.test_ratio * self.n_folds + self.holdout_ratio >= 1.0:
            raise ValueError(
                "cv.test_ratio * n_folds + holdout_ratio must leave room for "
                f"training data (got {self.test_ratio} * {self.n_folds} + "
                f"{self.holdout_ratio} >= 1.0)"
            )


# ── Tuning configuration ────────────────────────────────────────────────────

ALGORITHM_TPE = "tpe"
ALGORITHM_GRID = "grid"
VALID_ALGORITHMS = frozenset({ALGORITHM_TPE, ALGORITHM_GRID})

CORRECTION_BONFERRONI = "bonferroni"
CORRECTION_FDR_BH = "fdr_bh"
CORRECTION_NONE = "none"
VALID_CORRECTIONS = frozenset({CORRECTION_BONFERRONI, CORRECTION_FDR_BH, CORRECTION_NONE})


@dataclass(frozen=True)
class EligibilityConfig:
    """Pre-tune filter — decides which factors are worth tuning.

    This is *not* the promote ``HardFilterConfig``. The promote filter is
    "is this good enough to ship to production"; eligibility is "does this
    have enough signal that tuning could plausibly produce a winner".

    Design philosophy
    -----------------
    A factor with |ICIR|=0.05 on its current parameters can realistically
    improve to 0.15 with tuning (≈3x). A factor with |ICIR|=0.02 is most
    likely noise — even after tuning it tends to land at 0.04-0.06,
    consuming search budget for no gain. The defaults below split the
    middle: high enough to skip noise, low enough that "potential" factors
    still get a chance.

    Key gates (all AND-ed within a single period; OR across periods so a
    factor that excels at any one horizon survives):

    - ``icir_abs_min``: minimum |ICIR| magnitude — rules out pure noise
    - ``t_stat_nw_abs_min``: Newey-West significance — rules out spurious
      strong-looking ICIRs from short or autocorrelated samples
    - ``coverage_min`` / ``n_samples_min``: data quality gates
    - ``min_valid_periods``: how many of the available forward-return
      horizons must clear all gates simultaneously (cross-period
      consistency check; ≥2 sharply reduces overfitting risk)
    """

    icir_abs_min: float = 0.05
    t_stat_nw_abs_min: float = 1.5
    coverage_min: float = 0.30
    n_samples_min: int = 1000
    min_valid_periods: int = 1


@dataclass(frozen=True)
class CandidatesConfig:
    """How to select factors for tuning from the registry.

    The four filters (``prototype``, ``source``, ``status``, ``tags``) AND
    together. ``eligibility`` is an additional metric-based gate.
    """

    env: str = "test"
    prototype: str | None = None
    source: str | None = None
    status: str | None = "candidate"
    tags: tuple[str, ...] = ()
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)


@dataclass(frozen=True)
class DimensionsConfig:
    """Which tuning dimensions are active. Each dimension multiplies the
    search space; keep ``operators`` / ``variables`` off unless explicitly
    exploring structural alternatives.
    """

    numeric: bool = True
    operators: bool = False
    variables: bool = False


@dataclass(frozen=True)
class TuneConfig:
    """Top-level configuration for the ``alpha tune`` command.

    Mirrors the ``tune:`` section of ``alpha_mining.yaml``. Defaults are tuned
    for a "safe first run" — only numeric parameters are tuned, by prototype,
    with TPE + 50 trials.
    """

    enabled: bool = False
    source: str = "tune"
    output_dir: str = "logs/alpha_tune"
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    by_prototype: bool = True
    algorithm: str = ALGORITHM_TPE
    trials: int = 50
    # Number of parallel trials inside one Optuna study. The objective
    # function is DB-free, so threads only share immutable panel data —
    # safe even without a lock. Typical speed-up is 2-2.5x at n_jobs=4
    # because pandas / numpy release the GIL for the heavy operators,
    # while the AST walk remains Python-bound.
    n_jobs: int = 1
    # Convergence early-stop. After ``early_stop_min_trials`` Optuna has
    # explored enough of the space to justify checking for convergence;
    # if the most recent ``early_stop_patience`` trials all sampled the
    # same parameter tuple, TPE has almost certainly settled on its
    # optimum region and further trials are pure redundancy. Setting
    # ``early_stop_patience = 0`` disables the feature.
    early_stop_patience: int = 20
    early_stop_min_trials: int = 30
    timeout: int = 3600
    seed: int = 42
    register_top_k: int = 3
    search_space_overrides: dict[str, Any] = field(default_factory=dict)
    cv: CVConfig = field(default_factory=CVConfig)
    stability_min: float = 0.5
    correction_method: str = CORRECTION_BONFERRONI
    significance_alpha: float = 0.05
    charts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(f"Invalid algorithm '{self.algorithm}', expected one of " f"{sorted(VALID_ALGORITHMS)}")
        if self.correction_method not in VALID_CORRECTIONS:
            raise ValueError(
                f"Invalid correction_method '{self.correction_method}', " f"expected one of {sorted(VALID_CORRECTIONS)}"
            )
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if self.early_stop_patience < 0:
            raise ValueError(
                f"early_stop_patience must be >= 0 (0 = disabled), "
                f"got {self.early_stop_patience}"
            )
        if self.early_stop_min_trials < 1:
            raise ValueError(
                f"early_stop_min_trials must be >= 1, got {self.early_stop_min_trials}"
            )
        if self.register_top_k < 1:
            raise ValueError(f"register_top_k must be >= 1, got {self.register_top_k}")
        if not 0.0 <= self.stability_min <= 1.0:
            raise ValueError(f"stability_min must be in [0, 1], got {self.stability_min}")
        if not 0.0 < self.significance_alpha < 1.0:
            raise ValueError("significance_alpha must be in (0, 1), got " f"{self.significance_alpha}")


# ── Trial / result ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrialResult:
    """One evaluated point in the search space.

    ``params`` carries both numeric values and operator/variable names; the
    downstream caller differentiates by key prefix (``op_*``, ``var_*``).
    """

    trial_number: int
    params: dict[str, float | str]
    expression: str
    cv_icir: tuple[float, ...]
    mean_icir: float
    objective_value: float
    pruned: bool = False


@dataclass(frozen=True)
class TuneResult:
    """Aggregate outcome of tuning a single factor or prototype group."""

    template: str
    original_expression: str
    original_params: dict[str, float]
    best_params: dict[str, float | str]
    best_expression: str
    best_icir_cv: float
    holdout_icir: float | None = None
    holdout_t_stat_nw: float | None = None
    holdout_p_value_nw: float | None = None
    stability_score: float | None = None
    adjusted_p_value: float | None = None
    top_k: tuple[TrialResult, ...] = ()
    param_importance: dict[str, float] | None = None
    operator_comparison: dict[str, dict[str, float]] | None = None
    variable_comparison: dict[str, dict[str, float]] | None = None
    n_trials: int = 0
    n_pruned: int = 0
