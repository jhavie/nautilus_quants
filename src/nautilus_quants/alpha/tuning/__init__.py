# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Alpha factor expression tuning — parameter/operator/variable optimization.

Runs Optuna-based Bayesian search over three dimensions (numeric parameters,
operator substitution, variable substitution) with time-series cross-validation
to discover optimal factor configurations. New variants are written back to the
``test`` registry environment alongside LLM-mined factors; downstream
``promote`` decides which variants migrate to ``dev``.
"""

from __future__ import annotations

from nautilus_quants.alpha.tuning.config import (
    CandidatesConfig,
    CVConfig,
    DimensionsConfig,
    EligibilityConfig,
    OperatorAlternative,
    OperatorSlot,
    ParamSpec,
    TrialResult,
    TuneConfig,
    TuneResult,
    VariableGroup,
    VariableSlot,
)

__all__ = [
    "CandidatesConfig",
    "CVConfig",
    "DimensionsConfig",
    "EligibilityConfig",
    "OperatorAlternative",
    "OperatorSlot",
    "ParamSpec",
    "TrialResult",
    "TuneConfig",
    "TuneResult",
    "VariableGroup",
    "VariableSlot",
]
