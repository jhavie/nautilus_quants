# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Stability / sensitivity analysis for tuned factors.

The optimizer ships a quick ``stability_score`` — this module expands on it
with a post-hoc analysis that can be consumed by the report generator. The
analysis evaluates every categorical parameter at ±N steps around the
optimum and returns a structured summary so the CLI can pretty-print a
"parameter landscape" table.

All evaluation is delegated to a caller-supplied ``evaluate_fn`` — the
module deliberately avoids importing ``optimizer`` to prevent a circular
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from nautilus_quants.alpha.tuning.config import ParamSpec


@dataclass(frozen=True)
class ParameterSensitivity:
    """Per-parameter trace of values → |ICIR|.

    ``best_value`` is the optimum returned by tuning; ``neighbours`` contains
    ``(candidate_value, icir_abs)`` pairs in grid order. ``drop_ratio`` is
    the worst neighbour's ratio against ``best_icir``.
    """

    name: str
    best_value: float
    best_icir_abs: float
    neighbours: tuple[tuple[float, float], ...]
    drop_ratio: float


@dataclass(frozen=True)
class SensitivityReport:
    """Aggregate sensitivity summary for a single ``TuneResult``."""

    best_icir_abs: float
    stability_score: float | None
    parameters: tuple[ParameterSensitivity, ...]

    def is_stable(self, threshold: float) -> bool:
        """Convenience: ``True`` iff every parameter clears ``threshold``."""
        if self.stability_score is None:
            return True
        return self.stability_score >= threshold


def analyze_parameter_stability(
    best_params: Mapping[str, float],
    numeric_specs: tuple[ParamSpec, ...],
    evaluate_fn: Callable[[dict[str, float]], float],
    *,
    perturbation_steps: int = 1,
) -> SensitivityReport:
    """Evaluate every tunable categorical parameter at ±``perturbation_steps``
    around its optimal value.

    Continuous-range parameters (``low``/``high`` with no ``values``) are
    skipped — perturbing them meaningfully requires sampling density that
    would add a second inner loop. Callers that care about continuous
    parameters should reformulate them as categorical ladders before
    invoking this function.
    """
    try:
        best_icir_abs = abs(float(evaluate_fn(dict(best_params))))
    except Exception:  # noqa: BLE001
        best_icir_abs = float("nan")

    if not np.isfinite(best_icir_abs) or best_icir_abs <= 0:
        return SensitivityReport(
            best_icir_abs=best_icir_abs,
            stability_score=None,
            parameters=(),
        )

    per_param: list[ParameterSensitivity] = []
    all_drops: list[float] = []

    for spec in numeric_specs:
        if not spec.is_tunable or spec.values is None:
            continue
        best_val = best_params.get(spec.name)
        if best_val is None:
            continue
        values = list(spec.values)
        try:
            idx = values.index(float(best_val))
        except ValueError:
            continue

        trace: list[tuple[float, float]] = []
        neighbour_icirs: list[float] = []
        for offset in range(-perturbation_steps, perturbation_steps + 1):
            neighbour_idx = idx + offset
            if not 0 <= neighbour_idx < len(values):
                continue
            candidate = float(values[neighbour_idx])
            perturbed = dict(best_params)
            perturbed[spec.name] = candidate
            try:
                icir_abs = abs(float(evaluate_fn(perturbed)))
            except Exception:  # noqa: BLE001
                continue
            trace.append((candidate, icir_abs))
            if offset != 0 and np.isfinite(icir_abs):
                neighbour_icirs.append(icir_abs)

        if neighbour_icirs:
            worst = min(neighbour_icirs)
            drop_ratio = worst / best_icir_abs if best_icir_abs > 0 else 0.0
        else:
            drop_ratio = 1.0
        all_drops.append(drop_ratio)

        per_param.append(
            ParameterSensitivity(
                name=spec.name,
                best_value=float(best_val),
                best_icir_abs=best_icir_abs,
                neighbours=tuple(trace),
                drop_ratio=float(drop_ratio),
            )
        )

    stability_score = min(all_drops) if all_drops else 1.0
    return SensitivityReport(
        best_icir_abs=best_icir_abs,
        stability_score=float(stability_score),
        parameters=tuple(per_param),
    )
