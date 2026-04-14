# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio exposure computation — pure functions, no side effects.

Used by SnapshotAggregatorActor to produce the ``snapshot:risk`` JSON
alongside its existing 5 monitoring snapshots (venue/execution/factor/
strategy/health). No new Actor needed — monitoring flows through the
existing Cache → Grafana pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nautilus_quants.portfolio.types import RiskModelOutput


@dataclass(frozen=True)
class Breach:
    """A constraint breach record for risk alerting.

    Attributes
    ----------
    kind : str
        "factor" | "sector" | "net" | "leverage" | "single_weight".
    name : str
        Specific factor or sector name (e.g. "btc_beta", "L1").
    limit : float
        Threshold that was exceeded.
    actual : float
        Observed value.
    """

    kind: str
    name: str
    limit: float
    actual: float


def _align_weights_to_instruments(
    weights: dict[str, float],
    instruments: Sequence[str],
) -> np.ndarray:
    """Produce ordered weight vector aligned to instruments; missing → 0.0."""
    return np.array([weights.get(inst, 0.0) for inst in instruments], dtype=np.float64)


def compute_portfolio_exposure(
    weights: dict[str, float],
    output: RiskModelOutput,
) -> dict[str, float]:
    """Compute per-factor portfolio exposure: exposure_k = X_k' w.

    Only meaningful for interpretable (Fundamental) risk models. For Statistical
    (PCA) models, factor names are synthetic ("PC_0", "PC_1", ...) and callers
    should check ``output.is_interpretable`` before displaying or alerting.

    Parameters
    ----------
    weights : dict[str, float]
        Current portfolio weights keyed by instrument ID.
    output : RiskModelOutput
        Latest risk model output containing factor_exposures (N, K).

    Returns
    -------
    dict[str, float]
        Factor name → portfolio signed exposure. Empty if no decomposition.
    """
    if not output.is_decomposed or output.factor_exposures is None or not output.factor_names:
        return {}
    w = _align_weights_to_instruments(weights, output.instruments)
    exposures = output.factor_exposures.T @ w  # (K,) = X' w
    return {name: float(val) for name, val in zip(output.factor_names, exposures)}


def check_factor_limits(
    exposures: dict[str, float],
    limits: dict[str, float] | None,
) -> list[Breach]:
    """Detect breaches where |exposure| > limit.

    Parameters
    ----------
    exposures : dict[str, float]
        Factor name → signed exposure (from compute_portfolio_exposure).
    limits : dict[str, float] | None
        Factor name → max allowed absolute exposure. None disables all checks.

    Returns
    -------
    list[Breach]
        All factor breaches (empty if none or limits is None).
    """
    if not limits:
        return []
    breaches: list[Breach] = []
    for name, value in exposures.items():
        limit = limits.get(name)
        if limit is None:
            continue
        if abs(value) > limit:
            breaches.append(Breach(kind="factor", name=name, limit=limit, actual=value))
    return breaches


def check_sector_limits(
    weights: dict[str, float],
    sector_map: dict[str, str],
    limits: dict[str, float] | None,
) -> list[Breach]:
    """Detect per-sector gross-exposure breaches.

    Sector exposure is defined as sum(|w_i| for i in sector) — gross weight
    inside the sector. Use ``limits`` like {"L1": 0.4, "DeFi": 0.3, ...}.

    Parameters
    ----------
    weights : dict[str, float]
        Current portfolio weights.
    sector_map : dict[str, str]
        Instrument ID → sector name mapping.
    limits : dict[str, float] | None
        Sector name → max gross exposure. None disables checks.

    Returns
    -------
    list[Breach]
        Sector exposure breaches.
    """
    if not limits:
        return []
    totals: dict[str, float] = {}
    for inst, w in weights.items():
        sector = sector_map.get(inst)
        if sector is None:
            continue
        totals[sector] = totals.get(sector, 0.0) + abs(float(w))
    breaches: list[Breach] = []
    for sector, total in totals.items():
        limit = limits.get(sector)
        if limit is None:
            continue
        if total > limit:
            breaches.append(Breach(kind="sector", name=sector, limit=limit, actual=total))
    return breaches
