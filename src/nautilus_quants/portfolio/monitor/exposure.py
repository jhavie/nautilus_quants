# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio exposure computation — pure function, no side effects, no alerting.

Produces raw exposure data for the ``snapshot:risk`` JSON that
SnapshotAggregatorActor writes to Nautilus Cache. Grafana reads from Redis
and handles alerting via its own rule engine — there is intentionally no
limit-breach check in this module.

Exposure breadcrumbs (returned as plain Python dicts, Grafana-friendly):
- factor_exposures: portfolio signed exposure per named risk factor (X' w)
- sector_exposures: per-sector gross exposure (sum of |w_i| inside each sector)
- gross, net, long, short: top-level portfolio statistics
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nautilus_quants.portfolio.types import RiskModelOutput


def _align_weights_to_instruments(
    weights: dict[str, float],
    instruments: Sequence[str],
) -> np.ndarray:
    """Produce ordered weight vector aligned to instruments; missing → 0.0."""
    return np.array([weights.get(inst, 0.0) for inst in instruments], dtype=np.float64)


def compute_portfolio_exposure(
    weights: dict[str, float],
    output: RiskModelOutput,
) -> dict[str, object]:
    """Compute raw portfolio exposure breadcrumbs for monitoring.

    Parameters
    ----------
    weights : dict[str, float]
        Current portfolio weights keyed by instrument ID (signed: long>0, short<0).
    output : RiskModelOutput
        Latest risk model output. factor_exposures (N, K) required for
        named-factor exposures; sector_map required for sector aggregation.

    Returns
    -------
    dict[str, object]
        Structure for direct JSON serialization:

        ``{
            "gross": float,           # sum(|w_i|)
            "net": float,             # sum(w_i)
            "long": float,            # sum(max(w_i, 0))
            "short": float,           # sum(min(w_i, 0))
            "factor_exposures": {factor_name: signed_exposure, ...},
            "sector_exposures": {sector_name: gross_exposure, ...},
        }``

        ``factor_exposures`` is empty for non-interpretable (PCA) outputs;
        ``sector_exposures`` is empty if ``output.sector_map`` is None.
    """
    gross = float(sum(abs(w) for w in weights.values()))
    net = float(sum(weights.values()))
    long = float(sum(w for w in weights.values() if w > 0))
    short = float(sum(w for w in weights.values() if w < 0))

    factor_exposures: dict[str, float] = {}
    if output.is_decomposed and output.factor_exposures is not None and output.factor_names:
        w_vec = _align_weights_to_instruments(weights, output.instruments)
        exposures = output.factor_exposures.T @ w_vec  # (K,) = X' w
        factor_exposures = {name: float(val) for name, val in zip(output.factor_names, exposures)}

    sector_exposures: dict[str, float] = {}
    if output.sector_map:
        # Mirror fundamental.py._assemble_sector_dummies: unmapped → "Other".
        # Keeping the two defaults in sync ensures the sector pie chart shows
        # the same buckets the risk model itself used for cov/constraint.
        for inst, w in weights.items():
            sector = output.sector_map.get(inst, "Other")
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + abs(float(w))

    return {
        "gross": gross,
        "net": net,
        "long": long,
        "short": short,
        "factor_exposures": factor_exposures,
        "sector_exposures": sector_exposures,
    }
