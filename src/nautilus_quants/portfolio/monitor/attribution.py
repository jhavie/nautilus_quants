# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
P&L attribution — decompose portfolio return into named-factor contributions.

Pure function, no side effects. Called by SnapshotAggregatorActor.

    total_return = w' r = (w' X) f + w' u
                       = Σ_k exposure_k × f_k  +  Σ_i w_i × u_i
"""

from __future__ import annotations

import numpy as np


def compute_pnl_attribution(
    weights: np.ndarray,
    factor_exposures: np.ndarray,
    factor_returns_t: np.ndarray,
    specific_returns_t: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[str, object]:
    """Decompose the latest-period portfolio return by factor.

    Parameters
    ----------
    weights : np.ndarray
        (N,) current portfolio weights (signed).
    factor_exposures : np.ndarray
        (N, K) latest-period factor exposure matrix X.
    factor_returns_t : np.ndarray
        (K,) latest-period factor returns f_t from WLS/PCA.
    specific_returns_t : np.ndarray
        (N,) latest-period idiosyncratic residuals u_t.
    factor_names : tuple[str, ...]
        K factor names for labeling output.

    Returns
    -------
    dict
        ``{
            "period_return_total": float,
            "by_factor": {factor_name: contribution, ...},
            "specific": float,
        }``
    """
    # Portfolio exposure per factor: (K,) = X' w
    exposure_vec = factor_exposures.T @ weights  # (K,)

    # Per-factor contribution: exposure_k × f_k
    factor_contrib = exposure_vec * factor_returns_t  # (K,)
    specific_contrib = float(np.nansum(weights * specific_returns_t))

    by_factor = {
        name: float(factor_contrib[k])
        for k, name in enumerate(factor_names)
    }
    total = float(np.nansum(factor_contrib)) + specific_contrib

    return {
        "period_return_total": total,
        "by_factor": by_factor,
        "specific": specific_contrib,
    }
