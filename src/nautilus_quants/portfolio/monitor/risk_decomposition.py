# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Risk decomposition — decompose portfolio variance into factor-level budgets.

Pure function, no side effects. Called by SnapshotAggregatorActor.

    Var(r_p) = (X' w)' Σ_f (X' w) + w' D w
    Marginal contribution of factor k = exposure_k × (Σ_f × exposure)_k / total_var
"""

from __future__ import annotations

import numpy as np


def compute_risk_budget(
    weights: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[str, object]:
    """Decompose portfolio variance into factor + specific budgets.

    Parameters
    ----------
    weights : np.ndarray
        (N,) current portfolio weights (signed).
    factor_exposures : np.ndarray
        (N, K) latest factor exposure matrix X.
    factor_covariance : np.ndarray
        (K, K) factor covariance matrix Σ_f.
    specific_variance : np.ndarray
        (N,) idiosyncratic variance vector var_u.
    factor_names : tuple[str, ...]
        K factor names for labeling output.

    Returns
    -------
    dict
        ``{
            "total_vol": float,           # annualized if inputs are per-bar
            "factor_vol_share": {name: fraction, ...},
            "specific_vol_share": float,
        }``
        factor_vol_share values + specific_vol_share ≈ 1.0
    """
    # Portfolio factor exposure: (K,) = X' w
    exposure_vec = factor_exposures.T @ weights

    # Factor variance component: scalar
    factor_var = float(exposure_vec @ factor_covariance @ exposure_vec)

    # Specific variance component: scalar = w' D w
    specific_var = float(np.nansum(weights**2 * specific_variance))

    total_var = factor_var + specific_var
    if total_var <= 0:
        return {
            "total_vol": 0.0,
            "factor_vol_share": {name: 0.0 for name in factor_names},
            "specific_vol_share": 0.0,
        }

    # Marginal contribution per factor: exposure_k × (Σ_f × exposure)_k
    marginal = exposure_vec * (factor_covariance @ exposure_vec)  # (K,)
    factor_vol_share = {
        name: float(marginal[k] / total_var)
        for k, name in enumerate(factor_names)
    }

    return {
        "total_vol": float(np.sqrt(total_var)),
        "factor_vol_share": factor_vol_share,
        "specific_vol_share": float(specific_var / total_var),
    }
