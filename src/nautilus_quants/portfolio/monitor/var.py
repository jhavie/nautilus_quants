# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Historical Simulation VaR and CVaR — pure function, no side effects.

Uses current weights w applied to historical instrument returns to compute
portfolio-level Value-at-Risk and Conditional VaR (Expected Shortfall).

    r_p(t) = w @ r(t)    "if I held today's portfolio through history"
    VaR_α  = -quantile(r_p, α)
    CVaR_α = -mean(r_p | r_p ≤ -VaR_α)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_historical_var(
    instrument_returns: np.ndarray,
    weights: np.ndarray | dict[str, float],
    instruments: Sequence[str] | None = None,
    alpha: float = 0.05,
    lookback: int | None = None,
    equity: float | None = None,
) -> dict[str, float]:
    """Compute Historical Simulation VaR and CVaR.

    Parameters
    ----------
    instrument_returns : np.ndarray
        (T, N) matrix of instrument returns (preprocessed).
    weights : np.ndarray or dict[str, float]
        (N,) weight vector aligned to columns of instrument_returns,
        or dict {instrument_id: weight} if ``instruments`` is provided.
    instruments : Sequence[str] | None
        Column labels for instrument_returns. Required when weights is dict.
    alpha : float
        VaR confidence level (default 0.05 = 95% VaR).
    lookback : int | None
        Use only the most recent ``lookback`` periods. None = all.
    equity : float | None
        Current portfolio equity in USDT. If provided, VaR/CVaR are also
        returned in absolute terms (var_usdt / cvar_usdt).

    Returns
    -------
    dict[str, float]
        ``{
            "alpha": float,
            "lookback_bars": int,
            "var_pct": float,         # percentage VaR (positive = loss)
            "cvar_pct": float,        # percentage CVaR
            "var_usdt": float | None, # absolute VaR (if equity given)
            "cvar_usdt": float | None,
            "sample_count": int,
        }``
    """
    if isinstance(weights, dict):
        if instruments is None:
            raise ValueError("instruments required when weights is dict")
        w = np.array([weights.get(inst, 0.0) for inst in instruments], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    r = np.asarray(instrument_returns, dtype=np.float64)
    if lookback is not None and lookback < r.shape[0]:
        r = r[-lookback:]

    # Portfolio returns: (T,)
    port_returns = r @ w
    valid = port_returns[np.isfinite(port_returns)]
    n = len(valid)

    if n < 10:
        return {
            "alpha": alpha,
            "lookback_bars": n,
            "var_pct": 0.0,
            "cvar_pct": 0.0,
            "var_usdt": None,
            "cvar_usdt": None,
            "sample_count": n,
        }

    var_pct = float(-np.quantile(valid, alpha))
    tail = valid[valid <= -var_pct]
    cvar_pct = float(-np.mean(tail)) if len(tail) > 0 else var_pct

    var_usdt = var_pct * equity if equity is not None else None
    cvar_usdt = cvar_pct * equity if equity is not None else None

    return {
        "alpha": alpha,
        "lookback_bars": n,
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "var_usdt": var_usdt,
        "cvar_usdt": cvar_usdt,
        "sample_count": n,
    }
