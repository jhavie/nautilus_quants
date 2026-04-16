# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Portfolio concentration metrics — pure function, no side effects.

Herfindahl-Hirschman Index (HHI) and effective number of positions.
"""

from __future__ import annotations


def compute_concentration(weights: dict[str, float]) -> dict[str, float]:
    """Compute portfolio concentration metrics.

    Parameters
    ----------
    weights : dict[str, float]
        {instrument_id: signed_weight}. Long > 0, short < 0.

    Returns
    -------
    dict[str, float]
        ``{
            "n_effective": float,     # 1 / HHI (effective number of bets)
            "herfindahl": float,      # HHI = Σ (w_i / gross)²
            "max_abs_weight": float,  # max |w_i|
        }``
    """
    abs_w = [abs(w) for w in weights.values() if w != 0.0]
    if not abs_w:
        return {"n_effective": 0.0, "herfindahl": 0.0, "max_abs_weight": 0.0}

    gross = sum(abs_w)
    normed = [w / gross for w in abs_w]
    hhi = sum(w**2 for w in normed)

    return {
        "n_effective": 1.0 / hhi if hhi > 0 else 0.0,
        "herfindahl": hhi,
        "max_abs_weight": max(abs_w),
    }
