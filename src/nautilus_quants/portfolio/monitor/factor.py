# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor health and realized IC — pure functions for Grafana monitoring.

Called by SnapshotAggregatorActor (health) and FactorEngineActor (IC).
No side effects, no config, no alerting — raw metrics only.
Grafana handles thresholds and alerting via its own rule engine.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _avg_rank(arr: np.ndarray) -> np.ndarray:
    """Compute tie-aware average ranks (same as scipy rankdata, no dependency)."""
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    # Average ranks for tied values
    i = 0
    n = len(arr)
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        if j > i + 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    """Fast Spearman rank correlation with tie-aware average ranks.

    Returns NaN if either input is constant (zero variance in ranks).
    """
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    # Constant input → undefined correlation
    if np.ptp(arr_a) == 0 or np.ptp(arr_b) == 0:
        return float("nan")
    ra = _avg_rank(arr_a)
    rb = _avg_rank(arr_b)
    # Pearson correlation of ranks
    ra_mean = np.mean(ra)
    rb_mean = np.mean(rb)
    da = ra - ra_mean
    db = rb - rb_mean
    denom = np.sqrt(np.dot(da, da) * np.dot(db, db))
    if denom < 1e-15:
        return float("nan")
    return float(np.dot(da, db) / denom)


def compute_factor_health(
    current: dict[str, dict[str, float]],
    previous: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, Any]]:
    """Compute per-factor health metrics from factor values.

    Parameters
    ----------
    current
        Current factor values ``{factor_name: {instrument_id: value}}``.
    previous
        Previous bar's factor values, or ``None`` on first call.

    Returns
    -------
    dict
        ``{factor_name: {metric: value}}``.
        ``stuck_count`` and ``turnover`` only present when *previous* is not None.
    """
    if not current:
        return {}

    # All instruments across all factors (union)
    all_instruments: set[str] = set()
    for fvals in current.values():
        all_instruments.update(fvals.keys())

    total = len(all_instruments)
    if total == 0:
        return {}

    result: dict[str, dict[str, Any]] = {}

    for fname, fvals in current.items():
        # Anomaly detection: missing + inf unified
        present_keys = set(fvals.keys())
        missing = all_instruments - present_keys
        inf_insts = [k for k, v in fvals.items() if not math.isfinite(v)]
        anomaly_set = sorted(missing | set(inf_insts)) if (missing or inf_insts) else []
        anomaly_count = len(anomaly_set)

        # Valid values (finite only)
        valid_values = [v for v in fvals.values() if math.isfinite(v)]
        n_valid = len(valid_values)

        # Dispersion & kurtosis
        dispersion = 0.0
        kurtosis = 0.0
        if n_valid >= 2:
            arr = np.array(valid_values)
            dispersion = float(np.std(arr))
            if dispersion > 1e-12:
                mean = np.mean(arr)
                m2 = np.mean((arr - mean) ** 2)
                m4 = np.mean((arr - mean) ** 4)
                kurtosis = float(m4 / (m2 * m2) - 3.0)

        entry: dict[str, Any] = {
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / total,
            "anomaly_instruments": anomaly_set,
            "dispersion": dispersion,
            "kurtosis": kurtosis,
            "instrument_count": n_valid,
        }

        # Stuck count & turnover: only when previous available for this factor
        if previous is not None and fname in previous:
            prev_fvals = previous[fname]
            common = [
                inst
                for inst in fvals
                if inst in prev_fvals
                and math.isfinite(fvals[inst])
                and math.isfinite(prev_fvals[inst])
            ]
            if common:
                entry["stuck_count"] = sum(1 for inst in common if fvals[inst] == prev_fvals[inst])
            else:
                entry["stuck_count"] = 0

            if len(common) >= 3:
                cur_vals = [fvals[inst] for inst in common]
                prev_vals = [prev_fvals[inst] for inst in common]
                corr = _spearman(cur_vals, prev_vals)
                entry["turnover"] = float(1.0 - corr) if math.isfinite(corr) else 0.0
            elif common:
                entry["turnover"] = 0.0

        result[fname] = entry

    return result


def compute_factor_ic(
    prev_factors: dict[str, dict[str, float]],
    close_current: dict[str, float],
    close_previous: dict[str, float],
) -> dict[str, float]:
    """Compute realized IC: Spearman(factor[t-1], return[t-1→t]) per factor.

    Parameters
    ----------
    prev_factors
        Factor values from the previous bar ``{factor_name: {inst: value}}``.
    close_current
        Current bar close prices ``{inst: price}``.
    close_previous
        Previous bar close prices ``{inst: price}``.

    Returns
    -------
    dict
        ``{factor_name: ic_value}``. Factors with fewer than 3 valid
        instruments are skipped.
    """
    if not prev_factors:
        return {}

    # Cross-sectional returns
    returns: dict[str, float] = {}
    for inst in close_current:
        prev_price = close_previous.get(inst, 0.0)
        if prev_price > 0:
            returns[inst] = close_current[inst] / prev_price - 1.0

    if len(returns) < 3:
        return {}

    result: dict[str, float] = {}

    for fname, fvals in prev_factors.items():
        common = [inst for inst in fvals if inst in returns and math.isfinite(fvals[inst])]
        if len(common) < 3:
            continue

        factor_arr = [fvals[inst] for inst in common]
        return_arr = [returns[inst] for inst in common]

        corr = _spearman(factor_arr, return_arr)
        if math.isfinite(corr):
            result[fname] = float(corr)

    return result
