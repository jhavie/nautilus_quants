# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""Operator mapping between nautilus_quants and Qlib/pandas references.

Each entry defines: (operator_class, qlib_expression_template, match_type, notes)

Match types:
  EXACT   — np.allclose with rtol=1e-6
  CORR    — correlation > threshold (batch vs incremental precision)
  SPEAR   — Spearman rank correlation (different semantics but monotonic)
  RATIO_N — nautilus / qlib ≈ N due to known Qlib bug
  PANDAS  — reference is pandas, not Qlib
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MatchType(Enum):
    EXACT = "exact"
    CORR = "correlation"
    SPEAR = "spearman"
    RATIO_N = "ratio_n"
    PANDAS = "pandas"


@dataclass(frozen=True)
class OperatorSpec:
    """Specification for one operator cross-validation."""

    name: str
    operator_cls_name: str  # class name in time_series.py / cross_sectional.py
    qlib_expr: str | None  # Qlib expression template (use {field}, {window})
    match_type: MatchType
    field: str = "close"  # OHLCV field to test on
    window: int = 20
    extra_kwargs: dict[str, Any] | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Time-Series operators
# ---------------------------------------------------------------------------
TS_OPERATORS: list[OperatorSpec] = [
    # --- EXACT match vs Qlib ---
    OperatorSpec("ts_mean", "TsMean", "Mean(${field}, {window})", MatchType.EXACT),
    OperatorSpec("ts_sum", "TsSum", "Sum(${field}, {window})", MatchType.EXACT),
    OperatorSpec("ts_std", "TsStd", "Std(${field}, {window})", MatchType.EXACT),
    OperatorSpec("ts_max", "TsMax", "Max(${field}, {window})", MatchType.EXACT),
    OperatorSpec("ts_min", "TsMin", "Min(${field}, {window})", MatchType.EXACT),
    OperatorSpec("delay", "Delay", "Ref(${field}, {window})", MatchType.EXACT, window=1),
    OperatorSpec("delta", "Delta", "Delta(${field}, {window})", MatchType.EXACT, window=5),
    OperatorSpec(
        "ts_argmax", "TsArgmax", "IdxMax(${field}, {window})", MatchType.EXACT,
    ),
    OperatorSpec(
        "ts_argmin", "TsArgmin", "IdxMin(${field}, {window})", MatchType.EXACT,
    ),
    OperatorSpec(
        "correlation", "Correlation",
        "Corr($close, $volume, {window})", MatchType.EXACT,
        notes="two-data operator: close vs volume",
    ),
    OperatorSpec(
        "covariance", "Covariance",
        "Cov($close, $volume, {window})", MatchType.EXACT,
        notes="two-data operator",
    ),
    OperatorSpec("ts_slope", "TsSlope", "Slope(${field}, {window})", MatchType.EXACT),
    OperatorSpec(
        "ts_rsquare", "TsRsquare", "Rsquare(${field}, {window})", MatchType.EXACT,
    ),
    OperatorSpec(
        "ts_percentile", "TsPercentile",
        "Quantile(${field}, {window}, 0.8)", MatchType.EXACT,
        extra_kwargs={"extra_0": 0.8},
    ),

    # --- CORR match (batch vs incremental precision) ---
    OperatorSpec(
        "ts_residual", "TsResidual", "Resi(${field}, {window})", MatchType.CORR,
        notes="batch vs incremental OLS — float cancellation amplifies diff",
    ),

    # --- SPEARMAN match (different semantics) ---
    OperatorSpec(
        "ts_rank", "TsRank", "Rank(${field}, {window})", MatchType.SPEAR,
        notes="nautilus: raw rank [1,d]; qlib: pct rank [0,1]",
    ),

    # --- RATIO_N (known Qlib WMA bug) ---
    OperatorSpec(
        "decay_linear", "DecayLinear", "WMA(${field}, {window})", MatchType.RATIO_N,
        notes="Qlib WMA has nanmean bug: result = correct / N",
    ),

    # --- PANDAS reference (no Qlib equivalent) ---
    OperatorSpec(
        "ema", "Ema", None, MatchType.PANDAS,
        window=12,
        notes="ref: pandas ewm(span=N, adjust=False).mean()",
    ),
    OperatorSpec(
        "ts_product", "TsProduct", None, MatchType.PANDAS,
        window=5,  # small window to avoid overflow on crypto prices
        notes="ref: pandas rolling(N).apply(np.prod) — use small window",
    ),
    OperatorSpec(
        "ts_skew", "TsSkew", None, MatchType.CORR,
        notes="ref: pandas rolling(N).skew() — float32/64 precision diff",
    ),
]

# ---------------------------------------------------------------------------
# Cross-Sectional operators (verified against pandas row-wise ops)
# ---------------------------------------------------------------------------
CS_OPERATORS: list[OperatorSpec] = [
    OperatorSpec(
        "cs_rank", "CsRank", None, MatchType.PANDAS,
        notes="ref: df.rank(axis=1, method='min', pct=True)",
    ),
    OperatorSpec(
        "cs_zscore", "CsZscore", None, MatchType.PANDAS,
        notes="ref: (x - row_mean) / row_std",
    ),
    OperatorSpec(
        "cs_demean", "CsDemean", None, MatchType.PANDAS,
        notes="ref: x - row_mean",
    ),
    OperatorSpec(
        "cs_scale", "CsScale", None, MatchType.PANDAS,
        notes="ref: x / abs(x).sum(axis=1) — per-row scale to unit abs sum",
    ),
    OperatorSpec(
        "cs_max", "CsMax", None, MatchType.PANDAS,
        notes="ref: broadcast row max to all columns",
    ),
    OperatorSpec(
        "cs_min", "CsMin", None, MatchType.PANDAS,
        notes="ref: broadcast row min to all columns",
    ),
    OperatorSpec(
        "normalize", "CsNormalize", None, MatchType.PANDAS,
        extra_kwargs={"use_std": True, "limit": 0},
        notes="ref: demean + divide by std (BRAIN normalize with useStd=true)",
    ),
    OperatorSpec(
        "clip_quantile", "CsClipQuantile", None, MatchType.PANDAS,
        extra_kwargs={"lower": 0.2, "upper": 0.8},
        notes="ref: clip each row to [quantile(lower), quantile(upper)]",
    ),
    OperatorSpec(
        "winsorize", "CsWinsorize", None, MatchType.PANDAS,
        extra_kwargs={"std_mult": 3.0},
        notes="ref: clip to [mean - N*std, mean + N*std] per row",
    ),
]

# ---------------------------------------------------------------------------
# Alpha expressions for E2E tests
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlphaSpec:
    """Specification for an alpha expression E2E test."""

    name: str
    expression: str
    qlib_expr: str | None  # Qlib equivalent (for pure TS alphas)
    pandas_ref: str | None  # Python function name in this module (for CS+TS)
    match_type: MatchType
    min_coverage: float = 0.5  # min fraction of non-NaN output


ALPHA_EXPRESSIONS: list[AlphaSpec] = [
    # Pure TS — direct Qlib comparison
    AlphaSpec(
        "alpha044_ts_only",
        "-1 * correlation(high, volume, 5)",
        "-1 * Corr($high, $volume, 5)",
        None,
        MatchType.EXACT,
    ),
    # CS+TS — needs decomposed pandas reference
    AlphaSpec(
        "alpha044_full",
        "-1 * correlation(high, rank(volume), 5)",
        None,
        "ref_alpha044",
        MatchType.CORR,
    ),
    AlphaSpec(
        "alpha158_MIN60",
        "ts_min(low, 60) / close",
        "Min($low, 60) / $close",
        None,
        MatchType.EXACT,
        min_coverage=0.3,  # needs 60-bar warmup
    ),
    AlphaSpec(
        "alpha158_MA5",
        "ts_mean(close, 5) / close",
        "Mean($close, 5) / $close",
        None,
        MatchType.EXACT,
    ),
    AlphaSpec(
        "alpha158_CORD10",
        "correlation(close / delay(close, 1), log(volume / delay(volume, 1) + 1), 10)",
        None,
        "ref_alpha158_cord10",
        MatchType.CORR,
    ),
]


# ---------------------------------------------------------------------------
# Decomposed pandas reference functions for CS+TS alphas
# ---------------------------------------------------------------------------
def ref_alpha044(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """-1 * correlation(high, rank(volume), 5)"""
    import pandas as pd

    cs_rank_vol = panels["volume"].rank(axis=1, method="min", pct=True)
    return -1 * panels["high"].rolling(5).corr(cs_rank_vol)


def ref_alpha158_cord10(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """correlation(close/delay(close,1), log(volume/delay(volume,1)+1), 10)"""
    import numpy as np

    close_ret = panels["close"] / panels["close"].shift(1)
    vol_ret = np.log(panels["volume"] / panels["volume"].shift(1) + 1)
    return close_ret.rolling(10).corr(vol_ret)


# Map function names to callables
PANDAS_REFS: dict[str, Any] = {
    "ref_alpha044": ref_alpha044,
    "ref_alpha158_cord10": ref_alpha158_cord10,
}
