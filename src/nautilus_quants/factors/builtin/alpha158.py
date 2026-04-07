# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Alpha158 Built-in Factors — programmatically generated from qlib's Alpha158DL.

Generates 157 factors (VWAP0 skipped) across categories:
  - kbar (9): Candlestick shape features
  - price (3): Normalized price features (OPEN0, HIGH0, LOW0)
  - Rolling (145 = 29 types x 5 windows): Momentum, volatility, volume, etc.

Reference: https://github.com/microsoft/qlib (Alpha158DL feature config)

Only uses OHLCV fields: open, high, low, close, volume. No vwap.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS = [5, 10, 20, 30, 60]

# ---------------------------------------------------------------------------
# Category definitions for rolling factors
# ---------------------------------------------------------------------------

_ROLLING_CATEGORY = {
    "ROC": "momentum",
    "MA": "momentum",
    "STD": "volatility",
    "BETA": "momentum",
    "RSQR": "momentum",
    "RESI": "momentum",
    "MAX": "volatility",
    "MIN": "volatility",
    "QTLU": "volatility",
    "QTLD": "volatility",
    "RANK": "momentum",
    "RSV": "momentum",
    "IMAX": "momentum",
    "IMIN": "momentum",
    "IMXD": "momentum",
    "CORR": "correlation",
    "CORD": "correlation",
    "CNTP": "count",
    "CNTN": "count",
    "CNTD": "count",
    "SUMP": "rsi_like",
    "SUMN": "rsi_like",
    "SUMD": "rsi_like",
    "VMA": "volume",
    "VSTD": "volume",
    "WVMA": "volume",
    "VSUMP": "volume",
    "VSUMN": "volume",
    "VSUMD": "volume",
}

_ROLLING_DESCRIPTION = {
    "ROC": "Rate of change: lagged close / current close",
    "MA": "Moving average of close / current close",
    "STD": "Rolling std of close / current close",
    "BETA": "Rolling slope of close / current close",
    "RSQR": "Rolling R-squared of close trend",
    "RESI": "Rolling regression residual of close / current close",
    "MAX": "Rolling max of high / current close",
    "MIN": "Rolling min of low / current close",
    "QTLU": "80th percentile of close / current close",
    "QTLD": "20th percentile of close / current close",
    "RANK": "Rolling rank of close",
    "RSV": "Stochastic RSV: (close - rolling min) / (rolling max - rolling min)",
    "IMAX": "Normalized argmax of high within window",
    "IMIN": "Normalized argmin of low within window",
    "IMXD": "Normalized distance between argmax(high) and argmin(low)",
    "CORR": "Correlation between close and log(volume+1)",
    "CORD": "Correlation between close return and log(volume return+1)",
    "CNTP": "Fraction of up days in window",
    "CNTN": "Fraction of down days in window",
    "CNTD": "Net fraction of up vs down days in window",
    "SUMP": "Positive return sum ratio (RSI-like, price)",
    "SUMN": "Negative return sum ratio (RSI-like, price)",
    "SUMD": "Net return sum ratio (RSI-like, price)",
    "VMA": "Volume moving average / current volume",
    "VSTD": "Volume rolling std / current volume",
    "WVMA": "Weighted volume-return moving average coefficient of variation",
    "VSUMP": "Positive volume change sum ratio (RSI-like, volume)",
    "VSUMN": "Negative volume change sum ratio (RSI-like, volume)",
    "VSUMD": "Net volume change sum ratio (RSI-like, volume)",
}

# ---------------------------------------------------------------------------
# Rolling factor expression templates (d is the window)
# ---------------------------------------------------------------------------


def _rolling_expression(name: str, d: int) -> str:
    """Return the expression string for a rolling factor type at window d."""
    templates = {
        "ROC": f"delay(close, {d}) / close",
        "MA": f"ts_mean(close, {d}) / close",
        "STD": f"stddev(close, {d}) / close",
        "BETA": f"ts_slope(close, {d}) / close",
        "RSQR": f"ts_rsquare(close, {d})",
        "RESI": f"ts_residual(close, {d}) / close",
        "MAX": f"ts_max(high, {d}) / close",
        "MIN": f"ts_min(low, {d}) / close",
        "QTLU": f"ts_percentile(close, {d}, 0.8) / close",
        "QTLD": f"ts_percentile(close, {d}, 0.2) / close",
        "RANK": f"ts_rank(close, {d})",
        "RSV": (
            f"(close - ts_min(low, {d})) / "
            f"(ts_max(high, {d}) - ts_min(low, {d}) + 1e-12)"
        ),
        "IMAX": f"ts_argmax(high, {d}) / {d}",
        "IMIN": f"ts_argmin(low, {d}) / {d}",
        "IMXD": f"(ts_argmax(high, {d}) - ts_argmin(low, {d})) / {d}",
        "CORR": f"correlation(close, log(volume + 1), {d})",
        "CORD": (
            f"correlation(close / delay(close, 1), "
            f"log(volume / delay(volume, 1) + 1), {d})"
        ),
        "CNTP": f"ts_mean(if_else(close > delay(close, 1), 1, 0), {d})",
        "CNTN": f"ts_mean(if_else(close < delay(close, 1), 1, 0), {d})",
        "CNTD": (
            f"ts_mean(if_else(close > delay(close, 1), 1, 0), {d}) - "
            f"ts_mean(if_else(close < delay(close, 1), 1, 0), {d})"
        ),
        "SUMP": (
            f"ts_sum(max(close - delay(close, 1), 0), {d}) / "
            f"(ts_sum(abs(close - delay(close, 1)), {d}) + 1e-12)"
        ),
        "SUMN": (
            f"ts_sum(max(delay(close, 1) - close, 0), {d}) / "
            f"(ts_sum(abs(close - delay(close, 1)), {d}) + 1e-12)"
        ),
        "SUMD": (
            f"(ts_sum(max(close - delay(close, 1), 0), {d}) - "
            f"ts_sum(max(delay(close, 1) - close, 0), {d})) / "
            f"(ts_sum(abs(close - delay(close, 1)), {d}) + 1e-12)"
        ),
        "VMA": f"ts_mean(volume, {d}) / (volume + 1e-12)",
        "VSTD": f"stddev(volume, {d}) / (volume + 1e-12)",
        "WVMA": (
            f"stddev(abs(close / delay(close, 1) - 1) * volume, {d}) / "
            f"(ts_mean(abs(close / delay(close, 1) - 1) * volume, {d}) + 1e-12)"
        ),
        "VSUMP": (
            f"ts_sum(max(volume - delay(volume, 1), 0), {d}) / "
            f"(ts_sum(abs(volume - delay(volume, 1)), {d}) + 1e-12)"
        ),
        "VSUMN": (
            f"ts_sum(max(delay(volume, 1) - volume, 0), {d}) / "
            f"(ts_sum(abs(volume - delay(volume, 1)), {d}) + 1e-12)"
        ),
        "VSUMD": (
            f"(ts_sum(max(volume - delay(volume, 1), 0), {d}) - "
            f"ts_sum(max(delay(volume, 1) - volume, 0), {d})) / "
            f"(ts_sum(abs(volume - delay(volume, 1)), {d}) + 1e-12)"
        ),
    }
    return templates[name]


# ---------------------------------------------------------------------------
# Build ALPHA158_FACTORS dict
# ---------------------------------------------------------------------------

def _build_factors() -> dict[str, dict[str, str]]:
    """Programmatically build all 157 Alpha158 factors."""
    factors: dict[str, dict[str, str]] = {}

    # --- Kbar factors (9) ---
    kbar_defs = {
        "KMID": {
            "expression": "(close - open) / open",
            "description": "Kbar midpoint: (close - open) / open",
        },
        "KLEN": {
            "expression": "(high - low) / open",
            "description": "Kbar length: (high - low) / open",
        },
        "KMID2": {
            "expression": "(close - open) / (high - low + 1e-12)",
            "description": "Kbar midpoint ratio: (close - open) / (high - low)",
        },
        "KUP": {
            "expression": "(high - max(open, close)) / open",
            "description": "Kbar upper shadow: (high - greater(open, close)) / open",
        },
        "KUP2": {
            "expression": "(high - max(open, close)) / (high - low + 1e-12)",
            "description": "Kbar upper shadow ratio: upper shadow / bar range",
        },
        "KLOW": {
            "expression": "(min(open, close) - low) / open",
            "description": "Kbar lower shadow: (lesser(open, close) - low) / open",
        },
        "KLOW2": {
            "expression": "(min(open, close) - low) / (high - low + 1e-12)",
            "description": "Kbar lower shadow ratio: lower shadow / bar range",
        },
        "KSFT": {
            "expression": "(2 * close - high - low) / open",
            "description": "Kbar shift: (2*close - high - low) / open",
        },
        "KSFT2": {
            "expression": "(2 * close - high - low) / (high - low + 1e-12)",
            "description": "Kbar shift ratio: (2*close - high - low) / bar range",
        },
    }
    for name, defn in kbar_defs.items():
        factors[name] = {
            "expression": defn["expression"],
            "description": defn["description"],
            "category": "kbar",
        }

    # --- Price factors (3, skip VWAP0) ---
    price_defs = {
        "OPEN0": {
            "expression": "open / close",
            "description": "Normalized open: open / close",
        },
        "HIGH0": {
            "expression": "high / close",
            "description": "Normalized high: high / close",
        },
        "LOW0": {
            "expression": "low / close",
            "description": "Normalized low: low / close",
        },
    }
    for name, defn in price_defs.items():
        factors[name] = {
            "expression": defn["expression"],
            "description": defn["description"],
            "category": "price",
        }

    # --- Rolling factors (29 types x 5 windows = 145) ---
    rolling_types = [
        "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
        "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
        "IMAX", "IMIN", "IMXD",
        "CORR", "CORD",
        "CNTP", "CNTN", "CNTD",
        "SUMP", "SUMN", "SUMD",
        "VMA", "VSTD", "WVMA",
        "VSUMP", "VSUMN", "VSUMD",
    ]
    for rtype in rolling_types:
        for d in WINDOWS:
            factor_name = f"{rtype}{d}"
            factors[factor_name] = {
                "expression": _rolling_expression(rtype, d),
                "description": f"{_ROLLING_DESCRIPTION[rtype]} (window={d})",
                "category": _ROLLING_CATEGORY[rtype],
            }

    return factors


ALPHA158_FACTORS: dict[str, dict[str, str]] = _build_factors()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_alpha158_expression(name: str) -> str:
    """Get the expression for an Alpha158 factor.

    Parameters
    ----------
    name : str
        Factor name, e.g. ``"KMID"``, ``"ROC5"``, ``"CORR60"``.

    Returns
    -------
    str
        The expression string in nautilus_quants syntax.

    Raises
    ------
    ValueError
        If the factor name is not found.
    """
    if name not in ALPHA158_FACTORS:
        raise ValueError(
            f"Unknown Alpha158 factor: {name}. "
            f"Use list_alpha158_factors() to see available factors."
        )
    return ALPHA158_FACTORS[name]["expression"]


def list_alpha158_factors() -> list[str]:
    """List all available Alpha158 factor names, sorted alphabetically.

    Returns
    -------
    list[str]
        Sorted list of factor names.
    """
    return sorted(ALPHA158_FACTORS.keys())


# ---------------------------------------------------------------------------
# Module-level report
# ---------------------------------------------------------------------------

print(f"Alpha158: {len(ALPHA158_FACTORS)} factors loaded")
