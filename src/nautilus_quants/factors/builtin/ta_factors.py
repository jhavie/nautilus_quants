# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Technical Analysis Built-in Factors — classic TA indicator expressions.

Expressions use ONLY OHLCV fields: open, high, low, close, volume.
No vwap, taker_buy_volume, or amount fields are used.

Categories:
    - trend:      Moving averages, MACD
    - momentum:   RSI, ROC, Stochastic, Williams %R, CCI
    - volatility: ATR, Bollinger Bands, Realized Volatility
    - volume:     OBV, Volume Ratio, PVT, Correlations
"""

from __future__ import annotations


# Technical Analysis factor expressions
# Classic TA indicators expressed in the nautilus_quants expression engine

TA_FACTORS = {
    # -------------------------------------------------------------------------
    # Trend (趋势)
    # -------------------------------------------------------------------------
    "sma_5": {
        "expression": "ts_mean(close, 5) / close",
        "description": "5-period simple moving average normalized by close",
        "category": "trend",
    },
    "sma_10": {
        "expression": "ts_mean(close, 10) / close",
        "description": "10-period simple moving average normalized by close",
        "category": "trend",
    },
    "sma_20": {
        "expression": "ts_mean(close, 20) / close",
        "description": "20-period simple moving average normalized by close",
        "category": "trend",
    },
    "sma_60": {
        "expression": "ts_mean(close, 60) / close",
        "description": "60-period simple moving average normalized by close",
        "category": "trend",
    },
    "ema_12": {
        "expression": "ema(close, 12) / close",
        "description": "12-period exponential moving average normalized by close",
        "category": "trend",
    },
    "ema_26": {
        "expression": "ema(close, 26) / close",
        "description": "26-period exponential moving average normalized by close",
        "category": "trend",
    },
    "macd_line": {
        "expression": "(ema(close, 12) - ema(close, 26)) / close",
        "description": "MACD line (EMA12 - EMA26) normalized by close",
        "category": "trend",
    },
    "macd_hist": {
        "expression": "ema(close, 12) - ema(close, 26)",
        "description": "Raw MACD histogram (EMA12 - EMA26, unnormalized)",
        "category": "trend",
    },
    # -------------------------------------------------------------------------
    # Momentum (动量)
    # -------------------------------------------------------------------------
    "rsi_14": {
        "expression": (
            "ts_sum(if_else(delta(close, 1) > 0, delta(close, 1), 0), 14)"
            " / (ts_sum(abs(delta(close, 1)), 14) + 1e-12)"
        ),
        "description": "14-period Relative Strength Index (0-1 scale)",
        "category": "momentum",
    },
    "roc_5": {
        "expression": "delta(close, 5) / delay(close, 5)",
        "description": "5-period Rate of Change",
        "category": "momentum",
    },
    "roc_10": {
        "expression": "delta(close, 10) / delay(close, 10)",
        "description": "10-period Rate of Change",
        "category": "momentum",
    },
    "roc_20": {
        "expression": "delta(close, 20) / delay(close, 20)",
        "description": "20-period Rate of Change",
        "category": "momentum",
    },
    "williams_r": {
        "expression": (
            "(ts_max(high, 14) - close)"
            " / (ts_max(high, 14) - ts_min(low, 14) + 1e-12)"
        ),
        "description": "14-period Williams %R (0-1 scale, inverted)",
        "category": "momentum",
    },
    "cci_20": {
        "expression": (
            "(close - ts_mean(close, 20))"
            " / (0.015 * ts_mean(abs(close - ts_mean(close, 20)), 20) + 1e-12)"
        ),
        "description": "20-period Commodity Channel Index",
        "category": "momentum",
    },
    "stochastic_k": {
        "expression": (
            "(close - ts_min(low, 14))"
            " / (ts_max(high, 14) - ts_min(low, 14) + 1e-12)"
        ),
        "description": "14-period Stochastic %K (Fast)",
        "category": "momentum",
    },
    "stochastic_d": {
        "expression": (
            "ts_mean((close - ts_min(low, 14))"
            " / (ts_max(high, 14) - ts_min(low, 14) + 1e-12), 3)"
        ),
        "description": "3-period SMA of Stochastic %K (Slow %D)",
        "category": "momentum",
    },
    "momentum_10": {
        "expression": "close / delay(close, 10)",
        "description": "10-period momentum (price ratio)",
        "category": "momentum",
    },
    "momentum_20": {
        "expression": "close / delay(close, 20)",
        "description": "20-period momentum (price ratio)",
        "category": "momentum",
    },
    # -------------------------------------------------------------------------
    # Volatility (波动率)
    # -------------------------------------------------------------------------
    "atr_14": {
        "expression": (
            "ts_mean(max(high - low,"
            " max(abs(high - delay(close, 1)), abs(low - delay(close, 1)))), 14)"
            " / close"
        ),
        "description": "14-period Average True Range normalized by close",
        "category": "volatility",
    },
    "bb_upper": {
        "expression": "(ts_mean(close, 20) + 2 * stddev(close, 20)) / close",
        "description": "Bollinger Band upper (SMA20 + 2*std) normalized by close",
        "category": "volatility",
    },
    "bb_lower": {
        "expression": "(ts_mean(close, 20) - 2 * stddev(close, 20)) / close",
        "description": "Bollinger Band lower (SMA20 - 2*std) normalized by close",
        "category": "volatility",
    },
    "bb_width": {
        "expression": "4 * stddev(close, 20) / (ts_mean(close, 20) + 1e-12)",
        "description": "Bollinger Band width (4*std / SMA20)",
        "category": "volatility",
    },
    "realized_vol_10": {
        "expression": "stddev(delta(close, 1) / delay(close, 1), 10)",
        "description": "10-period realized volatility of returns",
        "category": "volatility",
    },
    "realized_vol_20": {
        "expression": "stddev(delta(close, 1) / delay(close, 1), 20)",
        "description": "20-period realized volatility of returns",
        "category": "volatility",
    },
    # -------------------------------------------------------------------------
    # Volume (成交量)
    # -------------------------------------------------------------------------
    "obv_change": {
        "expression": (
            "if_else(close > delay(close, 1), volume,"
            " if_else(close < delay(close, 1), -1 * volume, 0))"
        ),
        "description": "On-Balance Volume single-period change",
        "category": "volume",
    },
    "volume_ratio": {
        "expression": "volume / (ts_mean(volume, 20) + 1e-12)",
        "description": "Volume ratio vs 20-period average",
        "category": "volume",
    },
    "volume_roc": {
        "expression": "delta(volume, 5) / (delay(volume, 5) + 1e-12)",
        "description": "5-period volume Rate of Change",
        "category": "volume",
    },
    "pvt_change": {
        "expression": "volume * delta(close, 1) / (delay(close, 1) + 1e-12)",
        "description": "Price Volume Trend single-period change",
        "category": "volume",
    },
    "volume_price_corr": {
        "expression": "correlation(close, volume, 20)",
        "description": "20-period close-volume correlation",
        "category": "volume",
    },
}


def get_ta_expression(name: str) -> str:
    """Get the expression for a TA factor."""
    if name not in TA_FACTORS:
        raise ValueError(f"Unknown TA factor: {name}")
    return TA_FACTORS[name]["expression"]


def list_ta_factors() -> list[str]:
    """List all available TA factors."""
    return sorted(TA_FACTORS.keys())
