# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Alpha101 Built-in Factors.

Implementation of WorldQuant Alpha101 factors as expressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_quants.factors.engine.factor_engine import FactorEngine


# Alpha101 factor expressions
# Reference: https://arxiv.org/abs/1601.00991

ALPHA101_FACTORS = {
    "alpha001": {
        "expression": "(close > delay(close, 1)) ? -1 * ts_rank(ts_argmax(close, 30), 5) : 1",
        "description": "Conditional rank of argmax",
        "category": "momentum",
    },
    "alpha002": {
        "expression": "-1 * correlation(ts_rank(delta(log(volume), 2), 6), ts_rank((close - open) / open, 6), 6)",
        "description": "Volume-price correlation",
        "category": "volume",
    },
    "alpha003": {
        "expression": "-1 * correlation(ts_rank(open, 10), ts_rank(volume, 10), 10)",
        "description": "Open-volume correlation",
        "category": "volume",
    },
    "alpha004": {
        "expression": "-1 * ts_rank(ts_rank(low, 9), 7)",
        "description": "Double rank of low",
        "category": "reversion",
    },
    "alpha005": {
        "expression": "ts_rank(open - ts_mean(close, 10), 2) * ts_rank(close - open, 1)",
        "description": "Open deviation rank",
        "category": "momentum",
    },
    "alpha006": {
        "expression": "-1 * correlation(open, volume, 10)",
        "description": "Open-volume correlation",
        "category": "volume",
    },
    "alpha007": {
        "expression": "(volume < delay(volume, 1)) ? -1 * ts_rank(abs(delta(close, 7)), 60) : -1",
        "description": "Volume conditional momentum",
        "category": "volume",
    },
    "alpha008": {
        "expression": "-1 * ts_rank(ts_sum(open, 5) * ts_sum(close, 5) - delay(ts_sum(open, 5) * ts_sum(close, 5), 10), 2)",
        "description": "Open-close sum momentum",
        "category": "momentum",
    },
    "alpha009": {
        "expression": "(ts_min(delta(close, 1), 5) > 0) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -1 * delta(close, 1))",
        "description": "Delta direction",
        "category": "momentum",
    },
    "alpha010": {
        "expression": "ts_rank((ts_min(delta(close, 1), 4) > 0) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -1 * delta(close, 1)), 6)",
        "description": "Ranked delta direction",
        "category": "momentum",
    },
}


def register_alpha101_factors(engine: FactorEngine, factors: list[str] | None = None) -> None:
    """
    Register Alpha101 factors with the engine.
    
    Args:
        engine: FactorEngine instance
        factors: List of factor names to register (default: all)
    """
    if factors is None:
        factors = list(ALPHA101_FACTORS.keys())
    
    for factor_name in factors:
        if factor_name not in ALPHA101_FACTORS:
            raise ValueError(f"Unknown Alpha101 factor: {factor_name}")
        
        factor_def = ALPHA101_FACTORS[factor_name]
        engine.register_expression_factor(
            name=factor_name,
            expression=factor_def["expression"],
            description=factor_def["description"],
        )


def get_alpha101_expression(name: str) -> str:
    """Get the expression for an Alpha101 factor."""
    if name not in ALPHA101_FACTORS:
        raise ValueError(f"Unknown Alpha101 factor: {name}")
    return ALPHA101_FACTORS[name]["expression"]


def list_alpha101_factors() -> list[str]:
    """List all available Alpha101 factors."""
    return sorted(ALPHA101_FACTORS.keys())
