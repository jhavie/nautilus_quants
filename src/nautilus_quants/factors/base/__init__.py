# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Factor base classes module.

Provides abstract base classes and common implementations for factors.
"""

from nautilus_quants.factors.base.cross_sectional_factor import (
    CrossSectionalFactor,
    RankFactor,
    ZScoreFactor,
)
from nautilus_quants.factors.base.factor import (
    ExpressionFactor,
    Factor,
    FactorMetadata,
)
from nautilus_quants.factors.base.time_series_factor import (
    MeanReversionFactor,
    MomentumFactor,
    TimeSeriesFactor,
    VolatilityFactor,
)

__all__ = [
    # Base classes
    "Factor",
    "FactorMetadata",
    "ExpressionFactor",
    "TimeSeriesFactor",
    "CrossSectionalFactor",
    # Time-series implementations
    "MomentumFactor",
    "VolatilityFactor",
    "MeanReversionFactor",
    # Cross-sectional implementations
    "RankFactor",
    "ZScoreFactor",
]
