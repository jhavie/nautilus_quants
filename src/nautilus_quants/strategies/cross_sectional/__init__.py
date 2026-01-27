# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Cross-Sectional Factor Strategy Module.

Multi-factor cross-sectional selection strategy for cryptocurrency futures.
Based on: https://www.fmz.com/digest-topic/9647
"""

from nautilus_quants.strategies.cross_sectional.strategy import (
    CrossSectionalFactorStrategy,
    CrossSectionalFactorStrategyConfig,
)

__all__ = [
    "CrossSectionalFactorStrategy",
    "CrossSectionalFactorStrategyConfig",
]
