# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""FMZ cross-sectional factor strategy.

This strategy implements the FMZ research approach for multi-factor
cross-sectional selection with separated concerns:
- Signal generation: Factor ranking and target computation
- Risk management: Position sizing and leverage control
- Position tracking: Entry metadata and rank history
- Order execution: Order sizing and submission

Based on: https://www.fmz.com/digest-topic/9647
"""

from nautilus_quants.strategies.fmz_factor.strategy import (
    FMZFactorStrategy,
    FMZFactorStrategyConfig,
)

__all__ = ["FMZFactorStrategy", "FMZFactorStrategyConfig"]
