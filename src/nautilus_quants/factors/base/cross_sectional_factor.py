# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Cross-Sectional Factor Base Class.

Specialized factor class for cross-sectional computations across multiple instruments.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from nautilus_quants.factors.base.factor import Factor

if TYPE_CHECKING:
    from nautilus_quants.factors.types import FactorInput


class CrossSectionalFactor(Factor):
    """
    Base class for cross-sectional factors.
    
    Cross-sectional factors operate across multiple instruments at a single
    point in time, computing relative rankings, z-scores, etc.
    
    Unlike time-series factors, cross-sectional factors need values from
    all instruments before they can compute results.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            warmup_period=1,
        )
    
    def compute(self, data: FactorInput) -> float:
        """
        Compute factor for single instrument.
        
        Note: Cross-sectional factors typically need to be computed
        via compute_cross_section() with data from all instruments.
        This method returns NaN - use the engine for proper computation.
        """
        return float('nan')
    
    @abstractmethod
    def compute_cross_section(
        self,
        values: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute factor values across all instruments.
        
        Args:
            values: Dict of {instrument_id: raw_value} for all instruments
            
        Returns:
            Dict of {instrument_id: factor_value}
        """
        pass


class RankFactor(CrossSectionalFactor):
    """Cross-sectional rank factor (0 to 1)."""
    
    def __init__(self, source_factor: str, name: str | None = None) -> None:
        super().__init__(
            name=name or f"rank_{source_factor}",
            description=f"Cross-sectional rank of {source_factor}",
        )
        self.source_factor = source_factor
    
    def compute_cross_section(
        self,
        values: dict[str, float],
    ) -> dict[str, float]:
        """Compute percentile ranks across instruments."""
        if not values:
            return {}
        
        # Filter out NaN values
        valid_items = [(k, v) for k, v in values.items() if not np.isnan(v)]
        if not valid_items:
            return {k: float('nan') for k in values}
        
        # Sort by value
        sorted_items = sorted(valid_items, key=lambda x: x[1])
        n = len(sorted_items)
        
        # Assign ranks
        result: dict[str, float] = {}
        for i, (instrument_id, _) in enumerate(sorted_items):
            # Rank from 0 to 1
            result[instrument_id] = i / (n - 1) if n > 1 else 0.5
        
        # Set NaN for instruments with NaN input
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
        
        return result


class ZScoreFactor(CrossSectionalFactor):
    """Cross-sectional z-score factor."""
    
    def __init__(self, source_factor: str, name: str | None = None) -> None:
        super().__init__(
            name=name or f"zscore_{source_factor}",
            description=f"Cross-sectional z-score of {source_factor}",
        )
        self.source_factor = source_factor
    
    def compute_cross_section(
        self,
        values: dict[str, float],
    ) -> dict[str, float]:
        """Compute z-scores across instruments."""
        if not values:
            return {}
        
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if len(valid_values) < 2:
            return {k: float('nan') for k in values}
        
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        
        if std == 0:
            return {k: 0.0 for k in values}
        
        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float((v - mean) / std)
        
        return result
