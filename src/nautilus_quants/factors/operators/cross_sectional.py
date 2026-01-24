# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Cross-Sectional Operators.

Operators that work across multiple instruments at a single point in time,
computing rankings, z-scores, and other cross-sectional transformations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nautilus_quants.factors.operators.base import (
    CrossSectionalOperator,
    register_operator,
)


@register_operator
class CsRank(CrossSectionalOperator):
    """Cross-sectional percentile rank (0 to 1)."""
    
    name = "cs_rank"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute percentile ranks across instruments.
        
        Args:
            values: Dict of {instrument_id: value}
            
        Returns:
            Dict of {instrument_id: rank} where rank is in [0, 1]
        """
        if not values:
            return {}
        
        # Filter out NaN values
        valid_items = [(k, v) for k, v in values.items() if not np.isnan(v)]
        if not valid_items:
            return {k: float('nan') for k in values}
        
        # Sort by value
        sorted_items = sorted(valid_items, key=lambda x: x[1])
        n = len(sorted_items)
        
        # Assign ranks (0 to 1)
        result: dict[str, float] = {}
        for i, (instrument_id, _) in enumerate(sorted_items):
            result[instrument_id] = i / (n - 1) if n > 1 else 0.5
        
        # Set NaN for instruments with NaN input
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
        
        return result


@register_operator
class CsZscore(CrossSectionalOperator):
    """Cross-sectional z-score normalization."""
    
    name = "cs_zscore"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute z-scores across instruments.
        
        Args:
            values: Dict of {instrument_id: value}
            
        Returns:
            Dict of {instrument_id: zscore}
        """
        if not values:
            return {}
        
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if len(valid_values) < 2:
            return {k: float('nan') for k in values}
        
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        
        if std == 0:
            return {k: 0.0 if not np.isnan(v) else float('nan') 
                    for k, v in values.items()}
        
        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float((v - mean) / std)
        
        return result


@register_operator
class CsScale(CrossSectionalOperator):
    """Cross-sectional scaling to sum to 1 (for portfolio weights)."""
    
    name = "cs_scale"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Scale values so absolute values sum to 1.
        
        Args:
            values: Dict of {instrument_id: value}
            
        Returns:
            Dict of {instrument_id: scaled_value}
        """
        if not values:
            return {}
        
        # Filter out NaN values
        valid_items = [(k, v) for k, v in values.items() if not np.isnan(v)]
        if not valid_items:
            return {k: float('nan') for k in values}
        
        # Sum of absolute values
        total = sum(abs(v) for _, v in valid_items)
        
        if total == 0:
            # All zeros - distribute equally
            n = len(valid_items)
            return {k: 1.0 / n if not np.isnan(v) else float('nan') 
                    for k, v in values.items()}
        
        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float(v / total)
        
        return result


@register_operator
class CsDemean(CrossSectionalOperator):
    """Cross-sectional demeaning (subtract cross-sectional mean)."""
    
    name = "cs_demean"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Subtract cross-sectional mean from each value.
        
        Args:
            values: Dict of {instrument_id: value}
            
        Returns:
            Dict of {instrument_id: demeaned_value}
        """
        if not values:
            return {}
        
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {k: float('nan') for k in values}
        
        mean = np.mean(valid_values)
        
        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float(v - mean)
        
        return result


@register_operator  
class CsMax(CrossSectionalOperator):
    """Cross-sectional maximum value."""
    
    name = "cs_max"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return the maximum value across all instruments."""
        if not values:
            return {}
        
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {k: float('nan') for k in values}
        
        max_val = max(valid_values)
        return {k: max_val for k in values}


@register_operator
class CsMin(CrossSectionalOperator):
    """Cross-sectional minimum value."""
    
    name = "cs_min"
    min_args = 1
    max_args = 1
    
    def compute(
        self, 
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return the minimum value across all instruments."""
        if not values:
            return {}
        
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {k: float('nan') for k in values}
        
        min_val = min(valid_values)
        return {k: min_val for k in values}


# Convenience function wrappers
def cs_rank(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional rank."""
    return CsRank().compute(values)


def cs_zscore(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score."""
    return CsZscore().compute(values)


def cs_scale(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional scale to sum to 1."""
    return CsScale().compute(values)


def cs_demean(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional demean."""
    return CsDemean().compute(values)


def cs_max(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional max."""
    return CsMax().compute(values)


def cs_min(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional min."""
    return CsMin().compute(values)


# Export all function wrappers
CROSS_SECTIONAL_OPERATORS = {
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "cs_scale": cs_scale,
    "cs_demean": cs_demean,
    "cs_max": cs_max,
    "cs_min": cs_min,
}
