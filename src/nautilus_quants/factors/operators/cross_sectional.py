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


# ============================================================================
# WorldQuant BRAIN-Compatible Operators (无前缀，与平台规范一致)
# ============================================================================


@register_operator
class CsNormalize(CrossSectionalOperator):
    """
    Cross-sectional normalize (WorldQuant BRAIN style).

    normalize(x, useStd=false, limit=0.0)

    Calculates the mean value of all valid alpha values for a certain date,
    then subtracts that mean from each element. If useStd=true, the operator
    calculates the standard deviation of the resulting values and divides
    each normalized element by it. If limit is not equal to 0.0, operator
    puts the limit of the resulting alpha values (between -limit to +limit).

    Example:
        x = [3, 5, 6, 2], mean = 4, std = 1.82
        normalize(x, useStd=false, limit=0.0) = [-1, 1, 2, -2]
        normalize(x, useStd=true, limit=0.0) = [-0.55, 0.55, 1.1, -1.1]
    """

    name = "normalize"
    min_args = 1
    max_args = 3

    def compute(
        self,
        values: dict[str, float],
        use_std: bool = False,
        limit: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, float]:
        if not values:
            return {}

        valid_values = [v for v in values.values() if not np.isnan(v)]
        if len(valid_values) < 2:
            return {k: float('nan') for k in values}

        mean = np.mean(valid_values)

        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float(v - mean)

        if use_std:
            normalized_values = [r for r in result.values() if not np.isnan(r)]
            std = np.std(normalized_values, ddof=0)  # WorldQuant uses population std
            if std > 0:
                result = {
                    k: v / std if not np.isnan(v) else float('nan')
                    for k, v in result.items()
                }

        if limit > 0:
            result = {
                k: max(-limit, min(limit, v)) if not np.isnan(v) else float('nan')
                for k, v in result.items()
            }

        return result


@register_operator
class CsWinsorize(CrossSectionalOperator):
    """
    Cross-sectional winsorize (WorldQuant BRAIN style).

    winsorize(x, std=4)

    Winsorizes x to make sure that all values in x are between the lower
    and upper limits, which are specified as multiple of std.

    Example:
        x = [2, 4, 5, 6, 3, 8, 10], std=1
        mean = 5.42, SD = 2.61
        lower = 5.42 - 1*2.61 = 2.81, upper = 5.42 + 1*2.61 = 8.03
        Output: [2.81, 4, 5, 6, 3, 8, 8.03]
    """

    name = "winsorize"
    min_args = 1
    max_args = 2

    def compute(
        self,
        values: dict[str, float],
        std_mult: float = 4.0,
        **kwargs: Any,
    ) -> dict[str, float]:
        if not values:
            return {}

        valid_values = [v for v in values.values() if not np.isnan(v)]
        if len(valid_values) < 3:
            return values.copy()

        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=0)  # WorldQuant uses population std

        if std == 0:
            return values.copy()

        lower = mean - std_mult * std
        upper = mean + std_mult * std

        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                result[k] = float(max(lower, min(upper, v)))

        return result


@register_operator
class CsScaleDown(CrossSectionalOperator):
    """
    Cross-sectional scale_down (WorldQuant BRAIN style).

    scale_down(x, constant=0)

    Scales all values in each day proportionately between 0 and 1 such that
    minimum value maps to 0 and maximum value maps to 1. Constant is the
    offset by which final result is subtracted.

    Example:
        x = [15, 7, 0, 20], max = 20, min = 0
        scale_down(x, constant=0) = [0.75, 0.35, 0, 1]
        scale_down(x, constant=1) = [-0.25, -0.65, -1, 0]
    """

    name = "scale_down"
    min_args = 1
    max_args = 2

    def compute(
        self,
        values: dict[str, float],
        constant: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, float]:
        if not values:
            return {}

        valid_values = [v for v in values.values() if not np.isnan(v)]
        if len(valid_values) < 2:
            return {k: float('nan') for k in values}

        min_val = min(valid_values)
        max_val = max(valid_values)
        range_val = max_val - min_val

        if range_val == 0:
            return {k: 0.5 - constant if not np.isnan(v) else float('nan')
                    for k, v in values.items()}

        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                scaled = (v - min_val) / range_val
                result[k] = float(scaled - constant)

        return result


@register_operator
class CsQuantile(CrossSectionalOperator):
    """
    Cross-sectional quantile transform (WorldQuant BRAIN style).

    quantile(x, driver="gaussian", sigma=1.0)

    Rank the raw vector, shift the ranked Alpha vector, apply distribution
    (gaussian, cauchy, uniform). If driver is uniform, it simply subtracts
    each Alpha value with the mean of all Alpha values.

    Steps:
    1. Rank input to [0, 1]
    2. Shift to [1/N, 1-1/N]
    3. Apply inverse CDF of specified distribution

    This operator may help reduce outliers.
    """

    name = "quantile"
    min_args = 1
    max_args = 3

    def compute(
        self,
        values: dict[str, float],
        driver: str = "gaussian",
        sigma: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, float]:
        from scipy import stats

        if not values:
            return {}

        valid_items = [(k, v) for k, v in values.items() if not np.isnan(v)]
        if len(valid_items) < 2:
            return {k: float('nan') for k in values}

        n = len(valid_items)

        # Step 1: Rank (0 to 1)
        sorted_items = sorted(valid_items, key=lambda x: x[1])
        ranks = {k: i / (n - 1) if n > 1 else 0.5 for i, (k, _) in enumerate(sorted_items)}

        # Step 2: Shift ranks to [1/N, 1-1/N]
        shifted = {k: 1/n + r * (1 - 2/n) for k, r in ranks.items()}

        # Step 3: Apply inverse CDF based on driver
        result: dict[str, float] = {}
        for k, v in values.items():
            if np.isnan(v):
                result[k] = float('nan')
            else:
                p = shifted[k]
                if driver == "gaussian":
                    result[k] = float(stats.norm.ppf(p) * sigma)
                elif driver == "cauchy":
                    result[k] = float(stats.cauchy.ppf(p) * sigma)
                elif driver == "uniform":
                    result[k] = float(p - 0.5)
                else:
                    result[k] = float(stats.norm.ppf(p) * sigma)

        return result


# Convenience function wrappers for new operators
def normalize(
    values: dict[str, float],
    use_std: bool = False,
    limit: float = 0.0,
) -> dict[str, float]:
    """Cross-sectional normalize (WorldQuant style)."""
    return CsNormalize().compute(values, use_std=use_std, limit=limit)


def winsorize(values: dict[str, float], std_mult: float = 4.0) -> dict[str, float]:
    """Cross-sectional winsorize (WorldQuant style)."""
    return CsWinsorize().compute(values, std_mult=std_mult)


def scale_down(values: dict[str, float], constant: float = 0.0) -> dict[str, float]:
    """Cross-sectional scale_down (WorldQuant style)."""
    return CsScaleDown().compute(values, constant=constant)


def quantile(
    values: dict[str, float],
    driver: str = "gaussian",
    sigma: float = 1.0,
) -> dict[str, float]:
    """Cross-sectional quantile transform (WorldQuant style)."""
    return CsQuantile().compute(values, driver=driver, sigma=sigma)


# ============================================================================
# Aliases for WorldQuant BRAIN compatibility (无前缀版本)
# ============================================================================

# Register aliases for existing operators
@register_operator
class CsRankAlias(CrossSectionalOperator):
    """Alias for cs_rank with WorldQuant-compatible name and rate parameter."""

    name = "rank"
    min_args = 1
    max_args = 2

    def compute(
        self,
        values: dict[str, float],
        rate: int = 2,
        **kwargs: Any,
    ) -> dict[str, float]:
        # rate parameter is for precision control, ignored in our implementation
        return CsRank().compute(values)


@register_operator
class CsScaleAlias(CrossSectionalOperator):
    """Alias for cs_scale with WorldQuant-compatible name."""

    name = "scale"
    min_args = 1
    max_args = 4

    def compute(
        self,
        values: dict[str, float],
        scale_factor: float = 1.0,
        longscale: float = 1.0,
        shortscale: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, float]:
        # WorldQuant scale has more parameters, but basic behavior is same
        return CsScale().compute(values)


@register_operator
class CsZscoreAlias(CrossSectionalOperator):
    """Alias for cs_zscore with WorldQuant-compatible name."""

    name = "zscore"
    min_args = 1
    max_args = 1

    def compute(
        self,
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        return CsZscore().compute(values)


@register_operator
class CsDemeanAlias(CrossSectionalOperator):
    """Alias for cs_demean with WorldQuant-compatible name (same as normalize without std)."""

    name = "demean"
    min_args = 1
    max_args = 1

    def compute(
        self,
        values: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        return CsDemean().compute(values)


# Alias function wrappers
def rank(values: dict[str, float], rate: int = 2) -> dict[str, float]:
    """Cross-sectional rank (WorldQuant style alias)."""
    return CsRankAlias().compute(values, rate=rate)


def scale(
    values: dict[str, float],
    scale_factor: float = 1.0,
    longscale: float = 1.0,
    shortscale: float = 1.0,
) -> dict[str, float]:
    """Cross-sectional scale (WorldQuant style alias)."""
    return CsScaleAlias().compute(values, scale_factor, longscale, shortscale)


def zscore(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional zscore (WorldQuant style alias)."""
    return CsZscoreAlias().compute(values)


def demean(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional demean (WorldQuant style alias)."""
    return CsDemeanAlias().compute(values)


# Export all function wrappers
CROSS_SECTIONAL_OPERATORS = {
    # Original cs_ prefixed operators
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "cs_scale": cs_scale,
    "cs_demean": cs_demean,
    "cs_max": cs_max,
    "cs_min": cs_min,
    # WorldQuant BRAIN-compatible operators (no prefix)
    "normalize": normalize,
    "winsorize": winsorize,
    "scale_down": scale_down,
    "quantile": quantile,
    # Aliases for backward compatibility
    "rank": rank,
    "scale": scale,
    "zscore": zscore,
    "demean": demean,
}
