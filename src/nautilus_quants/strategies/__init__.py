"""Strategies module - 交易策略"""

from nautilus_quants.strategies.breakout import (
    PriceVolumeBreakoutStrategy,
    PriceVolumeBreakoutStrategyConfig,
)
from nautilus_quants.strategies.fmz import (
    FMZFactorStrategy,
    FMZFactorStrategyConfig,
)

# Strategy registry for backtest module
# Maps strategy name to (StrategyClass, ConfigClass) tuple
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "breakout": (PriceVolumeBreakoutStrategy, PriceVolumeBreakoutStrategyConfig),
    "fmz": (FMZFactorStrategy, FMZFactorStrategyConfig),
}

__all__ = [
    "PriceVolumeBreakoutStrategy",
    "PriceVolumeBreakoutStrategyConfig",
    "FMZFactorStrategy",
    "FMZFactorStrategyConfig",
    "STRATEGY_REGISTRY",
]
