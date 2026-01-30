"""Strategies module - 交易策略"""

from nautilus_quants.strategies.breakout import (
    PriceVolumeBreakoutStrategy,
    PriceVolumeBreakoutStrategyConfig,
)

# Strategy registry for backtest module
# Maps strategy name to (StrategyClass, ConfigClass) tuple
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "breakout": (PriceVolumeBreakoutStrategy, PriceVolumeBreakoutStrategyConfig),
}

__all__ = [
    "PriceVolumeBreakoutStrategy",
    "PriceVolumeBreakoutStrategyConfig",
    "STRATEGY_REGISTRY",
]
