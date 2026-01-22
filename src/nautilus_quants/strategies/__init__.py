"""Strategies module - 交易策略"""

from nautilus_quants.strategies.breakout import BreakoutStrategy, BreakoutStrategyConfig

# Strategy registry for backtest module
# Maps strategy name to (StrategyClass, ConfigClass) tuple
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "breakout": (BreakoutStrategy, BreakoutStrategyConfig),
}

__all__ = ["BreakoutStrategy", "BreakoutStrategyConfig", "STRATEGY_REGISTRY"]
