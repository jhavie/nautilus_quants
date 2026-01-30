"""
Nautilus Quants - Quantitative Trading Framework

基于 Nautilus Trader 的量化交易框架
充分利用 Nautilus 原生组件，最小化自定义代码
"""

__version__ = "0.1.0"

from nautilus_quants.strategies.breakout import (
    PriceVolumeBreakoutStrategy,
    PriceVolumeBreakoutStrategyConfig,
)

__all__ = [
    "PriceVolumeBreakoutStrategy",
    "PriceVolumeBreakoutStrategyConfig",
]
