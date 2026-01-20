"""
Nautilus Quants - Quantitative Trading Framework

基于 Nautilus Trader 的量化交易框架
充分利用 Nautilus 原生组件，最小化自定义代码
"""

__version__ = "0.1.0"

from nautilus_quants.core.data_types import UniverseUpdate
from nautilus_quants.actors.screening import ScreeningActor, ScreeningActorConfig
from nautilus_quants.strategies.breakout import BreakoutStrategy, BreakoutStrategyConfig
from nautilus_quants.indicators.breakout import BreakoutIndicator

__all__ = [
    "UniverseUpdate",
    "ScreeningActor",
    "ScreeningActorConfig",
    "BreakoutStrategy",
    "BreakoutStrategyConfig",
    "BreakoutIndicator",
]
