"""
自定义数据类型 - UniverseUpdate

用于 Actor -> Strategy 的选币池通信
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from nautilus_trader.core.data import Data


@dataclass
class UniverseUpdate(Data):
    """
    选币池更新数据

    由 ScreeningActor 发布，被 BreakoutStrategy 订阅

    Attributes
    ----------
    symbols : List[str]
        当前选币池内的币种列表 (按排名排序)
    rankings : Dict[str, float]
        每个币种的排名分数 {symbol: rank_score}
        分数范围 [0, 1]，1.0 表示第一名
    """

    symbols: List[str]
    rankings: Dict[str, float]

    def __repr__(self) -> str:
        top3 = self.symbols[:3] if len(self.symbols) >= 3 else self.symbols
        return f"UniverseUpdate(count={len(self.symbols)}, top3={top3})"
