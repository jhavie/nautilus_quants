"""
量价突破指标

检测价格和成交量同时突破近期高点
"""

from collections import deque
from typing import Deque

from nautilus_trader.indicators.base import Indicator
from nautilus_trader.model.data import Bar


class BreakoutIndicator(Indicator):
    """
    量价突破指标

    信号条件:
    - close > max(high, period)  -- 价格突破
    - volume > max(volume, period) -- 成交量突破

    Parameters
    ----------
    period : int
        回看周期，默认 60

    Attributes
    ----------
    price_breakout : bool
        价格是否突破
    volume_breakout : bool
        成交量是否突破
    signal : bool
        量价同时突破
    """

    def __init__(self, period: int = 60):
        super().__init__(params=[period])
        self.period = period

        # 使用 deque 限制长度，自动剔除旧数据
        self._highs: Deque[float] = deque(maxlen=period)
        self._volumes: Deque[float] = deque(maxlen=period)

        # 信号状态
        self.price_breakout: bool = False
        self.volume_breakout: bool = False
        self.signal: bool = False

    def handle_bar(self, bar: Bar) -> None:
        """
        处理新的 K 线数据

        Parameters
        ----------
        bar : Bar
            K 线数据
        """
        current_high = float(bar.high)
        current_close = float(bar.close)
        current_volume = float(bar.volume)

        # 检查是否有足够历史数据
        if len(self._highs) >= self.period:
            # 价格突破: 收盘价 > 历史最高价
            highest_high = max(self._highs)
            self.price_breakout = current_close > highest_high

            # 成交量突破: 当前成交量 > 历史最高成交量
            highest_volume = max(self._volumes)
            self.volume_breakout = current_volume > highest_volume

            # 综合信号
            self.signal = self.price_breakout and self.volume_breakout

            # 标记已初始化
            if not self.initialized:
                self._set_initialized(True)

        # 更新历史数据 (在判断后更新，避免自己和自己比)
        self._highs.append(current_high)
        self._volumes.append(current_volume)

    def reset(self) -> None:
        """重置指标状态"""
        self._highs.clear()
        self._volumes.clear()
        self.price_breakout = False
        self.volume_breakout = False
        self.signal = False
        self._set_initialized(False)
