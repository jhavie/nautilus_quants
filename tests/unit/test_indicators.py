"""
BreakoutIndicator 单元测试
"""

import pytest
from decimal import Decimal
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from nautilus_quants.indicators.breakout import BreakoutIndicator


def create_bar(
    high: float,
    low: float,
    close: float,
    volume: float,
    ts_event: int = 0,
) -> Bar:
    """创建测试用 Bar"""
    bar_type = BarType.from_str("BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL")
    return Bar(
        bar_type=bar_type,
        open=Price.from_str(str(close)),
        high=Price.from_str(str(high)),
        low=Price.from_str(str(low)),
        close=Price.from_str(str(close)),
        volume=Quantity.from_str(str(volume)),
        ts_event=ts_event,
        ts_init=ts_event,
    )


class TestBreakoutIndicator:
    """BreakoutIndicator 测试用例"""

    def test_initialization(self):
        """测试指标初始化"""
        indicator = BreakoutIndicator(period=60)

        assert indicator.period == 60
        assert indicator.price_breakout is False
        assert indicator.volume_breakout is False
        assert indicator.signal is False
        assert indicator.initialized is False

    def test_not_initialized_until_enough_data(self):
        """测试数据不足时不初始化"""
        indicator = BreakoutIndicator(period=5)

        # 只输入 4 根 K 线
        for i in range(4):
            bar = create_bar(
                high=100 + i,
                low=95 + i,
                close=98 + i,
                volume=1000 + i * 100,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        assert indicator.initialized is False

    def test_initialized_after_enough_data(self):
        """测试数据足够后初始化"""
        indicator = BreakoutIndicator(period=5)

        # 输入 5 根 K 线
        for i in range(5):
            bar = create_bar(
                high=100 + i,
                low=95 + i,
                close=98 + i,
                volume=1000 + i * 100,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        assert indicator.initialized is True

    def test_price_breakout_signal(self):
        """测试价格突破信号"""
        indicator = BreakoutIndicator(period=5)

        # 建立历史数据 (high: 100-104)
        for i in range(5):
            bar = create_bar(
                high=100 + i,
                low=95,
                close=98,
                volume=1000,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        # 价格突破 (close > max(high) = 104)
        breakout_bar = create_bar(
            high=110,
            low=100,
            close=105,  # > 104
            volume=1000,
            ts_event=5,
        )
        indicator.handle_bar(breakout_bar)

        assert indicator.price_breakout is True

    def test_volume_breakout_signal(self):
        """测试成交量突破信号"""
        indicator = BreakoutIndicator(period=5)

        # 建立历史数据 (volume: 1000-1400)
        for i in range(5):
            bar = create_bar(
                high=100,
                low=95,
                close=98,
                volume=1000 + i * 100,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        # 成交量突破 (volume > max = 1400)
        breakout_bar = create_bar(
            high=100,
            low=95,
            close=98,
            volume=2000,  # > 1400
            ts_event=5,
        )
        indicator.handle_bar(breakout_bar)

        assert indicator.volume_breakout is True

    def test_combined_signal(self):
        """测试量价同时突破"""
        indicator = BreakoutIndicator(period=5)

        # 建立历史数据
        for i in range(5):
            bar = create_bar(
                high=100 + i,
                low=95,
                close=98,
                volume=1000 + i * 100,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        # 量价同时突破
        breakout_bar = create_bar(
            high=110,
            low=100,
            close=105,  # 价格突破
            volume=2000,  # 成交量突破
            ts_event=5,
        )
        indicator.handle_bar(breakout_bar)

        assert indicator.price_breakout is True
        assert indicator.volume_breakout is True
        assert indicator.signal is True

    def test_reset(self):
        """测试重置功能"""
        indicator = BreakoutIndicator(period=5)

        # 输入数据
        for i in range(5):
            bar = create_bar(
                high=100 + i,
                low=95,
                close=98,
                volume=1000,
                ts_event=i,
            )
            indicator.handle_bar(bar)

        assert indicator.initialized is True

        # 重置
        indicator.reset()

        assert indicator.initialized is False
        assert indicator.price_breakout is False
        assert indicator.volume_breakout is False
        assert indicator.signal is False
