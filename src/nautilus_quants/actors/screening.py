"""
选币池筛选 Actor

根据成交量排名筛选 Top N 币种，发布 UniverseUpdate 给策略
"""

from typing import Set

import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType, DataType

from nautilus_quants.core.data_types import UniverseUpdate


class ScreeningActorConfig(ActorConfig, frozen=True):
    """选币 Actor 配置"""

    # 选币参数
    top_n: int = 10
    volume_period: int = 7
    min_volume_usd: float = 5_000_000.0

    # 更新间隔
    update_interval: str = "1D"

    # 候选币种列表 (必须提供)
    candidate_symbols: tuple = ()


class ScreeningActor(Actor):
    """
    选币池筛选 Actor

    职责:
    1. 订阅所有候选币种的日线数据
    2. 计算 7 日累计成交量排名
    3. 筛选 Top N 发布 UniverseUpdate

    数据流:
    ScreeningActor --[publish_data(UniverseUpdate)]--> BreakoutStrategy
    """

    def __init__(self, config: ScreeningActorConfig):
        super().__init__(config)

        # 状态
        self._current_universe: Set[str] = set()
        self._volume_cache: dict[str, list[float]] = {}
        self._subscribed_bars: Set[BarType] = set()

    def on_start(self) -> None:
        """Actor 启动时订阅数据"""
        self.log.info(f"ScreeningActor starting with top_n={self.config.top_n}")

        if not self.config.candidate_symbols:
            self.log.error("No candidate_symbols provided!")
            return

        # 订阅所有候选币种的日线
        for symbol in self.config.candidate_symbols:
            bar_type = BarType.from_str(f"{symbol}.BINANCE-1-DAY-LAST-EXTERNAL")
            self.subscribe_bars(bar_type)
            self._subscribed_bars.add(bar_type)
            self._volume_cache[symbol] = []

        # 设置每日定时器
        self.clock.set_timer(
            name="daily_screening",
            interval=pd.Timedelta(self.config.update_interval),
            callback=self._on_screening_timer,
        )

        self.log.info(f"Subscribed to {len(self._subscribed_bars)} bar types")

    def on_bar(self, bar: Bar) -> None:
        """收集 K 线数据"""
        symbol = str(bar.bar_type.instrument_id.symbol)

        if symbol not in self._volume_cache:
            return

        # 计算成交额 (volume * close)
        volume_usd = float(bar.volume) * float(bar.close)
        self._volume_cache[symbol].append(volume_usd)

        # 只保留最近 N 天
        period = self.config.volume_period
        if len(self._volume_cache[symbol]) > period:
            self._volume_cache[symbol] = self._volume_cache[symbol][-period:]

    def _on_screening_timer(self, event) -> None:
        """定时筛选回调"""
        self.log.info("Running daily screening...")
        self._calculate_and_publish()

    def _calculate_and_publish(self) -> None:
        """计算排名并发布选币池更新"""
        period = self.config.volume_period

        # Step 1: 计算每个币种的累计成交量
        volume_scores: dict[str, float] = {}
        for symbol, volumes in self._volume_cache.items():
            if len(volumes) >= period:
                total_volume = sum(volumes[-period:])
                avg_daily = total_volume / period

                # 过滤低成交量
                if avg_daily >= self.config.min_volume_usd:
                    volume_scores[symbol] = total_volume

        if not volume_scores:
            self.log.warning("No symbols passed volume filter")
            return

        # Step 2: 排名并选取 Top N
        sorted_symbols = sorted(
            volume_scores.keys(),
            key=lambda s: volume_scores[s],
            reverse=True,
        )[: self.config.top_n]

        # Step 3: 计算排名分数 (1.0 = 第一名, 递减)
        rankings = {
            symbol: 1.0 - (i / self.config.top_n)
            for i, symbol in enumerate(sorted_symbols)
        }

        # Step 4: 发布 UniverseUpdate
        update = UniverseUpdate(
            symbols=sorted_symbols,
            rankings=rankings,
            ts_event=self.clock.timestamp_ns(),
            ts_init=self.clock.timestamp_ns(),
        )

        self.publish_data(DataType(UniverseUpdate), update)

        # 更新内部状态并记录变化
        old_universe = self._current_universe
        self._current_universe = set(sorted_symbols)

        added = self._current_universe - old_universe
        removed = old_universe - self._current_universe

        self.log.info(
            f"Universe updated: {len(sorted_symbols)} symbols "
            f"(+{len(added)} -{len(removed)})"
        )

    def on_stop(self) -> None:
        """Actor 停止时清理"""
        for bar_type in self._subscribed_bars:
            self.unsubscribe_bars(bar_type)
        self.log.info("ScreeningActor stopped")
