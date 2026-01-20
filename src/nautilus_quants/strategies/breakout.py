"""
量价突破策略

基于 60 周期量价突破 + SMA200 趋势过滤的交易策略
"""

from decimal import Decimal
from typing import Optional, Set

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.model.data import Bar, BarType, DataType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from nautilus_quants.core.data_types import UniverseUpdate
from nautilus_quants.indicators.breakout import BreakoutIndicator


class BreakoutStrategyConfig(StrategyConfig, frozen=True):
    """量价突破策略配置"""

    # 信号参数
    breakout_period: int = 60
    sma_period: int = 200

    # 仓位管理
    position_size_pct: float = 0.10
    max_positions: int = 10

    # 风控 (Bracket Order)
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    # 交易所
    exchange: str = "BINANCE"

    # 时间周期 (用于构造 BarType)
    bar_spec: str = "1-HOUR-LAST-EXTERNAL"


class BreakoutStrategy(Strategy):
    """
    量价突破策略

    信号条件 (全部满足才做多):
    1. close > highest(high, 60)  -- 价格突破 60 周期最高
    2. volume > highest(volume, 60) -- 成交量突破 60 周期最高
    3. close > SMA(close, 200)    -- 价格在 SMA200 之上

    风控:
    - 使用 Bracket Order 原子提交入场单 + 止损单 + 止盈单
    - 固定 5% 止损，10% 止盈
    """

    def __init__(self, config: BreakoutStrategyConfig):
        super().__init__(config)

        # 当前选币池
        self._universe: Set[str] = set()
        self._rankings: dict[str, float] = {}

        # 已订阅的 BarType
        self._subscribed_bars: Set[BarType] = set()

        # 每个币种的指标实例
        self._sma_indicators: dict[str, SimpleMovingAverage] = {}
        self._breakout_indicators: dict[str, BreakoutIndicator] = {}

    def on_start(self) -> None:
        """策略启动"""
        self.log.info("BreakoutStrategy starting...")

        # 订阅选币池更新 (来自 ScreeningActor)
        self.subscribe_data(DataType(UniverseUpdate))

        self.log.info("Subscribed to UniverseUpdate")

    def on_data(self, data) -> None:
        """处理自定义数据"""
        if isinstance(data, UniverseUpdate):
            self._handle_universe_update(data)

    def _handle_universe_update(self, update: UniverseUpdate) -> None:
        """处理选币池更新"""
        new_universe = set(update.symbols)
        self._rankings = update.rankings

        # 计算变化
        to_subscribe = new_universe - self._universe
        to_unsubscribe = self._universe - new_universe

        # 取消订阅移出池的币种
        for symbol in to_unsubscribe:
            self._unsubscribe_symbol(symbol)
            # 如有持仓，平仓
            self._close_position_if_exists(symbol, "Removed from universe")

        # 订阅新加入的币种
        for symbol in to_subscribe:
            self._subscribe_symbol(symbol)

        self._universe = new_universe

        self.log.info(
            f"Universe synced: {len(self._universe)} symbols "
            f"(+{len(to_subscribe)} -{len(to_unsubscribe)})"
        )

    def _subscribe_symbol(self, symbol: str) -> None:
        """订阅币种并注册指标"""
        bar_type = BarType.from_str(
            f"{symbol}.{self.config.exchange}-{self.config.bar_spec}"
        )

        # 创建指标
        sma = SimpleMovingAverage(self.config.sma_period)
        breakout = BreakoutIndicator(self.config.breakout_period)

        # 注册指标到 BarType
        self.register_indicator_for_bars(bar_type, sma)
        self.register_indicator_for_bars(bar_type, breakout)

        # 保存引用
        self._sma_indicators[symbol] = sma
        self._breakout_indicators[symbol] = breakout

        # 订阅数据
        self.subscribe_bars(bar_type)
        self._subscribed_bars.add(bar_type)

    def _unsubscribe_symbol(self, symbol: str) -> None:
        """取消订阅币种"""
        bar_type = BarType.from_str(
            f"{symbol}.{self.config.exchange}-{self.config.bar_spec}"
        )

        if bar_type in self._subscribed_bars:
            self.unsubscribe_bars(bar_type)
            self._subscribed_bars.remove(bar_type)

        # 清理指标
        self._sma_indicators.pop(symbol, None)
        self._breakout_indicators.pop(symbol, None)

    def on_bar(self, bar: Bar) -> None:
        """处理 K 线 - 核心信号逻辑"""
        symbol = str(bar.bar_type.instrument_id.symbol)

        if symbol not in self._universe:
            return

        # 获取指标
        sma = self._sma_indicators.get(symbol)
        breakout = self._breakout_indicators.get(symbol)

        if sma is None or breakout is None:
            return

        # 检查指标是否已初始化
        if not sma.initialized or not breakout.initialized:
            return

        # 检查信号
        if self._check_long_signal(symbol, bar, sma, breakout):
            self._open_long(symbol, bar)

    def _check_long_signal(
        self,
        symbol: str,
        bar: Bar,
        sma: SimpleMovingAverage,
        breakout: BreakoutIndicator,
    ) -> bool:
        """检查做多信号"""
        current_close = float(bar.close)

        # 条件 1 & 2: 量价突破 (由 BreakoutIndicator 计算)
        if not breakout.signal:
            return False

        # 条件 3: 价格在 SMA200 之上
        if current_close <= sma.value:
            return False

        self.log.info(
            f"LONG signal on {symbol}: "
            f"close={current_close:.4f} > SMA200={sma.value:.4f}, "
            f"price_breakout={breakout.price_breakout}, "
            f"volume_breakout={breakout.volume_breakout}"
        )

        return True

    def _open_long(self, symbol: str, bar: Bar) -> None:
        """开多仓 - 使用 Bracket Order"""
        instrument_id = InstrumentId.from_str(f"{symbol}.{self.config.exchange}")
        instrument = self.cache.instrument(instrument_id)

        if instrument is None:
            self.log.warning(f"Instrument not found: {instrument_id}")
            return

        # 检查是否已有持仓
        position = self.cache.position(instrument_id)
        if position is not None:
            return

        # 检查持仓数量限制
        open_positions = len(self.cache.positions_open())
        if open_positions >= self.config.max_positions:
            return

        # 计算仓位大小
        account = self.portfolio.account(instrument_id.venue)
        if account is None:
            return

        equity = float(account.balance_total(instrument.quote_currency))
        entry_price = float(bar.close)
        position_value = equity * self.config.position_size_pct
        quantity = instrument.make_qty(position_value / entry_price)

        # 计算止损止盈价格
        stop_loss_price = instrument.make_price(
            entry_price * (1 - self.config.stop_loss_pct)
        )
        take_profit_price = instrument.make_price(
            entry_price * (1 + self.config.take_profit_pct)
        )

        # 创建 Bracket Order (入场 + 止损 + 止盈)
        order_list = self.order_factory.bracket(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            sl_trigger_price=stop_loss_price,
            tp_price=take_profit_price,
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order_list(order_list)

        self.log.info(
            f"Submitted LONG bracket order: {quantity} {symbol} @ ~{entry_price:.4f}, "
            f"SL={stop_loss_price}, TP={take_profit_price}"
        )

    def _close_position_if_exists(self, symbol: str, reason: str) -> None:
        """平仓 (如有)"""
        instrument_id = InstrumentId.from_str(f"{symbol}.{self.config.exchange}")
        position = self.cache.position(instrument_id)

        if position is not None and not position.is_closed:
            side = OrderSide.SELL if position.is_long else OrderSide.BUY
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=position.quantity,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.log.info(f"Closing {symbol}: {reason}")

    def on_stop(self) -> None:
        """策略停止"""
        for bar_type in list(self._subscribed_bars):
            self.unsubscribe_bars(bar_type)
        self._subscribed_bars.clear()
        self.log.info("BreakoutStrategy stopped")
