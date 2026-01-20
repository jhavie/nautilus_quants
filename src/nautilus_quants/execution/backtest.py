"""
回测入口

使用 Nautilus BacktestEngine 运行回测
支持从 parquet 文件加载数据
使用 Nautilus 原生 Reports 和 Visualization
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Nautilus Analysis imports
from nautilus_trader.analysis import PortfolioAnalyzer
from nautilus_trader.analysis import ReportProvider
from nautilus_trader.analysis import create_tearsheet

from nautilus_quants.core.config import load_config


# =============================================================================
# Data Loading
# =============================================================================


def load_parquet_bars(
    file_path: Path,
    instrument_id: InstrumentId,
    bar_spec: BarSpecification,
) -> list[Bar]:
    """
    从 parquet 文件加载 K 线数据并转换为 Nautilus Bar 对象

    Parameters
    ----------
    file_path : Path
        Parquet 文件路径
    instrument_id : InstrumentId
        交易对 ID
    bar_spec : BarSpecification
        K 线规格 (如 1-HOUR-LAST)

    Returns
    -------
    list[Bar]
        Nautilus Bar 对象列表
    """
    df = pd.read_parquet(file_path)

    # 确保 datetime 是索引
    if "datetime" in df.columns:
        df = df.set_index("datetime")

    # 构建 BarType
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )

    bars = []
    for idx, row in df.iterrows():
        # 转换时间戳为纳秒
        if isinstance(idx, pd.Timestamp):
            ts_ns = idx.value  # 纳秒
        else:
            ts_ns = pd.Timestamp(idx).value

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['open']:.8f}"),
            high=Price.from_str(f"{row['high']:.8f}"),
            low=Price.from_str(f"{row['low']:.8f}"),
            close=Price.from_str(f"{row['close']:.8f}"),
            volume=Quantity.from_str(f"{row['volume']:.8f}"),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)

    return bars


def create_crypto_perpetual(
    symbol: str,
    venue: Venue,
) -> CryptoPerpetual:
    """
    创建加密货币永续合约 Instrument

    Parameters
    ----------
    symbol : str
        交易对符号 (如 BTCUSDT)
    venue : Venue
        交易所

    Returns
    -------
    CryptoPerpetual
        合约 Instrument
    """
    instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")

    # 从符号推断基础货币和报价货币
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        quote = "USDT"
    elif symbol.endswith("USD"):
        base = symbol[:-3]
        quote = "USD"
    else:
        base = symbol[:3]
        quote = symbol[3:]

    return CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=instrument_id.symbol,
        base_currency=base,
        quote_currency=quote,
        settlement_currency=quote,
        is_inverse=False,
        price_precision=8,
        size_precision=8,
        price_increment=Price.from_str("0.00000001"),
        size_increment=Quantity.from_str("0.00000001"),
        max_quantity=Quantity.from_str("10000000"),
        min_quantity=Quantity.from_str("0.00000001"),
        max_notional=None,
        min_notional=Money(10, USDT),
        max_price=Price.from_str("1000000"),
        min_price=Price.from_str("0.00000001"),
        margin_init=Decimal("0.05"),  # 5% 初始保证金 (20x 杠杆)
        margin_maint=Decimal("0.025"),  # 2.5% 维持保证金
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
    )


# =============================================================================
# Backtest Engine Setup
# =============================================================================


def create_backtest_engine(
    initial_capital: float = 100_000.0,
    leverage: float = 3.0,
) -> BacktestEngine:
    """
    创建回测引擎

    Parameters
    ----------
    initial_capital : float
        初始资金
    leverage : float
        杠杆倍数

    Returns
    -------
    BacktestEngine
        配置好的回测引擎
    """
    config = BacktestEngineConfig(
        logging=LoggingConfig(log_level="INFO"),
    )

    engine = BacktestEngine(config=config)

    # 添加交易所
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(initial_capital, USDT)],
        default_leverage=Decimal(str(leverage)),
    )

    return engine


# =============================================================================
# Reports and Visualization
# =============================================================================


def generate_reports(
    engine: BacktestEngine,
    output_dir: Path,
) -> dict:
    """
    生成回测报告

    使用 Nautilus 原生的 ReportProvider 和 PortfolioAnalyzer

    Parameters
    ----------
    engine : BacktestEngine
        已运行完成的回测引擎
    output_dir : Path
        报告输出目录

    Returns
    -------
    dict
        包含各类报告 DataFrame 的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取订单、持仓、账户数据
    orders = engine.cache.orders()
    positions = engine.cache.positions()
    account = engine.cache.account_for_venue(Venue("BINANCE"))

    reports = {}

    # 1. 订单报告
    orders_report = ReportProvider.generate_orders_report(list(orders))
    if not orders_report.empty:
        orders_report.to_csv(output_dir / "orders_report.csv")
        reports["orders"] = orders_report
        print(f"✓ Orders report: {len(orders_report)} orders")

    # 2. 成交报告
    fills_report = ReportProvider.generate_fills_report(list(orders))
    if not fills_report.empty:
        fills_report.to_csv(output_dir / "fills_report.csv")
        reports["fills"] = fills_report
        print(f"✓ Fills report: {len(fills_report)} fills")

    # 3. 持仓报告
    positions_report = ReportProvider.generate_positions_report(list(positions))
    if not positions_report.empty:
        positions_report.to_csv(output_dir / "positions_report.csv")
        reports["positions"] = positions_report
        print(f"✓ Positions report: {len(positions_report)} positions")

    # 4. 账户报告
    if account:
        account_report = ReportProvider.generate_account_report(account)
        if not account_report.empty:
            account_report.to_csv(output_dir / "account_report.csv")
            reports["account"] = account_report
            print(f"✓ Account report: {len(account_report)} entries")

    # 5. 性能统计
    analyzer = engine.portfolio.analyzer
    stats_pnls = analyzer.get_performance_stats_pnls()
    stats_returns = analyzer.get_performance_stats_returns()
    stats_general = analyzer.get_performance_stats_general()

    reports["stats_pnls"] = stats_pnls
    reports["stats_returns"] = stats_returns
    reports["stats_general"] = stats_general

    print("\n=== Performance Statistics ===")
    for name, value in stats_pnls.items():
        print(f"  {name}: {value}")
    for name, value in stats_returns.items():
        print(f"  {name}: {value}")
    for name, value in stats_general.items():
        print(f"  {name}: {value}")

    return reports


def generate_tearsheet(
    engine: BacktestEngine,
    output_path: Path,
    title: str = "Nautilus Quants - Backtest Results",
) -> None:
    """
    生成交互式 HTML Tearsheet

    使用 Nautilus 原生的 create_tearsheet 函数

    Parameters
    ----------
    engine : BacktestEngine
        已运行完成的回测引擎
    output_path : Path
        HTML 文件输出路径
    title : str
        报告标题
    """
    try:
        create_tearsheet(
            engine=engine,
            output_path=str(output_path),
            title=title,
        )
        print(f"✓ Tearsheet saved to: {output_path}")
    except ImportError as e:
        print(f"⚠ Tearsheet generation skipped: {e}")
        print("  Install plotly: pip install plotly")


# =============================================================================
# Main Backtest Runner
# =============================================================================


def run_backtest(
    data_dir: str | Path = "/Users/joe/Sync/nautilus_quant/data/factors/ohlcv",
    config_dir: str | Path = "config",
    output_dir: str | Path = "data/reports",
    symbols: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestEngine:
    """
    运行回测

    Parameters
    ----------
    data_dir : str | Path
        OHLCV parquet 数据目录
    config_dir : str | Path
        配置文件目录
    output_dir : str | Path
        报告输出目录
    symbols : list[str], optional
        要回测的币种列表，默认使用 BTCUSDT, ETHUSDT
    start_date : str, optional
        开始日期 (YYYY-MM-DD)
    end_date : str, optional
        结束日期 (YYYY-MM-DD)

    Returns
    -------
    BacktestEngine
        回测引擎 (可用于进一步分析)
    """
    data_dir = Path(data_dir)
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)

    # 加载配置
    if config_dir.exists():
        configs = load_config(config_dir)
        trading_cfg = configs.get("trading", {})
        initial_capital = trading_cfg.get("capital", {}).get("initial", 100_000.0)
        leverage = trading_cfg.get("leverage", {}).get("default", 3.0)
    else:
        initial_capital = 100_000.0
        leverage = 3.0

    # 默认币种
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    print("=" * 60)
    print("Nautilus Quants - Backtest")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Initial capital: {initial_capital:,.2f} USDT")
    print(f"Leverage: {leverage}x")
    print(f"Symbols: {symbols}")
    print("=" * 60)

    # 创建回测引擎
    engine = create_backtest_engine(
        initial_capital=initial_capital,
        leverage=leverage,
    )

    venue = Venue("BINANCE")

    # K线规格: 1小时
    bar_spec = BarSpecification(
        step=1,
        aggregation=BarAggregation.HOUR,
        price_type=PriceType.LAST,
    )

    # 加载数据
    all_bars = []
    for symbol in symbols:
        file_path = data_dir / f"{symbol}_1h.parquet"
        if not file_path.exists():
            print(f"⚠ File not found: {file_path}")
            continue

        # 创建 Instrument
        instrument = create_crypto_perpetual(symbol, venue)
        engine.add_instrument(instrument)

        # 加载 K 线数据
        bars = load_parquet_bars(
            file_path=file_path,
            instrument_id=instrument.id,
            bar_spec=bar_spec,
        )

        # 时间过滤
        if start_date or end_date:
            start_ns = pd.Timestamp(start_date).value if start_date else 0
            end_ns = pd.Timestamp(end_date).value if end_date else float("inf")
            bars = [b for b in bars if start_ns <= b.ts_event <= end_ns]

        all_bars.extend(bars)
        print(f"✓ Loaded {len(bars)} bars for {symbol}")

    if not all_bars:
        print("❌ No data loaded!")
        return engine

    # 按时间排序
    all_bars.sort(key=lambda b: b.ts_event)

    # 添加数据到引擎
    engine.add_data(all_bars)

    print(f"\nTotal bars: {len(all_bars)}")
    print(f"Time range: {pd.Timestamp(all_bars[0].ts_event)} -> {pd.Timestamp(all_bars[-1].ts_event)}")

    # TODO: 添加 Actor 和 Strategy
    # 这里暂时只运行数据回放，不添加策略
    # 策略添加示例:
    # from nautilus_quants.actors.screening import ScreeningActor, ScreeningActorConfig
    # from nautilus_quants.strategies.breakout import BreakoutStrategy, BreakoutStrategyConfig
    #
    # screening_config = ScreeningActorConfig(...)
    # engine.add_actor(ScreeningActor(screening_config))
    #
    # strategy_config = BreakoutStrategyConfig(...)
    # engine.add_strategy(BreakoutStrategy(strategy_config))

    # 运行回测
    print("\n🚀 Running backtest...")
    engine.run()

    # 生成报告
    print("\n📊 Generating reports...")
    generate_reports(engine, output_dir)

    # 生成 Tearsheet
    tearsheet_path = output_dir / "tearsheet.html"
    generate_tearsheet(engine, tearsheet_path)

    print("\n✅ Backtest completed!")
    return engine


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import sys

    # 解析命令行参数
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/joe/Sync/nautilus_quant/data/factors/ohlcv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/reports"

    engine = run_backtest(
        data_dir=data_dir,
        output_dir=output_dir,
        symbols=["BTCUSDT", "ETHUSDT"],
    )
