"""
配置加载器

从 YAML 文件加载配置到 dataclass
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class TradingConfig:
    """交易配置"""

    initial_capital: float = 100_000.0
    exchange: str = "BINANCE"
    market_type: str = "FUTURES"
    max_positions: int = 10
    max_single_pct: float = 0.10
    default_leverage: float = 3.0


@dataclass
class StrategyConfig:
    """策略配置"""

    # 选币参数
    top_n: int = 10
    volume_period: int = 7
    min_volume_usd: float = 5_000_000.0

    # 信号参数
    breakout_period: int = 60
    sma_period: int = 200

    # 候选币种
    candidate_symbols: Optional[List[str]] = None


@dataclass
class RiskConfig:
    """风控配置"""

    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_daily_loss_pct: float = 0.10


def load_config(config_dir: str | Path) -> dict:
    """
    加载所有配置文件

    Parameters
    ----------
    config_dir : str | Path
        配置文件目录路径

    Returns
    -------
    dict
        包含 trading, strategy, risk, backtest 的配置字典
    """
    config_dir = Path(config_dir)
    configs = {}

    for name in ["trading", "strategy", "risk", "backtest"]:
        file_path = config_dir / f"{name}.yaml"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                configs[name] = yaml.safe_load(f)
        else:
            configs[name] = {}

    return configs
