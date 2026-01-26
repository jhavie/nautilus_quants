"""Backtest module - Configuration-driven backtesting with nautilus_trader.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/backtest_factor.yaml for example usage.
"""

from nautilus_quants.backtest.config import (
    BacktestResult,
    ReportConfig,
    TearsheetConfig,
)
from nautilus_quants.backtest.exceptions import (
    BacktestConfigError,
    BacktestDataError,
    BacktestError,
    BacktestExecutionError,
    BacktestReportError,
    BacktestStrategyError,
)
from nautilus_quants.backtest.reports import ReportGenerator

__all__ = [
    # Main classes
    "BacktestResult",
    "ReportGenerator",
    # Config dataclasses (project-specific)
    "ReportConfig",
    "TearsheetConfig",
    # Exceptions
    "BacktestError",
    "BacktestConfigError",
    "BacktestDataError",
    "BacktestStrategyError",
    "BacktestExecutionError",
    "BacktestReportError",
]
