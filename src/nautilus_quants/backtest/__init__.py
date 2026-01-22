"""Backtest module - Configuration-driven backtesting with nautilus_trader."""

from nautilus_quants.backtest.config import (
    BacktestConfig,
    BacktestDataConfig,
    BacktestResult,
    FeeModelConfig,
    FillModelConfig,
    LatencyModelConfig,
    LoggingSettings,
    ReportConfig,
    StrategyConfig,
    TearsheetConfig,
    VenueConfig,
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
from nautilus_quants.backtest.runner import BacktestRunner

__all__ = [
    # Main classes
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "ReportGenerator",
    # Config dataclasses
    "StrategyConfig",
    "BacktestDataConfig",
    "VenueConfig",
    "FillModelConfig",
    "FeeModelConfig",
    "LatencyModelConfig",
    "ReportConfig",
    "TearsheetConfig",
    "LoggingSettings",
    # Exceptions
    "BacktestError",
    "BacktestConfigError",
    "BacktestDataError",
    "BacktestStrategyError",
    "BacktestExecutionError",
    "BacktestReportError",
]
