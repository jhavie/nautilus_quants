"""Backtest module - Configuration-driven backtesting with nautilus_trader.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/factor/backtest.yaml for example usage.
"""

from nautilus_quants.backtest.config import (
    BacktestResult,
    QuantStatsConfig,
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
from nautilus_quants.backtest.protocols import (
    EQUITY_SNAPSHOTS_CACHE_KEY,
    EXECUTION_STATES_CACHE_KEY,
    POSITION_METADATA_CACHE_KEY,
    BaseMetadataRenderer,
    ColumnConfig,
    MetadataRenderer,
    PositionMetadataProvider,
)
from nautilus_quants.backtest.registry import RendererRegistry
from nautilus_quants.backtest.reports import ReportGenerator

__all__ = [
    # Main classes
    "BacktestResult",
    "ReportGenerator",
    # Config dataclasses (project-specific)
    "ReportConfig",
    "TearsheetConfig",
    "QuantStatsConfig",
    # Protocols and Registry
    "EQUITY_SNAPSHOTS_CACHE_KEY",
    "EXECUTION_STATES_CACHE_KEY",
    "POSITION_METADATA_CACHE_KEY",
    "PositionMetadataProvider",
    "MetadataRenderer",
    "BaseMetadataRenderer",
    "ColumnConfig",
    "RendererRegistry",
    # Exceptions
    "BacktestError",
    "BacktestConfigError",
    "BacktestDataError",
    "BacktestStrategyError",
    "BacktestExecutionError",
    "BacktestReportError",
]
