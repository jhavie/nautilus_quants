"""Backtest configuration dataclasses.

This module contains project-specific configuration classes that extend
Nautilus Trader's native configuration system.

For Actor-decoupled backtesting, use Nautilus native BacktestRunConfig directly.
See: config/factor/backtest.yaml for example usage.

Retained Classes (project-specific, no Nautilus equivalent):
    - TearsheetConfig: Report visualization settings
    - ReportConfig: Report generation settings
    - BacktestResult: Backtest execution result container

Removed Classes (use Nautilus native equivalents):
    - StrategyConfig -> ImportableStrategyConfig
    - BacktestDataConfig -> nautilus_trader.config.BacktestDataConfig
    - FillModelConfig -> ImportableFillModelConfig
    - FeeModelConfig -> ImportableFeeModelConfig
    - LatencyModelConfig -> ImportableLatencyModelConfig
    - VenueConfig -> BacktestVenueConfig
    - LoggingSettings -> LoggingConfig
    - BacktestConfig -> BacktestRunConfig
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TearsheetConfig:
    """Tearsheet visualization configuration."""

    enabled: bool = True
    title: str = "Backtest Results"
    theme: str = "plotly_dark"  # plotly_white | plotly_dark | nautilus
    height: int = 1500  # Total height in pixels
    show_logo: bool = True
    include_benchmark: bool = False
    benchmark_name: str = "Benchmark"
    charts: list[str] = field(
        default_factory=lambda: [
            "run_info",
            "stats_table",
            "equity",
            "drawdown",
            "monthly_returns",
            "distribution",
            "rolling_sharpe",
            "yearly_returns",
        ]
    )
    chart_args: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantStatsConfig:
    """QuantStats report configuration.

    Configures the generation of QuantStats-style HTML reports and PNG charts.
    This is complementary to the Nautilus native tearsheet.
    """

    enabled: bool = False
    title: str = "QuantStats Report"
    benchmark: str | None = None  # Optional benchmark ticker (e.g., "SPY")
    output_format: list[str] = field(default_factory=lambda: ["html"])  # html, png, or both
    charts: list[str] = field(
        default_factory=lambda: [
            "returns",
            "log_returns",
            "yearly_returns",
            "monthly_heatmap",
            "drawdown",
            "rolling_sharpe",
            "rolling_volatility",
        ]
    )


@dataclass(frozen=True)
class PositionVisualizationConfig:
    """Position visualization configuration for ECharts HTML report.

    Generates an interactive HTML chart showing equity curve and position timeline
    with long/short position details on hover.

    Attributes:
        metadata_renderer: Name of the MetadataRenderer to use for position
            metadata. Options: "base" (default), "cross_sectional".
            Use "cross_sectional" for strategies that store rank/composite info.
    """

    enabled: bool = True  # Default enabled
    title: str = "Position Timeline"
    output_subdir: str = "echarts"  # Subdirectory name, outputs to {output_dir}/{output_subdir}/
    chart_height: int = 500
    interval: str = "4h"  # Sampling interval, default 4 hours
    metadata_renderer: str | None = None  # None = use "base" default


@dataclass(frozen=True)
class ReportConfig:
    """Report generation configuration."""

    output_dir: str = "logs/backtest_runs"  # Base output directory
    formats: list[str] = field(default_factory=lambda: ["csv", "html"])
    tearsheet: TearsheetConfig | None = None
    quantstats: QuantStatsConfig | None = None
    position_viz: PositionVisualizationConfig | None = None


@dataclass
class BacktestResult:
    """Backtest execution result."""

    run_id: str
    success: bool
    output_dir: Path
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    statistics: dict[str, Any]  # Combined stats from PortfolioAnalyzer
    reports: dict[str, Path]  # Map of report type to file path
    errors: list[str] = field(default_factory=list)

    @property
    def quantstats_html_path(self) -> Path | None:
        """Path to QuantStats HTML report if generated."""
        return self.reports.get("quantstats_html")

    @property
    def tearsheet_path(self) -> Path | None:
        """Path to HTML tearsheet if generated."""
        return self.reports.get("tearsheet")

    @property
    def position_viz_path(self) -> Path | None:
        """Path to ECharts position visualization if generated."""
        return self.reports.get("position_viz")

    @property
    def total_pnl(self) -> float:
        """Total PnL from statistics."""
        return float(self.statistics.get("total_pnl", 0.0))

    @property
    def total_pnl_pct(self) -> float:
        """Total PnL percentage from statistics."""
        return float(self.statistics.get("total_pnl_pct", 0.0))

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio from statistics."""
        return float(self.statistics.get("sharpe_ratio", 0.0))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from statistics."""
        return float(self.statistics.get("max_drawdown", 0.0))

    @property
    def win_rate(self) -> float:
        """Win rate from statistics."""
        return float(self.statistics.get("win_rate", 0.0))
