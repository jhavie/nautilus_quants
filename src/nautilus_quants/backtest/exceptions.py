"""Backtest module exceptions."""


class BacktestError(Exception):
    """Base exception for backtest module."""

    pass


class BacktestConfigError(BacktestError):
    """Configuration validation or loading error."""

    pass


class BacktestDataError(BacktestError):
    """Data loading or validation error."""

    pass


class BacktestStrategyError(BacktestError):
    """Strategy instantiation or execution error."""

    pass


class BacktestExecutionError(BacktestError):
    """Backtest engine execution error."""

    pass


class BacktestReportError(BacktestError):
    """Report generation error."""

    pass
