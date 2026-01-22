"""Backtest configuration dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from nautilus_quants.backtest.exceptions import BacktestConfigError


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy configuration."""

    type: str  # Registry key (e.g., "breakout")
    instrument_id: str  # Symbol without exchange (e.g., "BTCUSDT")
    interval: str = "1h"  # Strategy timeframe (e.g., "1h", "4h", "1d")
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestDataConfig:
    """Backtest data configuration."""

    start_date: str  # YYYY-MM-DD format
    end_date: str  # YYYY-MM-DD format
    catalog_path: str  # Nautilus ParquetDataCatalog path (required)
    bar_spec: str = "1m"  # Data source timeframe (e.g., "1m", "1h")
    instruments: list[str] = field(default_factory=list)  # Instruments to load
    warmup_days: int = 0  # Extra days for indicator warmup


@dataclass(frozen=True)
class FillModelConfig:
    """Fill simulation configuration."""

    prob_fill_on_limit: float = 1.0  # [0.0, 1.0]
    prob_fill_on_stop: float = 1.0  # [0.0, 1.0]
    prob_slippage: float = 0.0  # [0.0, 1.0]
    random_seed: int | None = None  # For reproducibility


@dataclass(frozen=True)
class FeeModelConfig:
    """Fee model configuration."""

    type: str = "maker_taker"  # maker_taker | fixed | per_contract
    maker_fee: float = 0.0002  # For maker_taker type
    taker_fee: float = 0.0004  # For maker_taker type
    fixed_commission: float | None = None  # For fixed type


@dataclass(frozen=True)
class LatencyModelConfig:
    """Latency simulation configuration."""

    base_latency_ms: int = 0  # Base network latency
    insert_latency_ms: int = 0  # Order insertion latency
    update_latency_ms: int = 0  # Order update latency
    cancel_latency_ms: int = 0  # Order cancellation latency


@dataclass(frozen=True)
class VenueConfig:
    """Venue (exchange) configuration."""

    name: str = "BINANCE"  # Venue identifier
    oms_type: str = "NETTING"  # NETTING | HEDGING
    account_type: str = "MARGIN"  # CASH | MARGIN
    base_currency: str = "USDT"  # Account base currency
    starting_balance: str = "100000 USDT"  # Initial balance
    default_leverage: int = 1  # Leverage for margin accounts
    fill_model: FillModelConfig | None = None
    fee_model: FeeModelConfig | None = None
    latency_model: LatencyModelConfig | None = None


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
class ReportConfig:
    """Report generation configuration."""

    output_dir: str = "logs/backtest_runs"  # Base output directory
    formats: list[str] = field(default_factory=lambda: ["csv", "html"])
    tearsheet: TearsheetConfig | None = None


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True  # Write to {run_id}/nautilus.log
    bypass_logging: bool = False  # Disable console output


@dataclass(frozen=True)
class BacktestConfig:
    """Complete backtest configuration."""

    strategy: StrategyConfig
    backtest: BacktestDataConfig
    venue: VenueConfig
    report: ReportConfig
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BacktestConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            BacktestConfig instance

        Raises:
            BacktestConfigError: If loading or parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise BacktestConfigError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise BacktestConfigError(f"Invalid YAML syntax: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            BacktestConfig instance

        Raises:
            BacktestConfigError: If validation fails
        """
        try:
            # Parse nested configs
            strategy = StrategyConfig(**data.get("strategy", {}))
            backtest = BacktestDataConfig(**data.get("backtest", {}))

            # Parse venue with nested models
            venue_data = data.get("venue", {})
            if "fill_model" in venue_data and venue_data["fill_model"]:
                venue_data["fill_model"] = FillModelConfig(**venue_data["fill_model"])
            if "fee_model" in venue_data and venue_data["fee_model"]:
                venue_data["fee_model"] = FeeModelConfig(**venue_data["fee_model"])
            if "latency_model" in venue_data and venue_data["latency_model"]:
                venue_data["latency_model"] = LatencyModelConfig(
                    **venue_data["latency_model"]
                )
            venue = VenueConfig(**venue_data)

            # Parse report with nested tearsheet
            report_data = data.get("report", {})
            if "tearsheet" in report_data and report_data["tearsheet"]:
                report_data["tearsheet"] = TearsheetConfig(**report_data["tearsheet"])
            report = ReportConfig(**report_data)

            # Parse logging
            logging_data = data.get("logging", {})
            logging = LoggingSettings(**logging_data)

            config = cls(
                strategy=strategy,
                backtest=backtest,
                venue=venue,
                report=report,
                logging=logging,
            )

            # Validate
            errors = config.validate()
            if errors:
                raise BacktestConfigError(
                    f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )

            return config

        except TypeError as e:
            raise BacktestConfigError(f"Invalid configuration structure: {e}") from e

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Strategy validation
        if not self.strategy.type:
            errors.append("strategy.type: Required field missing")
        if not self.strategy.instrument_id:
            errors.append("strategy.instrument_id: Required field missing")
        if "." in self.strategy.instrument_id:
            errors.append(
                f"strategy.instrument_id: Should not contain exchange suffix. "
                f"Use '{self.strategy.instrument_id.split('.')[0]}' instead"
            )

        # Backtest data validation
        if not self.backtest.start_date:
            errors.append("backtest.start_date: Required field missing")
        if not self.backtest.end_date:
            errors.append("backtest.end_date: Required field missing")
        if not self.backtest.catalog_path:
            errors.append("backtest.catalog_path: Required field missing")

        # Date validation
        try:
            start = datetime.strptime(self.backtest.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.backtest.end_date, "%Y-%m-%d")
            if end < start:
                errors.append("backtest.end_date: Must be >= start_date")
        except ValueError:
            errors.append(
                "backtest.start_date/end_date: Invalid date format. Use YYYY-MM-DD"
            )

        # Venue validation
        valid_oms_types = {"NETTING", "HEDGING"}
        if self.venue.oms_type.upper() not in valid_oms_types:
            errors.append(
                f"venue.oms_type: Must be one of {valid_oms_types}"
            )

        valid_account_types = {"CASH", "MARGIN"}
        if self.venue.account_type.upper() not in valid_account_types:
            errors.append(
                f"venue.account_type: Must be one of {valid_account_types}"
            )

        # Fill model validation
        if self.venue.fill_model:
            fm = self.venue.fill_model
            for field_name in ["prob_fill_on_limit", "prob_fill_on_stop", "prob_slippage"]:
                value = getattr(fm, field_name)
                if not 0.0 <= value <= 1.0:
                    errors.append(
                        f"venue.fill_model.{field_name}: Must be in [0.0, 1.0]"
                    )

        # Logging validation
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if self.logging.level.upper() not in valid_levels:
            errors.append(f"logging.level: Must be one of {valid_levels}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        from dataclasses import asdict

        return asdict(self)

    def to_nautilus_data_configs(self) -> list:
        """Convert to nautilus BacktestDataConfig list.

        Returns:
            List of nautilus_trader BacktestDataConfig objects
        """
        from datetime import timedelta

        from nautilus_trader.config import BacktestDataConfig as NautilusDataConfig
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.identifiers import InstrumentId

        from nautilus_quants.backtest.utils.bar_spec import format_bar_spec

        configs = []
        instruments = self.backtest.instruments or [self.strategy.instrument_id]

        # Calculate start with warmup
        start_date = datetime.strptime(self.backtest.start_date, "%Y-%m-%d")
        if self.backtest.warmup_days > 0:
            start_date = start_date - timedelta(days=self.backtest.warmup_days)
        start_str = start_date.strftime("%Y-%m-%d")

        # End date (add 1 day to include full end day)
        end_date = datetime.strptime(self.backtest.end_date, "%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_str = end_date.strftime("%Y-%m-%d")

        # Format bar spec for nautilus
        bar_spec = format_bar_spec(self.backtest.bar_spec, internal=False)

        for symbol in instruments:
            instrument_id = f"{symbol}.{self.venue.name}"
            configs.append(
                NautilusDataConfig(
                    catalog_path=self.backtest.catalog_path,
                    data_cls=Bar,
                    instrument_id=InstrumentId.from_str(instrument_id),
                    bar_spec=bar_spec,
                    start_time=start_str,
                    end_time=end_str,
                )
            )
        return configs

    def to_nautilus_venue_config(self):
        """Convert to nautilus BacktestVenueConfig.

        Returns:
            nautilus_trader BacktestVenueConfig object
        """
        from nautilus_trader.config import BacktestVenueConfig as NautilusVenueConfig

        return NautilusVenueConfig(
            name=self.venue.name,
            oms_type=self.venue.oms_type,
            account_type=self.venue.account_type,
            base_currency=self.venue.base_currency,
            starting_balances=[self.venue.starting_balance],
            default_leverage=self.venue.default_leverage,
        )


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
    def tearsheet_path(self) -> Path | None:
        """Path to HTML tearsheet if generated."""
        return self.reports.get("tearsheet")

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
