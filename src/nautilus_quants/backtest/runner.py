"""Backtest runner - orchestrates backtest execution using high-level API."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import (
    BacktestEngineConfig,
    BacktestRunConfig,
    ImportableStrategyConfig,
    LoggingConfig,
)

from nautilus_quants.backtest.config import BacktestConfig, BacktestResult
from nautilus_quants.backtest.exceptions import BacktestStrategyError
from nautilus_quants.backtest.reports import ReportGenerator
from nautilus_quants.backtest.utils.reporting import (
    create_output_directory,
    generate_run_id,
)

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine


class BacktestRunner:
    """Backtest execution orchestrator using high-level API.

    Uses BacktestRunConfig + BacktestNode for proper timestamp handling
    in tearsheet generation.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize runner.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.run_id: str = ""
        self.output_dir: Path = Path()
        self._engine: "BacktestEngine | None" = None
        self._node: BacktestNode | None = None

    def run(self) -> BacktestResult:
        """Execute backtest and generate reports.

        Returns:
            BacktestResult with execution outcome

        Raises:
            BacktestDataError: If data loading fails
            BacktestStrategyError: If strategy instantiation fails
            BacktestExecutionError: If backtest execution fails
        """
        start_time = datetime.now()
        errors: list[str] = []

        try:
            # Initialize run
            self.run_id = generate_run_id()
            self.output_dir = create_output_directory(
                self.config.report.output_dir, self.run_id
            )

            # Build run config using high-level API
            run_config = self._build_run_config()

            # Create and run node
            self._node = BacktestNode(configs=[run_config])
            self._node.run()

            # Get engine from node
            engines = self._node.get_engines()
            if not engines:
                raise RuntimeError("No engines returned from BacktestNode")
            self._engine = engines[0]

            # Generate reports
            report_generator = ReportGenerator(
                engine=self._engine,
                output_dir=self.output_dir,
                config=self.config.report,
            )
            reports = report_generator.generate_all()
            statistics = report_generator.generate_statistics()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return BacktestResult(
                run_id=self.run_id,
                success=True,
                output_dir=self.output_dir,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                statistics=statistics,
                reports=reports,
                errors=errors,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            errors.append(str(e))

            return BacktestResult(
                run_id=self.run_id or "FAILED",
                success=False,
                output_dir=self.output_dir or Path(),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                statistics={},
                reports={},
                errors=errors,
            )

        finally:
            # Cleanup
            if self._engine:
                self._engine.dispose()

    def _build_run_config(self) -> BacktestRunConfig:
        """Build BacktestRunConfig from our config.

        Returns:
            Configured BacktestRunConfig for BacktestNode
        """
        # Build logging config
        logging_config = self._build_logging_config()

        # Build strategy config
        strategy_config = self._build_strategy_config()

        # Build engine config
        engine_config = BacktestEngineConfig(
            strategies=[strategy_config],
            logging=logging_config,
        )

        # Build data configs using high-level API
        data_configs = self.config.to_nautilus_data_configs()

        # Build venue config using high-level API
        venue_config = self.config.to_nautilus_venue_config()

        return BacktestRunConfig(
            engine=engine_config,
            venues=[venue_config],
            data=data_configs,
        )

    def _build_logging_config(self) -> LoggingConfig:
        """Build LoggingConfig for nautilus.

        Returns:
            Configured LoggingConfig
        """
        settings = self.config.logging
        should_log_to_file = settings.log_to_file and self.output_dir

        return LoggingConfig(
            log_level=settings.level.upper(),
            log_level_file=settings.level.upper() if should_log_to_file else None,
            log_directory=str(self.output_dir) if should_log_to_file else None,
            log_file_name="nautilus.log" if should_log_to_file else None,
            bypass_logging=settings.bypass_logging,
        )

    def _build_strategy_config(self) -> ImportableStrategyConfig:
        """Build ImportableStrategyConfig from strategy registry.

        Returns:
            Configured ImportableStrategyConfig

        Raises:
            BacktestStrategyError: If strategy type not found
        """
        from nautilus_quants.strategies import STRATEGY_REGISTRY

        strategy_cfg = self.config.strategy

        # Validate strategy exists
        if strategy_cfg.type not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise BacktestStrategyError(
                f"Unknown strategy type '{strategy_cfg.type}'. Available: {available}"
            )

        strategy_class, config_class = STRATEGY_REGISTRY[strategy_cfg.type]

        # Build strategy params
        params = {**strategy_cfg.params}

        # Add exchange from venue config
        if "exchange" not in params:
            params["exchange"] = self.config.venue.name

        # Pass timeframe info
        params["bar_spec"] = self.config.backtest.bar_spec
        params["interval"] = self.config.strategy.interval

        # Add instruments
        backtest_instruments = self.config.backtest.instruments
        if backtest_instruments:
            params["instruments"] = tuple(backtest_instruments)
        else:
            params["instruments"] = (strategy_cfg.instrument_id,)

        # Get strategy module paths
        strategy_module = strategy_class.__module__
        strategy_name = strategy_class.__name__
        config_module = config_class.__module__
        config_name = config_class.__name__

        return ImportableStrategyConfig(
            strategy_path=f"{strategy_module}:{strategy_name}",
            config_path=f"{config_module}:{config_name}",
            config=params,
        )
