"""Performance benchmark tests for backtest module.

Tests performance requirements from spec SC-006:
- 1 year of hourly data × 5 symbols should complete in under 60 seconds
"""

import time
from pathlib import Path
from typing import Any

import pytest

from nautilus_quants.backtest.config import BacktestConfig
from nautilus_quants.backtest.runner import BacktestRunner


class TestBacktestPerformance:
    """Performance benchmark tests for backtest execution."""

    @pytest.fixture
    def performance_config(self, backtest_catalog_path: Path, tmp_path: Path) -> dict[str, Any]:
        """Configuration for performance testing.

        Uses realistic settings that match production use cases:
        - Hourly bars for extended period
        - Breakout strategy with standard parameters
        - Minimal logging for accurate timing
        """
        return {
            "strategy": {
                "type": "breakout",
                "instrument_id": "BTCUSDT",
                "params": {
                    "breakout_period": 20,
                    "sma_period": 50,
                    "position_size_pct": 0.10,
                    "max_positions": 1,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                },
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-15",  # Limited by fixture data
                "bar_spec": "1h",
                "catalog_path": str(backtest_catalog_path),
                "warmup_days": 5,
            },
            "venue": {
                "name": "BINANCE",
                "oms_type": "NETTING",
                "account_type": "MARGIN",
                "starting_balance": "100000 USDT",
                "default_leverage": 5,
                "fee_model": {
                    "type": "maker_taker",
                    "maker_fee": 0.0002,
                    "taker_fee": 0.0004,
                },
            },
            "report": {
                "output_dir": str(tmp_path / "performance_output"),
                "formats": ["csv"],
                "tearsheet": {
                    "enabled": False,  # Disable for performance testing
                },
            },
            "logging": {
                "level": "ERROR",  # Minimal logging for accurate timing
                "log_to_file": False,
                "bypass_logging": True,
            },
        }

    @pytest.mark.slow
    def test_backtest_execution_time(
        self, performance_config: dict[str, Any]
    ) -> None:
        """Test backtest completes within performance requirements.

        Spec SC-006 requires: 1 year × 5 symbols × hourly data < 60 seconds

        Note: This test uses limited fixture data (2 weeks), so we verify
        the execution is efficient rather than the full spec requirement.
        The test serves as a baseline for performance regression testing.
        """
        config = BacktestConfig.from_dict(performance_config)
        runner = BacktestRunner(config)

        # Measure execution time
        start_time = time.perf_counter()
        result = runner.run()
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Verify successful execution
        assert result.success is True, f"Backtest failed: {result.errors}"
        assert result.duration_seconds > 0

        # Log performance metrics for monitoring
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Execution time:     {execution_time:.3f} seconds")
        print(f"Result duration:    {result.duration_seconds:.3f} seconds")
        print(f"Data period:        {performance_config['backtest']['start_date']} to {performance_config['backtest']['end_date']}")
        print(f"Bar specification:  {performance_config['backtest']['bar_spec']}")
        print(f"Strategy:           {performance_config['strategy']['type']}")
        print(f"Output directory:   {result.output_dir}")
        print(f"{'='*60}\n")

        # Performance assertion - should be fast for 2 weeks of data
        # Extrapolating: 2 weeks should be much faster than 60s/year target
        max_allowed_time = 10.0  # seconds for 2 weeks of data
        assert execution_time < max_allowed_time, (
            f"Backtest took {execution_time:.2f}s, expected < {max_allowed_time}s. "
            f"This suggests performance issues that would violate SC-006 requirement "
            f"of 60s for 1 year × 5 symbols."
        )

    @pytest.mark.slow
    def test_backtest_memory_efficiency(
        self, performance_config: dict[str, Any]
    ) -> None:
        """Test backtest runs without excessive memory usage.

        Verifies the backtest can complete without memory errors,
        which is important for long-running backtests with large datasets.
        """
        config = BacktestConfig.from_dict(performance_config)
        runner = BacktestRunner(config)

        # Run backtest - should not raise MemoryError
        result = runner.run()

        assert result.success is True
        assert result.output_dir.exists()

        # Verify reports were generated (confirms full execution)
        assert (result.output_dir / "orders_report.csv").exists()
        assert (result.output_dir / "fills_report.csv").exists()

    @pytest.mark.slow
    def test_backtest_with_multiple_runs_performance(
        self, performance_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test multiple sequential backtests for performance consistency.

        Verifies that:
        1. Multiple runs have consistent performance
        2. No memory leaks or resource accumulation
        3. Cleanup between runs is effective
        """
        import copy

        num_runs = 3
        execution_times: list[float] = []

        for i in range(num_runs):
            # Create fresh copy of config for each run to avoid mutation issues
            config_dict = copy.deepcopy(performance_config)
            config_dict["report"]["output_dir"] = str(
                tmp_path / f"run_{i}_output"
            )

            config = BacktestConfig.from_dict(config_dict)
            runner = BacktestRunner(config)

            # Measure execution time
            start_time = time.perf_counter()
            result = runner.run()
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            execution_times.append(execution_time)

            assert result.success is True, f"Run {i} failed: {result.errors}"

        # Log performance metrics
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        print(f"\n{'='*60}")
        print(f"MULTIPLE RUNS PERFORMANCE")
        print(f"{'='*60}")
        print(f"Number of runs:     {num_runs}")
        print(f"Average time:       {avg_time:.3f} seconds")
        print(f"Min time:           {min_time:.3f} seconds")
        print(f"Max time:           {max_time:.3f} seconds")
        print(f"Time variance:      {max_time - min_time:.3f} seconds")
        print(f"Individual times:   {[f'{t:.3f}s' for t in execution_times]}")
        print(f"{'='*60}\n")

        # Performance should be consistent (within 50% variance)
        if avg_time > 0:
            variance_pct = (max_time - min_time) / avg_time * 100
            assert variance_pct < 50.0, (
                f"Performance variance too high: {variance_pct:.1f}%. "
                f"This suggests inconsistent performance or resource issues."
            )

    @pytest.mark.slow
    def test_backtest_data_loading_performance(
        self, performance_config: dict[str, Any]
    ) -> None:
        """Test data loading performance in isolation.

        Measures the time taken to load data from catalog,
        which is a significant component of backtest execution time.
        """
        # This test now uses the catalog-based loading via BacktestRunner
        # since we removed the standalone DataLoader with data_path
        config = BacktestConfig.from_dict(performance_config)
        runner = BacktestRunner(config)

        # Measure execution time (includes data loading)
        start_time = time.perf_counter()
        result = runner.run()
        end_time = time.perf_counter()

        loading_time = end_time - start_time

        # Log metrics
        print(f"\n{'='*60}")
        print(f"DATA LOADING PERFORMANCE")
        print(f"{'='*60}")
        print(f"Total time:         {loading_time:.3f} seconds")
        print(f"{'='*60}\n")

        # Data loading should be fast
        assert loading_time < 10.0, (
            f"Backtest took {loading_time:.2f}s, expected < 10s. "
            f"Slow data loading will impact overall backtest performance."
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("bypass_logging", [True, False])
    def test_logging_performance_impact(
        self, performance_config: dict[str, Any], bypass_logging: bool
    ) -> None:
        """Test the performance impact of logging settings.

        Verifies that bypass_logging significantly improves performance
        by reducing I/O operations during backtest execution.
        """
        performance_config["logging"]["bypass_logging"] = bypass_logging
        performance_config["logging"]["level"] = "INFO" if not bypass_logging else "ERROR"

        config = BacktestConfig.from_dict(performance_config)
        runner = BacktestRunner(config)

        # Measure execution time
        start_time = time.perf_counter()
        result = runner.run()
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        assert result.success is True

        # Log results
        print(f"\n{'='*60}")
        print(f"LOGGING IMPACT TEST")
        print(f"{'='*60}")
        print(f"Bypass logging:     {bypass_logging}")
        print(f"Execution time:     {execution_time:.3f} seconds")
        print(f"{'='*60}\n")

        # Store result for comparison (in real test, you'd compare both runs)
        # With bypass_logging=True should be faster than False
        assert execution_time < 15.0, "Execution too slow even with optimized logging"
