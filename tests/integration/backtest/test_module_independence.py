"""Test module independence - backtest module should be independently importable."""

import sys


class TestModuleIndependence:
    """Tests for backtest module independence."""

    def test_backtest_module_imports_independently(self) -> None:
        """Verify backtest module can be imported without other nautilus_quants modules."""
        # Import backtest module
        from nautilus_quants.backtest import BacktestResult, ReportConfig, ReportGenerator

        # Verify basic classes are available
        assert ReportGenerator is not None
        assert ReportConfig is not None
        assert BacktestResult is not None

        # Verify core backtest submodules are loaded
        loaded = {k for k in sys.modules if k.startswith("nautilus_quants.backtest")}
        assert any("backtest" in m for m in loaded), "backtest submodules should be loaded"

    def test_backtest_config_imports_without_runner(self) -> None:
        """Verify config classes can be imported without runner."""
        # Import only config - now uses project-specific report configs
        # Note: BacktestDataConfig, StrategyConfig, VenueConfig removed in favor of Nautilus native
        from nautilus_quants.backtest.config import (
            BacktestResult,
            QuantStatsConfig,
            ReportConfig,
            TearsheetConfig,
        )

        assert BacktestResult is not None
        assert ReportConfig is not None
        assert TearsheetConfig is not None
        assert QuantStatsConfig is not None

    def test_utilities_import_independently(self) -> None:
        """Verify utility modules can be imported independently."""
        # Import utilities directly
        from nautilus_quants.backtest.utils.bar_spec import parse_bar_spec
        from nautilus_quants.backtest.utils.reporting import generate_run_id

        assert parse_bar_spec is not None
        assert generate_run_id is not None
