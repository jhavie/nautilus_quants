"""Unit tests for ReportGenerator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nautilus_quants.backtest.config import ReportConfig, TearsheetConfig
from nautilus_quants.backtest.reports import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock BacktestEngine."""
        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.accounts.return_value = []
        engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {
            "PnL (total)": 1000.0,
            "PnL% (total)": 0.10,
        }
        engine.portfolio.analyzer.get_performance_stats_returns.return_value = {
            "Sharpe Ratio": 1.5,
            "Max Drawdown": -0.05,
        }
        engine.portfolio.analyzer.get_performance_stats_general.return_value = {
            "Win Rate": 0.6,
        }
        return engine

    @pytest.fixture
    def report_config(self) -> ReportConfig:
        """Create report configuration."""
        return ReportConfig(
            output_dir="logs/backtest_runs",
            formats=["csv", "html"],
            tearsheet=TearsheetConfig(enabled=True),
        )

    def test_init_stores_parameters(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test initialization stores parameters."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        assert generator.engine == mock_engine
        assert generator.output_dir == tmp_path
        assert generator.config == report_config

    def test_generate_csv_reports_creates_files(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_csv_reports creates CSV files."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        reports = generator.generate_csv_reports()

        assert "orders" in reports
        assert "fills" in reports
        assert "positions" in reports
        assert "account" in reports
        assert reports["orders"].exists()
        assert reports["fills"].exists()

    def test_generate_csv_reports_handles_empty_orders(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_csv_reports handles no orders scenario."""
        mock_engine.cache.orders.return_value = []

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        reports = generator.generate_csv_reports()

        # Should still create files with empty indicator
        assert reports["orders"].exists()
        content = reports["orders"].read_text()
        assert "No orders" in content or content == ""

    def test_generate_statistics_returns_combined_stats(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_statistics returns combined statistics."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        stats = generator.generate_statistics()

        assert "PnL (total)" in stats
        assert "Sharpe Ratio" in stats
        assert "Win Rate" in stats
        # Check normalized keys
        assert "total_pnl" in stats
        assert "sharpe_ratio" in stats

    def test_generate_statistics_handles_error(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_statistics returns empty dict on error."""
        mock_engine.portfolio.analyzer.get_performance_stats_pnls.side_effect = Exception(
            "Error"
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        stats = generator.generate_statistics()

        assert stats == {}

    def test_generate_tearsheet_skipped_when_disabled(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_tearsheet skipped when disabled."""
        config = ReportConfig(
            tearsheet=TearsheetConfig(enabled=False),
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_tearsheet()

        assert result is None

    def test_generate_tearsheet_skipped_when_no_config(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_tearsheet skipped when no tearsheet config."""
        config = ReportConfig(tearsheet=None)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_tearsheet()

        assert result is None

    def test_generate_tearsheet_warns_when_plotly_missing(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_tearsheet warns when plotly not installed."""
        import builtins
        import sys

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        # Temporarily remove plotly from sys.modules to simulate it not being installed
        original_modules = sys.modules.copy()
        plotly_modules = [k for k in sys.modules if k.startswith("plotly")]
        for mod in plotly_modules:
            sys.modules.pop(mod, None)

        # Mock the import to raise ImportError
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("No module named 'plotly'")
            return original_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = generator.generate_tearsheet()
            # Should return None without raising
            assert result is None
        finally:
            # Restore modules
            sys.modules.update(original_modules)

    def test_generate_all_calls_all_generators(
        self, mock_engine: MagicMock, tmp_path: Path, report_config: ReportConfig
    ) -> None:
        """Test generate_all calls all report generators."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config,
        )

        with patch.object(generator, "generate_csv_reports") as mock_csv:
            mock_csv.return_value = {"orders": tmp_path / "orders.csv"}
            with patch.object(generator, "generate_tearsheet") as mock_tearsheet:
                mock_tearsheet.return_value = tmp_path / "tearsheet.html"

                reports = generator.generate_all()

        mock_csv.assert_called_once()
        mock_tearsheet.assert_called_once()
        assert "orders" in reports
        assert "tearsheet" in reports
