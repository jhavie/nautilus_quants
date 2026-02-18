"""Unit tests for QuantStats report generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nautilus_quants.backtest.config import QuantStatsConfig, ReportConfig
from nautilus_quants.backtest.reports import ReportGenerator


class TestQuantStatsConfig:
    """Tests for QuantStatsConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QuantStatsConfig()

        assert config.enabled is False
        assert config.title == "QuantStats Report"
        assert config.benchmark is None
        assert config.output_format == ["html"]
        assert "returns" in config.charts
        assert "drawdown" in config.charts

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = QuantStatsConfig(
            enabled=True,
            title="Custom Report",
            benchmark="SPY",
            output_format=["html", "png"],
            charts=["returns", "drawdown"],
        )

        assert config.enabled is True
        assert config.title == "Custom Report"
        assert config.benchmark == "SPY"
        assert config.output_format == ["html", "png"]
        assert config.charts == ["returns", "drawdown"]

    def test_frozen_dataclass(self) -> None:
        """Test that config is immutable."""
        config = QuantStatsConfig()

        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore


class TestReportConfigWithQuantStats:
    """Tests for ReportConfig with QuantStats integration."""

    def test_default_quantstats_is_none(self) -> None:
        """Test that quantstats is None by default."""
        config = ReportConfig()
        assert config.quantstats is None

    def test_with_quantstats_config(self) -> None:
        """Test ReportConfig with QuantStatsConfig."""
        qs_config = QuantStatsConfig(enabled=True)
        config = ReportConfig(quantstats=qs_config)

        assert config.quantstats is not None
        assert config.quantstats.enabled is True


class TestQuantStatsReportGeneration:
    """Tests for QuantStats report generation in ReportGenerator."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock BacktestEngine."""
        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.positions_closed.return_value = []
        engine.cache.position_snapshots.return_value = []
        engine.cache.accounts.return_value = []
        engine.portfolio.analyzer.get_performance_stats_pnls.return_value = {
            "PnL (total)": 1000.0,
        }
        engine.portfolio.analyzer.get_performance_stats_returns.return_value = {
            "Sharpe Ratio": 1.5,
        }
        engine.portfolio.analyzer.get_performance_stats_general.return_value = {
            "Win Rate": 0.6,
        }
        return engine

    @pytest.fixture
    def quantstats_config(self) -> QuantStatsConfig:
        """Create QuantStats configuration."""
        return QuantStatsConfig(
            enabled=True,
            title="Test Report",
            output_format=["html"],
            charts=["returns", "drawdown"],
        )

    @pytest.fixture
    def report_config_with_quantstats(
        self, quantstats_config: QuantStatsConfig
    ) -> ReportConfig:
        """Create report configuration with QuantStats."""
        return ReportConfig(
            output_dir="logs/backtest_runs",
            formats=["csv"],
            quantstats=quantstats_config,
        )

    def test_generate_quantstats_reports_skipped_when_disabled(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_quantstats_reports skipped when disabled."""
        config = ReportConfig(
            quantstats=QuantStatsConfig(enabled=False),
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_quantstats_reports()

        assert result == {}

    def test_generate_quantstats_reports_skipped_when_no_config(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate_quantstats_reports skipped when no quantstats config."""
        config = ReportConfig(quantstats=None)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        result = generator.generate_quantstats_reports()

        assert result == {}

    def test_generate_quantstats_reports_warns_when_not_installed(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        report_config_with_quantstats: ReportConfig,
    ) -> None:
        """Test generate_quantstats_reports warns when quantstats not installed."""
        import builtins
        import sys

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config_with_quantstats,
        )

        # Temporarily remove quantstats from sys.modules
        original_modules = sys.modules.copy()
        qs_modules = [k for k in sys.modules if k.startswith("quantstats")]
        for mod in qs_modules:
            sys.modules.pop(mod, None)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "quantstats" or name.startswith("quantstats."):
                raise ImportError("No module named 'quantstats'")
            return original_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = generator.generate_quantstats_reports()
            # Should return empty dict without raising
            assert result == {}
        finally:
            sys.modules.update(original_modules)

    def test_generate_quantstats_reports_handles_empty_returns(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        report_config_with_quantstats: ReportConfig,
    ) -> None:
        """Test generate_quantstats_reports handles empty returns gracefully."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config_with_quantstats,
        )

        # Mock _get_returns_series to return empty Series
        with patch.object(generator, "_get_returns_series") as mock_returns:
            mock_returns.return_value = pd.Series(dtype=float)

            result = generator.generate_quantstats_reports()

        # Should return empty dict for empty returns
        assert result == {}

    def test_get_returns_series_returns_pandas_series(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _get_returns_series returns pandas Series with daily returns from equity."""
        config = ReportConfig(quantstats=QuantStatsConfig(enabled=True))

        # Setup mock account
        mock_account = MagicMock()
        mock_engine.cache.accounts.return_value = [mock_account]

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        # Create mock account report with equity data (spans multiple days)
        # Use 'total' column as that's the actual column name in account reports
        equity_data = pd.DataFrame(
            {"total": [10000.0, 10100.0, 10050.0, 10200.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        # Mock ReportProvider.generate_account_report at the import location
        with patch(
            "nautilus_trader.analysis.ReportProvider"
        ) as MockReportProvider:
            MockReportProvider.generate_account_report.return_value = equity_data

            result = generator._get_returns_series()

        assert isinstance(result, pd.Series)
        # Result should be daily returns calculated from equity pct_change
        assert len(result) == 3  # 4 days - 1 for pct_change
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_generate_quantstats_html_creates_file(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        quantstats_config: QuantStatsConfig,
    ) -> None:
        """Test _generate_quantstats_html creates HTML file."""
        config = ReportConfig(quantstats=quantstats_config)

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        returns = pd.Series(
            [0.01, -0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        # Mock quantstats module
        mock_qs = MagicMock()
        mock_qs.reports.html = MagicMock()

        result = generator._generate_quantstats_html(mock_qs, returns, quantstats_config)

        assert result is not None
        assert result.name == "quantstats_report.html"
        mock_qs.reports.html.assert_called_once()

    def test_generate_all_includes_quantstats(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        report_config_with_quantstats: ReportConfig,
    ) -> None:
        """Test generate_all includes quantstats reports."""
        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=report_config_with_quantstats,
        )

        with patch.object(generator, "generate_csv_reports") as mock_csv:
            mock_csv.return_value = {}
            with patch.object(generator, "generate_tearsheet") as mock_tearsheet:
                mock_tearsheet.return_value = None
                with patch.object(
                    generator, "generate_quantstats_reports"
                ) as mock_quantstats:
                    mock_quantstats.return_value = {
                        "quantstats_html": tmp_path / "quantstats_report.html"
                    }

                    reports = generator.generate_all()

        mock_quantstats.assert_called_once()
        assert "quantstats_html" in reports


class TestQuantStatsChartGeneration:
    """Tests for QuantStats PNG chart generation."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create mock BacktestEngine."""
        engine = MagicMock()
        engine.cache.positions_closed.return_value = []
        engine.cache.position_snapshots.return_value = []
        return engine

    def test_generate_quantstats_charts_creates_directory(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _generate_quantstats_charts creates charts directory."""
        config = ReportConfig(
            quantstats=QuantStatsConfig(
                enabled=True,
                output_format=["png"],
                charts=["returns"],
            )
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        returns = pd.Series(
            [0.01, -0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        # Mock quantstats module
        mock_qs = MagicMock()
        mock_fig = MagicMock()
        mock_qs.plots.returns = MagicMock(return_value=mock_fig)

        with patch("matplotlib.pyplot.close"):
            generator._generate_quantstats_charts(mock_qs, returns, config.quantstats)

        charts_dir = tmp_path / "quantstats_charts"
        assert charts_dir.exists()

    def test_generate_quantstats_charts_skips_unavailable_charts(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _generate_quantstats_charts skips charts not in chart_functions."""
        config = ReportConfig(
            quantstats=QuantStatsConfig(
                enabled=True,
                output_format=["png"],
                charts=["nonexistent_chart", "returns"],
            )
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        returns = pd.Series(
            [0.01, -0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        # Mock quantstats module
        mock_qs = MagicMock()
        mock_fig = MagicMock()
        mock_qs.plots.returns = MagicMock(return_value=mock_fig)

        with patch("matplotlib.pyplot.close"):
            # Should not raise for nonexistent chart
            result = generator._generate_quantstats_charts(
                mock_qs, returns, config.quantstats
            )

        # Only returns chart should be attempted
        assert isinstance(result, dict)

    def test_generate_quantstats_charts_skips_beta_without_benchmark(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        """Test _generate_quantstats_charts skips rolling_beta without benchmark."""
        config = ReportConfig(
            quantstats=QuantStatsConfig(
                enabled=True,
                benchmark=None,  # No benchmark
                output_format=["png"],
                charts=["rolling_beta"],
            )
        )

        generator = ReportGenerator(
            engine=mock_engine,
            output_dir=tmp_path,
            config=config,
        )

        returns = pd.Series(
            [0.01, -0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        mock_qs = MagicMock()

        result = generator._generate_quantstats_charts(mock_qs, returns, config.quantstats)

        # rolling_beta should be skipped without benchmark
        assert "quantstats_rolling_beta" not in result


class TestGetReturnsSeriesNoInfValues:
    """End-to-end test verifying _get_returns_series never contains inf."""

    def test_get_returns_series_no_inf_values(self, tmp_path: Path) -> None:
        """Verify _get_returns_series output never contains inf values."""
        import pickle

        engine = MagicMock()
        engine.cache.orders.return_value = []
        engine.cache.positions.return_value = []
        engine.cache.accounts.return_value = []

        # Simulate equity points with near-zero dip (root cause of +603% bug)
        base_ns = pd.Timestamp("2024-11-01").value
        day_ns = 86400 * 1_000_000_000
        equity_points = [
            (base_ns, 10000.0),
            (base_ns + day_ns, 5000.0),
            (base_ns + 2 * day_ns, 100.0),
            (base_ns + 3 * day_ns, 0.01),     # Near-zero equity
            (base_ns + 4 * day_ns, 5000.0),    # "Recovery" → would be +inf
            (base_ns + 5 * day_ns, 8000.0),
        ]
        engine.cache.get.return_value = pickle.dumps(equity_points)

        config = ReportConfig(formats=["csv"])
        generator = ReportGenerator(
            engine=engine,
            output_dir=tmp_path,
            config=config,
        )

        returns = generator._get_returns_series()

        # Must not contain any inf values
        assert not returns.isin([float("inf"), float("-inf")]).any()
        # Should still have valid return data
        assert not returns.empty
